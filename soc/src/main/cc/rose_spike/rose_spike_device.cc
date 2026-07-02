/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE co-simulation bridge as a Spike MMIO device plugin.
 *
 * Fuses what FireSim splits into a target RTL adapter + host C++ driver:
 *   - presents the SoC-side RoseAdapter register map (0x2000, see rose_port.h)
 *     to the guest via load()/store();
 *   - runs the RoSE synchronizer protocol internally (rose_sync_client);
 *   - DMAs served camera data into guest DRAM and raises the completion IRQ.
 *
 * Load as a dynamic module:
 *   spike --extlib=librose_spike.so \
 *         --device="rose_bridge,<base>,<irq>,<dma_base>,<nreqrsp>,<ndma>" ... elf
 * Defaults: base=0x2000 irq=3 dma_base=0x88000000 nreqrsp=2 ndma=1
 *
 * Cycle-lockstep: the synchronizer grants one token per physics step and blocks
 * until the bridge acks it. This device acks a grant only after the guest retires
 * step_size cycles (mcycle, IPC=1 proxy), so physics advances one step per
 * step_size guest cycles. This is a functional lockstep (correct compute/physics
 * *rate*), not RTL cycle-accuracy (no pipeline/latency model). See
 * ROSE_SPIKE_BRIDGE_PLAN.md.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <vector>
#include <string>

#include "abstract_device.h"
#include "abstract_interrupt_controller.h"
#include "simif.h"
#include "sim.h"
#include <map>

#include "../rose_sync_client.h"

static bool rose_dbg() {
	static int v = -1;
	if (v < 0) { v = getenv("ROSE_SPIKE_DEBUG") ? 1 : 0; }
	return v == 1;
}
#define RDBG(...) do { if (rose_dbg()) { fprintf(stderr, "[rose_spike] " __VA_ARGS__); fflush(stderr); } } while (0)

/* Register offsets from base (match rose_port.h / RoSEAdapter.scala regmap). */
static const reg_t ROSE_STATUS   = 0x00;
static const reg_t ROSE_TX_DATA  = 0x08;
static const reg_t ROSE_RX_BASE  = 0x0c; /* reqrsp index j -> 0x0c + j*4 */
/* DMA cfg = 0x0c + (nreqrsp+ch)*4 ; int-pending = 0x0c + (nreqrsp+2*ndma+1)*4 */

/* STATUS bits. */
static const uint32_t ST_TX_READY  = 1u << 0;
static const uint32_t ST_RX2_VALID = 1u << 1; /* channel 2 */
static const uint32_t ST_RX1_VALID = 1u << 2; /* channel 1 */
static const uint32_t ST_DMA0_DONE = 1u << 3;

class rose_bridge_t : public abstract_device_t {
public:
	rose_bridge_t(const sim_t *sim, reg_t base, uint32_t irq, reg_t dma_base,
		      int nreqrsp, int ndma)
		: sim(const_cast<sim_t *>(sim)), base(base), irq(irq), dma_base(dma_base),
		  nreqrsp(nreqrsp), ndma(ndma) {
		rx_fifo.resize(nreqrsp + 1); /* index by channel number (0=DMA,1,2..) */
		dma_counter_max = 0;
		dma_done = false;
		tx_state = 0;
		tx_cmd = 0;
		tx_remaining = 0;
		step_size = 0;
		grant_pending = false;
		grant_target = 0;
		last_grant_cycle = 0;
		/* Connect lazily in pump() so device construction never stalls Spike
		 * setup if the synchronizer is not up yet. */
	}

	reg_t size() override { return 0x1000; }

	bool load(reg_t addr, size_t len, uint8_t *bytes) override {
		pump(); /* service the synchronizer on every access */
		uint32_t val = 0;
		if (addr == ROSE_STATUS) {
			val = status();
		} else if (addr == rx_off(1)) {
			val = deq(1);
		} else if (addr == rx_off(2)) {
			val = deq(2);
		} else if (addr == dma_curr_off(0)) {
			val = dma_counter_max; /* simplified: report programmed size */
		} else if (addr == int_pend_off()) {
			/* pending bitmask, one bit per DMA channel (ch0 only for now) */
			val = dma_done ? 1u : 0u;
			RDBG("read INT_PEND -> 0x%x\n", val);
		}
		memcpy(bytes, &val, len < 4 ? len : 4);
		return true;
	}

	bool store(reg_t addr, size_t len, const uint8_t *bytes) override {
		client.ensure_connected(); /* so a TX packet is never sent unconnected */
		uint32_t val = 0;
		memcpy(&val, bytes, len < 4 ? len : 4);
		if (addr == ROSE_TX_DATA) {
			tx_word(val);
		} else if (addr == dma_cfg_off(0)) {
			dma_counter_max = val;
			dma_done = false;
		} else if (addr == int_pend_off()) {
			/* W1C: clear pending + deassert IRQ */
			if (val & 1u) {
				dma_done = false;
				set_irq(false);
			}
		}
		pump();
		return true;
	}

	void tick(reg_t rtc_ticks) override { pump(); }

private:
	/* Retired-cycle counter of hart 0. In functional Spike mcycle advances once
	 * per retired instruction (IPC=1), so it serves as the cycle cost model that
	 * gates the per-step token budget. */
	uint64_t cur_cycles() const {
		if (sim && sim->nprocs() > 0) {
			return sim->get_core((size_t)0)->get_state()->mcycle->read();
		}
		return 0;
	}

	reg_t rx_off(int ch) const { return ROSE_RX_BASE + (reg_t)(ch - 1) * 4; }
	reg_t dma_cfg_off(int i) const { return ROSE_RX_BASE + (reg_t)(nreqrsp + i) * 4; }
	reg_t dma_curr_off(int i) const { return ROSE_RX_BASE + (reg_t)(nreqrsp + ndma + i) * 4; }
	reg_t int_pend_off() const { return ROSE_RX_BASE + (reg_t)(nreqrsp + 2 * ndma + 1) * 4; }

	uint32_t status() {
		uint32_t s = ST_TX_READY;
		if (!rx_fifo[2].empty()) s |= ST_RX2_VALID;
		if (nreqrsp >= 1 && !rx_fifo[1].empty()) s |= ST_RX1_VALID;
		if (dma_done) s |= ST_DMA0_DONE;
		return s;
	}

	uint32_t deq(int ch) {
		if (rx_fifo[ch].empty()) {
			return 0;
		}
		uint32_t w = rx_fifo[ch].front();
		rx_fifo[ch].pop_front();
		return w;
	}

	/* Assemble SoC TX words [cmd][num_bytes][data...] into a synchronizer packet. */
	void tx_word(uint32_t w) {
		if (tx_state == 0) {
			tx_cmd = w;
			tx_state = 1;
		} else if (tx_state == 1) {
			uint32_t nbytes = w;
			tx_remaining = nbytes / 4;
			tx_data.clear();
			if (tx_remaining == 0) {
				RDBG("tx cmd=0x%x nwords=0\n", tx_cmd);
				client.send_tx(tx_cmd, nullptr, 0);
				tx_state = 0;
			} else {
				tx_state = 2;
			}
		} else {
			tx_data.push_back(w);
			if (--tx_remaining == 0) {
				client.send_tx(tx_cmd, tx_data.data(), (uint32_t)tx_data.size());
				tx_state = 0;
			}
		}
	}

	/* Deliver a served data packet to the routed destination. */
	void deliver(const rose_pkt_t &p) {
		int ch = route_for(p.cmd);
		RDBG("deliver cmd=0x%x nwords=%zu -> ch%d\n", p.cmd, p.data.size(), ch);
		if (ch == 0) {
			/* DMA channel: write to guest DRAM, raise completion IRQ.
			 * addr_to_mem is public on the simif_t base (private on sim_t). */
			char *mem = static_cast<simif_t *>(sim)->addr_to_mem(dma_base);
			RDBG("dma write base=0x%lx mem=%p nbytes=%zu\n",
			     (unsigned long)dma_base, (void *)mem, p.data.size() * 4);
			if (mem) {
				memcpy(mem, p.data.data(), p.data.size() * 4);
			}
			dma_done = true;
			set_irq(true);
		} else {
			/* reqrsp: frame as [header, num_bytes, data...] for the driver */
			rx_fifo[ch].push_back(p.cmd);
			rx_fifo[ch].push_back((uint32_t)(p.data.size() * 4));
			for (uint32_t d : p.data) {
				rx_fifo[ch].push_back(d);
			}
		}
	}

	int route_for(uint32_t cmd) {
		auto it = routes.find(cmd);
		return (it != routes.end()) ? it->second : 2; /* default reqrsp ch2 */
	}

	/* Service the synchronizer: apply config, deliver data, satisfy budget. */
	void pump() {
		if (!client.ensure_connected()) {
			return; /* synchronizer not up yet; retry on next access */
		}
		rose_pkt_t p;
		while (client.try_recv(p)) {
			if (!p.is_control) {
				deliver(p);
			} else {
				switch (p.cmd) {
				case ROSE_CS_CFG_ROUTE:
					/* data = [header, channel] */
					if (p.data.size() >= 2) {
						routes[p.data[0]] = (int)p.data[1];
						RDBG("route cmd=0x%x -> ch%d\n", p.data[0], p.data[1]);
					}
					break;
				case ROSE_CS_DEFINE_STEP:
					/* cycles the guest must retire per granted token */
					if (p.data.size() >= 1) {
						step_size = p.data[0];
						RDBG("define_step size=%u\n", step_size);
					}
					break;
				case ROSE_CS_REQ_CYCLES: {
					/* report cycles retired since the current grant started */
					uint32_t cyc = (uint32_t)(cur_cycles() - last_grant_cycle);
					client.send_tx(ROSE_CS_RSP_CYCLES, &cyc, 1);
					break;
				}
				case ROSE_CS_GRANT_TOKEN: {
					/* Cycle-lockstep: don't ack until the guest has retired
					 * step_size cycles. The synchronizer blocks in
					 * check_token_exhaustion() meanwhile, so physics advances
					 * exactly one step per step_size guest cycles. The ack is
					 * emitted below once the budget is spent. */
					last_grant_cycle = cur_cycles();
					if (step_size > 0) {
						grant_pending = true;
						grant_target = last_grant_cycle + step_size;
						RDBG("grant: budget=%u target=%llu\n", step_size,
						     (unsigned long long)grant_target);
					} else {
						/* no step defined yet: degenerate to instant ack */
						client.send_tx(ROSE_CS_RSP_STALL, nullptr, 0);
					}
					break;
				}
				case ROSE_CS_CFG_BW:
				default:
					break; /* functional model: bandwidth not gated */
				}
			}
		}

		/* Complete a granted token once its cycle budget has been retired. This
		 * runs on every pump() (tick / MMIO), including while the guest spins on
		 * STATUS or idles in WFI, so the budget is checked continuously. */
		if (grant_pending && cur_cycles() >= grant_target) {
			grant_pending = false;
			RDBG("grant done at cycle=%llu\n", (unsigned long long)cur_cycles());
			client.send_tx(ROSE_CS_RSP_STALL, nullptr, 0);
		}
	}

	void set_irq(bool level) {
		RDBG("set_irq id=%u level=%d intctrl=%p\n", irq, level ? 1 : 0,
		     sim ? (void *)sim->get_intctrl() : nullptr);
		if (sim && irq) {
			sim->get_intctrl()->set_interrupt_level(irq, level ? 1 : 0);
		}
	}

	sim_t *sim;
	reg_t base, dma_base;
	uint32_t irq;
	int nreqrsp, ndma;
	rose_sync_client_t client;

	std::vector<std::deque<uint32_t>> rx_fifo; /* per channel */
	uint32_t dma_counter_max;
	bool dma_done;
	std::map<uint32_t, int> routes;

	int tx_state;
	uint32_t tx_cmd, tx_remaining;
	std::vector<uint32_t> tx_data;

	/* Cycle-lockstep budget: physics advances one step per granted token, and a
	 * grant is acked only after step_size retired cycles elapse (see pump()). */
	uint32_t step_size;       /* cycles per firesim step (from CS_DEFINE_STEP) */
	bool grant_pending;       /* a token is granted but its budget not yet spent */
	uint64_t grant_target;    /* mcycle value at which the current grant completes */
	uint64_t last_grant_cycle;/* mcycle when the current grant started */
};

/* ---- Spike plugin registration ---- */

static reg_t sarg_u(const std::vector<std::string> &a, size_t i, reg_t def) {
	return (i < a.size() && !a[i].empty()) ? (reg_t)strtoull(a[i].c_str(), nullptr, 0) : def;
}

static rose_bridge_t *rose_bridge_parse(const void *fdt, const sim_t *sim,
					reg_t *base, const std::vector<std::string> &sargs) {
	reg_t b        = sarg_u(sargs, 0, 0x2000);
	uint32_t irq   = (uint32_t)sarg_u(sargs, 1, 3);
	reg_t dma_base = sarg_u(sargs, 2, 0x88000000);
	int nreqrsp    = (int)sarg_u(sargs, 3, 2);
	int ndma       = (int)sarg_u(sargs, 4, 1);
	*base = b;
	return new rose_bridge_t(sim, b, irq, dma_base, nreqrsp, ndma);
}

static std::string rose_bridge_generate_dts(const sim_t *sim,
					    const std::vector<std::string> &sargs) {
	reg_t b = sarg_u(sargs, 0, 0x2000);
	char buf[256];
	snprintf(buf, sizeof(buf),
		 "    rose@%lx {\n"
		 "      compatible = \"ucbbar,RoseAdapter\";\n"
		 "      reg = <0x0 0x%lx 0x0 0x1000>;\n"
		 "    };\n",
		 (unsigned long)b, (unsigned long)b);
	return std::string(buf);
}

REGISTER_DEVICE(rose_bridge, rose_bridge_parse, rose_bridge_generate_dts)
