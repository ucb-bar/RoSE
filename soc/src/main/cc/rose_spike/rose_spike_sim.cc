/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * rose_spike_sim: a control-inverted Spike harness that runs the RoSE co-sim in
 * TRUE cycle-lockstep with the physics synchronizer.
 *
 * Unlike the passive --extlib device (librose_spike.so), this harness OWNS the
 * step loop: rose_sim_t subclasses sim_t and overrides idle() so the machine only
 * advances when the synchronizer has granted a token, and advances EXACTLY one
 * step_size cycle-budget per grant before acking. This gives a two-sided barrier
 * (neither side free-runs), is deterministic (no wall-clock race), and is
 * multicore-correct because the budget is gated on the CLINT's global `mtime`
 * (not a single hart's mcycle). See ROSE_SPIKE_LOCKSTEP_PLAN.md.
 *
 *   rose_spike_sim [--isa=..] [-p N] [--rose-host H] [--rose-port P]
 *                  [--rose-base 0x2000] [--rose-irq 3] [--rose-dma-base 0x88000000]
 *                  [--rose-nreqrsp 2] [--rose-ndma 1] <elf>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <riscv/sim.h>
#include <riscv/mmu.h>
#include <riscv/devices.h>
#include <riscv/cfg.h>
#include <riscv/processor.h>
#include <riscv/debug_module.h>
#include <riscv/simif.h>
#include <riscv/abstract_interrupt_controller.h>
#include <riscv/platform.h>

#include "../rose_sync_client.h"

static bool rose_dbg() {
	static int v = -1;
	if (v < 0) { v = getenv("ROSE_SPIKE_DEBUG") ? 1 : 0; }
	return v == 1;
}
#define RDBG(...) do { if (rose_dbg()) { fprintf(stderr, "[rose_sim] " __VA_ARGS__); fflush(stderr); } } while (0)

/* RoseAdapter register offsets from base (match rose_port.h / RoSEAdapter.scala). */
static const reg_t ROSE_STATUS  = 0x00;
static const reg_t ROSE_TX_DATA = 0x08;
static const reg_t ROSE_RX_BASE = 0x0c; /* reqrsp j -> 0x0c + j*4 */

/* STATUS bits. */
static const uint32_t ST_TX_READY  = 1u << 0;
static const uint32_t ST_RX2_VALID = 1u << 1; /* channel 2 */
static const uint32_t ST_RX1_VALID = 1u << 2; /* channel 1 */
static const uint32_t ST_DMA0_DONE = 1u << 3;

/* ---------------------------------------------------------------------------
 * rose_bridge_ctrl: protocol + regmap state. Owns the synchronizer socket and
 * all bridge state; the harness (rose_sim_t) drives its step-boundary hooks and
 * the thin MMIO device forwards guest accesses here.
 * ------------------------------------------------------------------------- */
class rose_bridge_ctrl {
public:
	rose_bridge_ctrl(sim_t *sim, reg_t base, uint32_t irq, reg_t dma_base,
			 int nreqrsp, int ndma, const char *host, int port)
		: sim(sim), base(base), irq(irq), dma_base(dma_base),
		  nreqrsp(nreqrsp), ndma(ndma), client(host, port) {
		rx_fifo.resize(nreqrsp + 1);
	}

	/* ---- MMIO surface (called by the thin device) ---- */
	uint32_t reg_load(reg_t off) {
		uint32_t val = 0;
		if (off == ROSE_STATUS)          val = status();
		else if (off == rx_off(1))       val = deq(1);
		else if (off == rx_off(2))       val = deq(2);
		else if (off == dma_curr_off(0)) val = dma_counter_max;
		else if (off == int_pend_off())  val = dma_done ? 1u : 0u;
		return val;
	}

	void reg_store(reg_t off, uint32_t val) {
		if (off == ROSE_TX_DATA)          tx_word(val);
		else if (off == dma_cfg_off(0)) { dma_counter_max = val; dma_done = false; }
		else if (off == int_pend_off()) { if (val & 1u) { dma_done = false; set_irq(false); } }
	}

	/* ---- Step-loop hooks (called by rose_sim_t::idle) ---- */

	/* Drain the socket: deliver served data, apply config, latch a grant. Never
	 * acks a grant here (that is the harness's decision at the step boundary). */
	void pump() {
		if (!client.ensure_connected()) return;
		rose_pkt_t p;
		while (client.try_recv(p)) {
			if (!p.is_control) { deliver(p); continue; }
			switch (p.cmd) {
			case ROSE_CS_CFG_ROUTE:
				if (p.data.size() >= 2) { routes[p.data[0]] = (int)p.data[1];
					RDBG("route cmd=0x%x -> ch%d\n", p.data[0], p.data[1]); }
				break;
			case ROSE_CS_DEFINE_STEP:
				if (p.data.size() >= 1) { step_size = p.data[0];
					RDBG("define_step size=%u\n", step_size); }
				break;
			case ROSE_CS_REQ_CYCLES: {
				uint32_t cyc = last_step_cycles;
				client.send_tx(ROSE_CS_RSP_CYCLES, &cyc, 1);
				break;
			}
			case ROSE_CS_GRANT_TOKEN:
				pending_grant = true;   /* consumed by the harness */
				RDBG("grant received\n");
				break;
			case ROSE_CS_RESET:
				reset_state();
				break;
			case ROSE_CS_CFG_BW:
			default:
				break; /* bandwidth not modeled */
			}
		}
	}

	bool take_grant() { if (pending_grant) { pending_grant = false; return true; } return false; }
	uint32_t cycle_budget() const { return step_size; }

	/* End of a granted step: the guest has already streamed its requests to the
	 * synchronizer as it ran; ack so the synchronizer serves the responses. */
	void ack_step(uint32_t cycles_spent) {
		last_step_cycles = cycles_spent;
		client.send_tx(ROSE_CS_RSP_STALL, nullptr, 0);
		RDBG("ack step (spent ~%u cyc)\n", cycles_spent);
	}

private:
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
		if (rx_fifo[ch].empty()) return 0;
		uint32_t w = rx_fifo[ch].front();
		rx_fifo[ch].pop_front();
		return w;
	}

	void tx_word(uint32_t w) {
		if (tx_state == 0) { tx_cmd = w; tx_state = 1; }
		else if (tx_state == 1) {
			tx_remaining = w / 4; tx_data.clear();
			if (tx_remaining == 0) { client.send_tx(tx_cmd, nullptr, 0); tx_state = 0; }
			else tx_state = 2;
		} else {
			tx_data.push_back(w);
			if (--tx_remaining == 0) {
				client.send_tx(tx_cmd, tx_data.data(), (uint32_t)tx_data.size());
				tx_state = 0;
			}
		}
	}

	void deliver(const rose_pkt_t &p) {
		int ch = route_for(p.cmd);
		RDBG("deliver cmd=0x%x nwords=%zu -> ch%d\n", p.cmd, p.data.size(), ch);
		if (ch == 0) {
			char *mem = static_cast<simif_t *>(sim)->addr_to_mem(dma_base);
			if (mem) memcpy(mem, p.data.data(), p.data.size() * 4);
			dma_done = true;
			set_irq(true);
		} else {
			rx_fifo[ch].push_back(p.cmd);
			rx_fifo[ch].push_back((uint32_t)(p.data.size() * 4));
			for (uint32_t d : p.data) rx_fifo[ch].push_back(d);
		}
	}

	int route_for(uint32_t cmd) {
		auto it = routes.find(cmd);
		return (it != routes.end()) ? it->second : 2;
	}

	void set_irq(bool level) {
		if (sim && irq) sim->get_intctrl()->set_interrupt_level(irq, level ? 1 : 0);
	}

	void reset_state() {
		for (auto &q : rx_fifo) q.clear();
		routes.clear();
		dma_done = false; dma_counter_max = 0;
		tx_state = 0; tx_remaining = 0; tx_data.clear();
		set_irq(false);
		RDBG("reset\n");
	}

	sim_t *sim;
	reg_t base, dma_base;
	uint32_t irq;
	int nreqrsp, ndma;
	rose_sync_client_t client;

	std::vector<std::deque<uint32_t>> rx_fifo;
	uint32_t dma_counter_max = 0;
	bool dma_done = false;
	std::map<uint32_t, int> routes;

	int tx_state = 0;
	uint32_t tx_cmd = 0, tx_remaining = 0;
	std::vector<uint32_t> tx_data;

	uint32_t step_size = 0;
	bool pending_grant = false;
	uint32_t last_step_cycles = 0;
};

/* ---------------------------------------------------------------------------
 * Thin MMIO device: forwards guest loads/stores to the controller.
 * ------------------------------------------------------------------------- */
class rose_mmio_device_t : public abstract_device_t {
public:
	explicit rose_mmio_device_t(rose_bridge_ctrl *ctrl) : ctrl(ctrl) {}
	reg_t size() override { return 0x1000; }
	bool load(reg_t addr, size_t len, uint8_t *bytes) override {
		uint32_t v = ctrl->reg_load(addr);
		memcpy(bytes, &v, len < 4 ? len : 4);
		return true;
	}
	bool store(reg_t addr, size_t len, const uint8_t *bytes) override {
		uint32_t v = 0; memcpy(&v, bytes, len < 4 ? len : 4);
		ctrl->reg_store(addr, v);
		return true;
	}
private:
	rose_bridge_ctrl *ctrl;
};

/* ---------------------------------------------------------------------------
 * rose_sim_t: owns the step loop via the idle() override.
 * ------------------------------------------------------------------------- */
class rose_sim_t : public sim_t {
public:
	rose_sim_t(const cfg_t *cfg, std::vector<std::pair<reg_t, abstract_mem_t*>> mems,
		   const std::vector<std::string> &htif_args,
		   const debug_module_config_t &dm_config,
		   reg_t rbase, uint32_t rirq, reg_t rdma, int nreqrsp, int ndma,
		   const char *rhost, int rport)
		: sim_t(cfg, /*halted*/false, std::move(mems),
			/*plugin_devices*/{}, htif_args, dm_config,
			/*log_path*/nullptr, /*dtb_enabled*/true, /*dtb_file*/nullptr,
			/*socket_enabled*/false, /*cmd_file*/nullptr, /*insn_limit*/std::nullopt),
		  ctrl(this, rbase, rirq, rdma, nreqrsp, ndma, rhost, rport) {
		add_device(rbase, std::make_shared<rose_mmio_device_t>(&ctrl));
	}

	/* Called repeatedly by htif_t::run(). Implements the lockstep loop. */
	void idle() override {
		if (done()) return;

		ctrl.pump();  /* connect + deliver served data + latch config/grant */

		if (!grant_active) {
			if (!ctrl.take_grant()) return;   /* no grant yet -> machine stays frozen */
			grant_active   = true;
			uint32_t budget = ctrl.cycle_budget();
			/* mtime advances 1 per INSNS_PER_RTC_TICK cycles; convert the cycle
			 * budget to a global mtime delta (multicore-safe). */
			uint64_t dmt = budget ? (budget / INSNS_PER_RTC_TICK) : 1;
			budget_target = mtime_now() + dmt;
			step_start_mt = mtime_now();
			RDBG("step start: budget=%u mtime=%llu -> target=%llu\n",
			     budget, (unsigned long long)step_start_mt,
			     (unsigned long long)budget_target);
		}

		/* Advance one quantum. step() ticks the clint (mtime) and round-robins
		 * all harts, so the barrier paces the whole machine, not one core. */
		step(INTERLEAVE);

		if (mtime_now() >= budget_target) {
			uint32_t spent = (uint32_t)((mtime_now() - step_start_mt) * INSNS_PER_RTC_TICK);
			ctrl.ack_step(spent);
			grant_active = false;
		}
	}

private:
	uint64_t mtime_now() { return get_core((size_t)0)->get_state()->time->read(); }

	rose_bridge_ctrl ctrl;
	bool grant_active = false;
	uint64_t budget_target = 0;
	uint64_t step_start_mt = 0;
};

/* ---------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
static std::vector<std::pair<reg_t, abstract_mem_t*>> make_mems(const std::vector<mem_cfg_t> &layout) {
	std::vector<std::pair<reg_t, abstract_mem_t*>> mems;
	for (const auto &c : layout) mems.push_back(std::make_pair(c.get_base(), new mem_t(c.get_size())));
	return mems;
}

int main(int argc, char **argv) {
	const char *isa = nullptr, *priv = nullptr;
	size_t nprocs = 1;
	reg_t rbase = 0x2000, rdma = 0x88000000;
	uint32_t rirq = 3;
	int nreqrsp = 2, ndma = 1, rport = 10001;
	const char *rhost = "localhost";
	std::vector<std::string> htif_args;

	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto val = [&](const char *pfx) -> const char * {
			size_t n = strlen(pfx);
			return (a.rfind(pfx, 0) == 0) ? a.c_str() + n : nullptr;
		};
		const char *v;
		if ((v = val("--isa="))) isa = strdup(v);
		else if ((v = val("--priv="))) priv = strdup(v);
		else if (a == "-p" && i + 1 < argc) nprocs = strtoul(argv[++i], nullptr, 0);
		else if ((v = val("-p"))) nprocs = strtoul(v, nullptr, 0);
		else if ((v = val("--rose-host="))) rhost = strdup(v);
		else if ((v = val("--rose-port="))) rport = (int)strtol(v, nullptr, 0);
		else if ((v = val("--rose-base="))) rbase = strtoull(v, nullptr, 0);
		else if ((v = val("--rose-irq="))) rirq = (uint32_t)strtoul(v, nullptr, 0);
		else if ((v = val("--rose-dma-base="))) rdma = strtoull(v, nullptr, 0);
		else if ((v = val("--rose-nreqrsp="))) nreqrsp = (int)strtol(v, nullptr, 0);
		else if ((v = val("--rose-ndma="))) ndma = (int)strtol(v, nullptr, 0);
		else htif_args.push_back(a);   /* elf + guest args */
	}
	if (htif_args.empty()) {
		fprintf(stderr, "usage: rose_spike_sim [opts] <elf>\n");
		return 1;
	}

	cfg_t cfg;
	if (isa)  cfg.isa  = isa;
	if (priv) cfg.priv = priv;
	std::vector<size_t> hartids;
	for (size_t i = 0; i < nprocs; i++) hartids.push_back(i);
	cfg.hartids = hartids;

	auto mems = make_mems(cfg.mem_layout);
	debug_module_config_t dm_config;

	rose_sim_t s(&cfg, mems, htif_args, dm_config,
		     rbase, rirq, rdma, nreqrsp, ndma, rhost, rport);

	RDBG("rose_spike_sim: nprocs=%zu rose@0x%lx irq=%u dma=0x%lx sync=%s:%d\n",
	     nprocs, (unsigned long)rbase, rirq, (unsigned long)rdma, rhost, rport);

	int rc = s.run();
	for (auto &m : mems) delete m.second;
	return rc;
}
