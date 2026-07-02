/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE co-simulation bridge adapter driver.
 *   - polling TX / reqrsp-RX FIFOs
 *   - interrupt-driven camera-DMA completion
 *
 * Register map (control window base from devicetree `reg`; offsets computed from
 * the port topology exactly as the HW regmap does, see RoSEAdapter.scala and
 * soc/sw/generated-src/rose_c_header/rose_port.h). For the standard
 * RoseTLRocketMMIOOnlyConfig (2 reqrsp + 1 DMA):
 *
 *   +0x00 STATUS   (RO)  bit0 TX_ENQ_READY, bit1 RX_VALID(ch2),
 *                        bit2 RX_VALID(ch1), bit3 DMA_BUFFER(ch0)
 *   +0x08 TX_DATA  (WO)  enqueue a TX word
 *   +0x0c RX_DATA_1(RO)  dequeue reqrsp0 (channel 1)
 *   +0x10 RX_DATA_2(RO)  dequeue reqrsp1 (channel 2)
 *   +0x14 DMA_CFG_0(WO)  DMA0 buffer size (bytes)
 *   +0x18 DMA_CUR_0(RO)  DMA0 progress
 *   +0x20 INT_PEND (W1C)  DMA-completion pending bits (one per DMA channel)
 */

#define DT_DRV_COMPAT ucbbar_roseadapter

#include <errno.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sys/sys_io.h>
#include <zephyr/sys/util.h>

#include <rose/rose.h>

/* Fixed register offsets. */
#define ROSE_REG_STATUS  0x00U
#define ROSE_REG_TX_DATA 0x08U
#define ROSE_REG_RX_BASE 0x0cU /* reqrsp index j -> 0x0c + j*4 */

/* STATUS bits. */
#define ROSE_ST_TX_READY  BIT(0)
#define ROSE_ST_RX2_VALID BIT(1) /* channel 2 (reqrsp1) */
#define ROSE_ST_RX1_VALID BIT(2) /* channel 1 (reqrsp0) */

struct rose_config {
	uintptr_t base;
	uintptr_t dma_base;
	uint8_t num_reqrsp;
	uint8_t num_dma;
	void (*irq_config)(const struct device *dev);
};

struct rose_data {
	struct k_sem *dma_sem; /* one per DMA channel */
};

static inline uint32_t rose_rd(const struct rose_config *cfg, uint32_t off)
{
	return sys_read32(cfg->base + off);
}

static inline void rose_wr(const struct rose_config *cfg, uint32_t off, uint32_t val)
{
	sys_write32(val, cfg->base + off);
}

/* DMA config-counter register for DMA channel `ch`. */
static inline uint32_t rose_dma_cfg_off(const struct rose_config *cfg, uint8_t ch)
{
	return ROSE_REG_RX_BASE + (uint32_t)(cfg->num_reqrsp + ch) * 4U;
}

/* Interrupt pending (write-1-to-clear) register. */
static inline uint32_t rose_int_pend_off(const struct rose_config *cfg)
{
	return ROSE_REG_RX_BASE + (uint32_t)(cfg->num_reqrsp + 2U * cfg->num_dma + 1U) * 4U;
}

static int rose_tx_impl(const struct device *dev, uint32_t word)
{
	const struct rose_config *cfg = dev->config;

	while (!(rose_rd(cfg, ROSE_REG_STATUS) & ROSE_ST_TX_READY)) {
		/* spin: TX FIFO full */
	}
	rose_wr(cfg, ROSE_REG_TX_DATA, word);
	return 0;
}

static int rose_rx_impl(const struct device *dev, uint8_t channel, uint32_t *word)
{
	const struct rose_config *cfg = dev->config;
	uint32_t data_off;
	uint32_t valid_mask;

	switch (channel) {
	case 1:
		data_off = ROSE_REG_RX_BASE;      /* reqrsp0 */
		valid_mask = ROSE_ST_RX1_VALID;
		break;
	case 2:
		data_off = ROSE_REG_RX_BASE + 4U; /* reqrsp1 */
		valid_mask = ROSE_ST_RX2_VALID;
		break;
	default:
		return -EINVAL;
	}

	while (!(rose_rd(cfg, ROSE_REG_STATUS) & valid_mask)) {
		/* spin: no RX word on this channel */
	}
	*word = rose_rd(cfg, data_off); /* read dequeues (asserts ready) */
	return 0;
}

static bool rose_tx_ready_impl(const struct device *dev)
{
	const struct rose_config *cfg = dev->config;

	return (rose_rd(cfg, ROSE_REG_STATUS) & ROSE_ST_TX_READY) != 0U;
}

static bool rose_rx_ready_impl(const struct device *dev, uint8_t channel)
{
	const struct rose_config *cfg = dev->config;
	uint32_t mask;

	switch (channel) {
	case 1:
		mask = ROSE_ST_RX1_VALID;
		break;
	case 2:
		mask = ROSE_ST_RX2_VALID;
		break;
	default:
		return false;
	}
	return (rose_rd(cfg, ROSE_REG_STATUS) & mask) != 0U;
}

static int rose_dma_arm_impl(const struct device *dev, uint8_t channel, size_t nbytes)
{
	const struct rose_config *cfg = dev->config;
	struct rose_data *data = dev->data;

	if (channel >= cfg->num_dma) {
		return -EINVAL;
	}
	/* discard any stale completion, then program the buffer size */
	k_sem_reset(&data->dma_sem[channel]);
	rose_wr(cfg, rose_dma_cfg_off(cfg, channel), (uint32_t)nbytes);
	return 0;
}

static int rose_dma_wait_impl(const struct device *dev, uint8_t channel, k_timeout_t timeout)
{
	const struct rose_config *cfg = dev->config;
	struct rose_data *data = dev->data;

	if (channel >= cfg->num_dma) {
		return -EINVAL;
	}
	return k_sem_take(&data->dma_sem[channel], timeout);
}

static uintptr_t rose_dma_buffer_impl(const struct device *dev, uint8_t channel)
{
	const struct rose_config *cfg = dev->config;

	ARG_UNUSED(channel); /* single DMA target base for now */
	return cfg->dma_base;
}

static void rose_isr(const struct device *dev)
{
	const struct rose_config *cfg = dev->config;
	struct rose_data *data = dev->data;
	uint32_t off = rose_int_pend_off(cfg);
	uint32_t pend = rose_rd(cfg, off);

	for (uint8_t i = 0; i < cfg->num_dma; i++) {
		if (pend & BIT(i)) {
			k_sem_give(&data->dma_sem[i]);
		}
	}
	rose_wr(cfg, off, pend); /* write-1-to-clear the serviced bits */
}

static const struct rose_driver_api rose_api = {
	.tx = rose_tx_impl,
	.rx = rose_rx_impl,
	.tx_ready = rose_tx_ready_impl,
	.rx_ready = rose_rx_ready_impl,
	.dma_arm = rose_dma_arm_impl,
	.dma_wait = rose_dma_wait_impl,
	.dma_buffer = rose_dma_buffer_impl,
};

static int rose_init(const struct device *dev)
{
	const struct rose_config *cfg = dev->config;
	struct rose_data *data = dev->data;

	for (uint8_t i = 0; i < cfg->num_dma; i++) {
		k_sem_init(&data->dma_sem[i], 0, 1);
	}
	if (cfg->irq_config != NULL) {
		cfg->irq_config(dev);
	}
	return 0;
}

/* Per-instance IRQ wiring, only if the node declares an interrupt. */
#define ROSE_IRQ_CONFIG(inst)                                                  \
	static void rose_irq_config_##inst(const struct device *dev)           \
	{                                                                      \
		IRQ_CONNECT(DT_INST_IRQN(inst), DT_INST_IRQ(inst, priority),    \
			    rose_isr, DEVICE_DT_INST_GET(inst), 0);            \
		irq_enable(DT_INST_IRQN(inst));                                \
	}

#define ROSE_IRQ_INIT(inst)                                                    \
	COND_CODE_1(DT_INST_IRQ_HAS_IDX(inst, 0), (ROSE_IRQ_CONFIG(inst)), ())

#define ROSE_IRQ_PTR(inst)                                                     \
	COND_CODE_1(DT_INST_IRQ_HAS_IDX(inst, 0), (rose_irq_config_##inst), (NULL))

#define ROSE_INIT(inst)                                                        \
	ROSE_IRQ_INIT(inst)                                                     \
	static struct k_sem rose_dma_sems_##inst[DT_INST_PROP_OR(inst, num_dma_channels, 1)]; \
	static struct rose_data rose_data_##inst = {                            \
		.dma_sem = rose_dma_sems_##inst,                               \
	};                                                                     \
	static const struct rose_config rose_config_##inst = {                 \
		.base = DT_INST_REG_ADDR(inst),                                \
		.dma_base = DT_INST_PROP_OR(inst, dma_base_address, 0),        \
		.num_reqrsp = DT_INST_PROP_OR(inst, num_reqrsp_channels, 2),   \
		.num_dma = DT_INST_PROP_OR(inst, num_dma_channels, 1),         \
		.irq_config = ROSE_IRQ_PTR(inst),                             \
	};                                                                     \
	DEVICE_DT_INST_DEFINE(inst, rose_init, NULL, &rose_data_##inst,        \
			      &rose_config_##inst, POST_KERNEL,                \
			      CONFIG_KERNEL_INIT_PRIORITY_DEVICE, &rose_api);

DT_INST_FOREACH_STATUS_OKAY(ROSE_INIT)
