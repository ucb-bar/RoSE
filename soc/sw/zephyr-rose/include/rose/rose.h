/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Public API for the RoSE co-simulation bridge adapter driver.
 *
 * Replaces the ad-hoc reg_read32/reg_write32-on-rose_port.h pattern of the
 * baremetal RoSE apps with a Zephyr device-model API. This initial version is
 * polling-only (TX/RX FIFOs); DMA + interrupt-driven completion is layered on
 * later.
 */

#ifndef ROSE_ROSE_H_
#define ROSE_ROSE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RoSE adapter driver API (device-model function table).
 *
 * Channels follow the generated register map (rose_port.h):
 *   - channel 1 -> reqrsp0 (RX_DATA_1)
 *   - channel 2 -> reqrsp1 (RX_DATA_2)
 * (channel 0 is the DMA path, handled by the DMA API added later.)
 */
typedef int  (*rose_tx_fn)(const struct device *dev, uint32_t word);
typedef int  (*rose_rx_fn)(const struct device *dev, uint8_t channel, uint32_t *word);
typedef bool (*rose_tx_ready_fn)(const struct device *dev);
typedef bool (*rose_rx_ready_fn)(const struct device *dev, uint8_t channel);
/* DMA (camera-buffer) path: channel 0.. */
typedef int       (*rose_dma_arm_fn)(const struct device *dev, uint8_t channel, size_t nbytes);
typedef int       (*rose_dma_wait_fn)(const struct device *dev, uint8_t channel, k_timeout_t timeout);
typedef uintptr_t (*rose_dma_buffer_fn)(const struct device *dev, uint8_t channel);

struct rose_driver_api {
	rose_tx_fn tx;
	rose_rx_fn rx;
	rose_tx_ready_fn tx_ready;
	rose_rx_ready_fn rx_ready;
	rose_dma_arm_fn dma_arm;
	rose_dma_wait_fn dma_wait;
	rose_dma_buffer_fn dma_buffer;
};

/** @brief Enqueue one 32-bit word on the TX FIFO (blocks until space). */
static inline int rose_tx(const struct device *dev, uint32_t word)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->tx(dev, word);
}

/** @brief Dequeue one 32-bit word from an RX (reqrsp) channel (blocks until valid). */
static inline int rose_rx(const struct device *dev, uint8_t channel, uint32_t *word)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->rx(dev, channel, word);
}

/** @brief True if the TX FIFO can accept a word right now. */
static inline bool rose_tx_ready(const struct device *dev)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->tx_ready(dev);
}

/** @brief True if the given RX channel has a word available right now. */
static inline bool rose_rx_ready(const struct device *dev, uint8_t channel)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->rx_ready(dev, channel);
}

/**
 * @brief Arm a camera-DMA channel to fill a buffer of @p nbytes.
 *
 * Programs the DMA buffer size and clears any pending completion. The engine
 * fills its buffer in memory (see rose_dma_buffer()) when the corresponding
 * data is served, then raises the DMA-complete interrupt.
 */
static inline int rose_dma_arm(const struct device *dev, uint8_t channel, size_t nbytes)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->dma_arm(dev, channel, nbytes);
}

/** @brief Block until the DMA channel finishes a buffer (interrupt-driven). */
static inline int rose_dma_wait(const struct device *dev, uint8_t channel, k_timeout_t timeout)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->dma_wait(dev, channel, timeout);
}

/** @brief Physical address of the memory buffer the DMA channel writes to. */
static inline uintptr_t rose_dma_buffer(const struct device *dev, uint8_t channel)
{
	const struct rose_driver_api *api = (const struct rose_driver_api *)dev->api;

	return api->dma_buffer(dev, channel);
}

#ifdef __cplusplus
}
#endif

#endif /* ROSE_ROSE_H_ */
