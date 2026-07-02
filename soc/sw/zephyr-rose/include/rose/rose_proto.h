/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE co-simulation transport/protocol layer (transport-neutral).
 *
 * Sits on top of the rose adapter driver API (rose_tx/rose_rx/rose_dma_*) and
 * encapsulates the RoSE packet protocol: request framing, reqrsp/DMA response
 * framing, and the co-sim command set. Higher layers (a rose-i2c bus controller,
 * or a rose sensor driver exposing sensor.h) marshal through this so the same
 * user-facing API works in co-sim or on real hardware — only the bound
 * devicetree node/bus differs.
 */

#ifndef ROSE_ROSE_PROTO_H_
#define ROSE_ROSE_PROTO_H_

#include <stddef.h>
#include <stdint.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RoSE co-sim command set (mirrors soc/sw/generated-src/rose_c_header/rose_packet.h).
 * Data commands (< 0x80) are issued by the SoC; control commands (>= 0x80) are
 * part of the host driver <-> synchronizer handshake and are not sent from here.
 */
#define ROSE_CMD_CAMERA_STEREO 0x10U
#define ROSE_CMD_CAMERA_LEFT   0x11U
/* Test/demo command routed to a reqrsp channel by the samples/rose config. */
#define ROSE_CMD_DATA_REQRSP   0x12U

/**
 * @brief Issue a request to the co-sim: a command word followed by an argument
 *        (num_bytes / parameter) over the TX FIFO.
 */
int rose_request(const struct device *dev, uint32_t cmd, uint32_t arg);

/**
 * @brief Receive a framed reqrsp response on @p channel.
 *
 * Reads the [header, num_bytes, data...] frame the synchronizer serves: stores up
 * to @p max_words data words into @p out and drains any remainder so the FIFO
 * stays framed.
 *
 * @return number of data words stored (>= 0), or negative errno.
 */
int rose_recv_reqrsp(const struct device *dev, uint8_t channel, uint32_t *out, size_t max_words);

/**
 * @brief Receive a DMA response: arm @p channel for @p nbytes, then block (on the
 *        DMA-complete interrupt) until the buffer is filled.
 *
 * The caller issues the request (e.g. rose_request_camera()) between arming and
 * waiting is handled internally: this arms first, so issue the request, then it
 * waits. On success @p *buf points at the filled DMA buffer in memory.
 *
 * NOTE: arm happens inside; the request must be sent by the caller AFTER arming.
 * Use rose_dma_arm()/rose_request()/rose_dma_wait() directly if finer control is
 * needed. This helper arms + waits and expects the request in between via the
 * @p request_cmd argument (0 to skip).
 *
 * @return 0 on success, negative errno on timeout/error.
 */
int rose_recv_dma(const struct device *dev, uint8_t channel, uint32_t request_cmd,
		  size_t nbytes, const volatile uint32_t **buf, k_timeout_t timeout);

/** @brief Convenience: request a camera frame (num_bytes arg = 0). */
static inline int rose_request_camera(const struct device *dev, uint32_t which)
{
	return rose_request(dev, which, 0U);
}

#ifdef __cplusplus
}
#endif

#endif /* ROSE_ROSE_PROTO_H_ */
