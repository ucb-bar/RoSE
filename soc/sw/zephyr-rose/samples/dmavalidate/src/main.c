/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE DMA RX validation, ported to Zephyr using the rose driver API.
 *
 * Mirrors soc/sw/rose-images/airsim-packettest/airsim-packettest-dmavalidate.c
 * but interrupt-driven: arm the DMA channel, request a camera frame, then block
 * on the DMA-complete IRQ (no busy-poll of the DMA status bit), and validate the
 * pattern the camera-DMA engine wrote to memory.
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <rose/rose.h>

/* RoSE protocol command (see soc/sw/generated-src/rose_c_header/rose_packet.h). */
#define CS_CAMERA_LEFT 0x11u

#define PATTERN_BASE 0xC0DE0000u
#define N          16          /* PatternEnv serves 16 words */
#define BUF_BYTES  (N * 4)
#define DMA_CHANNEL 0          /* channel 0 = DMA0 */

int main(void)
{
	const struct device *rose = DEVICE_DT_GET_ONE(ucbbar_roseadapter);
	volatile uint32_t *dma;
	uint32_t first = 0, last = 0;
	int ok = 1;
	int rc;

	if (!device_is_ready(rose)) {
		printk("DMAVALIDATE: FAIL rose device not ready\n");
		return 1;
	}

	/* 1) arm DMA0 for N words (clears stale completion, sets buffer size) */
	rose_dma_arm(rose, DMA_CHANNEL, BUF_BYTES);

	/* 2) request CS_CAMERA_LEFT (routed to channel 0 = DMA0 by the synchronizer) */
	rose_tx(rose, CS_CAMERA_LEFT);
	rose_tx(rose, 0);

	/* 3) block until the DMA-complete interrupt fires */
	rc = rose_dma_wait(rose, DMA_CHANNEL, K_FOREVER);
	if (rc != 0) {
		printk("DMAVALIDATE: FAIL dma_wait rc=%d\n", rc);
		return 1;
	}

	/* 4) read the DMA'd buffer from memory and validate the known pattern */
	dma = (volatile uint32_t *)rose_dma_buffer(rose, DMA_CHANNEL);
	for (int i = 0; i < N; i++) {
		uint32_t v = dma[i];

		if (i == 0) {
			first = v;
		}
		if (i == N - 1) {
			last = v;
		}
		if (v != (PATTERN_BASE + (uint32_t)i)) {
			ok = 0;
		}
	}

	if (ok) {
		printk("DMAVALIDATE: PASS rx[0]=0x%08x rx[%d]=0x%08x\n", first, N - 1, last);
	} else {
		printk("DMAVALIDATE: FAIL rx[0]=0x%08x rx[%d]=0x%08x\n", first, N - 1, last);
	}

	return ok ? 0 : 1;
}
