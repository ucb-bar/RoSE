/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE bridge DMA+reqrsp STRESS test. Reproduces the gate-nav flight's mid-flight
 * freeze, which the instrumented flight localized to the bridge/RTL DELIVERY of a
 * channel-crossing reqrsp that immediately follows a large camera DMA:
 *   large DMA 0x11 (ch0, 5400B) -> reqrsp 0x41 (ch1) -> reqrsp 0x42 (ch2 -> STALL).
 * The guest WFI-hangs on the 0x42 response (which the sync served+sent). Isolated
 * DMA->ch2 passes 300 iters; the ch1 reqrsp in between is the missing trigger.
 * Markers before each blocking op pinpoint which op / iter hangs. Pair with:
 *   ROSE_GYM_ENV=PatternEnv-v0 ROSE_PATTERN_CAM_LEN=1400 run_sync_only.py \
 *     --yaml_path config_gym_PatternEnv-camstress.yaml
 */
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/device.h>
#include <rose/rose.h>
#include <rose/rose_proto.h>
#include "rose_check.h"

#ifndef STRESS_ITERS
#define STRESS_ITERS 300
#endif
/* Camera-sized DMA (5400 bytes = 1350 u32), like the cam_front frame. */
#ifndef DMA_WORDS
#define DMA_WORDS 1350
#endif
/* Channel-crossing reqrsp sequence after the DMA (mirrors the flight's vision tick). */
#define REQRSP_CH1_CMD 0x41u  /* tof_cross -> reqrsp channel 1 */
#define REQRSP_CH1     1
#define REQRSP_CH2_CMD 0x42u  /* lowdim    -> reqrsp channel 2  (the response that stalls) */
#define REQRSP_CH2     2

static int do_dma(const struct device *rose)
{
	volatile uint32_t *dma;

	rose_dma_arm(rose, ROSE_TEST_DMA_CH, DMA_WORDS * 4);
	rose_tx(rose, ROSE_TEST_CMD_DMA);
	rose_tx(rose, 0);
	if (rose_dma_wait(rose, ROSE_TEST_DMA_CH, K_FOREVER) != 0) {
		return 0;
	}
	dma = (volatile uint32_t *)rose_dma_buffer(rose, ROSE_TEST_DMA_CH);
	return (dma[0] == ROSE_PATTERN_BASE) &&
	       (dma[DMA_WORDS - 1] == (uint32_t)(ROSE_PATTERN_BASE + DMA_WORDS - 1));
}

static int do_reqrsp_ch(const struct device *rose, uint32_t cmd, uint8_t channel)
{
	uint32_t buf[ROSE_PATTERN_LEN];
	int n;

	rose_request(rose, cmd, 0);
	n = rose_recv_reqrsp(rose, channel, buf, ROSE_PATTERN_LEN);
	return rose_check_pattern("reqrsp", buf, n, ROSE_PATTERN_BASE);
}

int main(void)
{
	const struct device *rose = DEVICE_DT_GET_ONE(ucbbar_roseadapter);
	int fails = 0;

	if (!device_is_ready(rose)) {
		printk("ROSE stress: FAIL (device not ready)\n");
		sys_reboot(SYS_REBOOT_COLD);
		return 1;
	}

	for (int it = 0; it < STRESS_ITERS; it++) {
		int dma_ok, rr1, rr2;

		printk("STRESS iter %d: dma...\n", it);
		dma_ok = do_dma(rose);

		printk("STRESS iter %d: reqrsp_ch1...\n", it);
		rr1 = do_reqrsp_ch(rose, REQRSP_CH1_CMD, REQRSP_CH1);

		/* This ch2 reqrsp — right after the large DMA + ch1 reqrsp — is where the
		 * gate-nav flight froze (guest WFI on the 0x42 response). */
		printk("STRESS iter %d: reqrsp_ch2...\n", it);
		rr2 = do_reqrsp_ch(rose, REQRSP_CH2_CMD, REQRSP_CH2);

		if (!dma_ok || !rr1 || !rr2) {
			printk("STRESS iter %d: FAIL dma=%d ch1=%d ch2=%d\n", it, dma_ok, rr1, rr2);
			fails++;
		}
	}

	printk("ROSE stress: %d iters %d fails => %s\n",
	       STRESS_ITERS, fails, fails ? "FAIL" : "PASS");
	sys_reboot(SYS_REBOOT_COLD);
	return fails ? 1 : 0;
}
