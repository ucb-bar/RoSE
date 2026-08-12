/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE bridge DMA+reqrsp STRESS test: loops interleaved DMA (ch0) + reqrsp (ch2)
 * requests to reproduce the mid-flight bridge stall the gate-nav co-sim hits
 * (guest WFI-frozen on a reqrsp after a preceding DMA, after ~N iters). Prints a
 * marker BEFORE each blocking op so a hang pinpoints WHICH op (dma_wait vs
 * recv_reqrsp) stalls and after how many iterations. Pair with PatternEnv sync:
 *   ROSE_GYM_ENV=PatternEnv-v0 run_sync_only.py --yaml_path config_gym_PatternEnv-v0.yaml
 * SUCCESS = "ROSE stress: N iters M fails => PASS". A HANG (uartlog frozen at
 * "iter K: <op>...") == the bug reproduced.
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
#define BUF_BYTES (ROSE_PATTERN_LEN * 4)

static int do_dma(const struct device *rose)
{
	volatile uint32_t *dma;
	uint32_t buf[ROSE_PATTERN_LEN];

	rose_dma_arm(rose, ROSE_TEST_DMA_CH, BUF_BYTES);
	rose_tx(rose, ROSE_TEST_CMD_DMA);
	rose_tx(rose, 0);
	if (rose_dma_wait(rose, ROSE_TEST_DMA_CH, K_FOREVER) != 0) {
		return 0;
	}
	dma = (volatile uint32_t *)rose_dma_buffer(rose, ROSE_TEST_DMA_CH);
	for (int i = 0; i < ROSE_PATTERN_LEN; i++) {
		buf[i] = dma[i];
	}
	return rose_check_pattern("dma", buf, ROSE_PATTERN_LEN, ROSE_PATTERN_BASE);
}

static int do_reqrsp(const struct device *rose)
{
	uint32_t buf[ROSE_PATTERN_LEN];
	int n;

	rose_request(rose, ROSE_TEST_CMD_REQRSP, 0);
	n = rose_recv_reqrsp(rose, ROSE_TEST_REQRSP_CH, buf, ROSE_PATTERN_LEN);
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
		int dma_ok, rr_ok;

		/* marker BEFORE the blocking DMA op */
		printk("STRESS iter %d: dma...\n", it);
		dma_ok = do_dma(rose);

		/* marker BEFORE the blocking reqrsp op (this is where the co-sim froze) */
		printk("STRESS iter %d: reqrsp...\n", it);
		rr_ok = do_reqrsp(rose);

		if (!dma_ok || !rr_ok) {
			printk("STRESS iter %d: FAIL dma=%d reqrsp=%d\n", it, dma_ok, rr_ok);
			fails++;
		}
	}

	printk("ROSE stress: %d iters %d fails => %s\n",
	       STRESS_ITERS, fails, fails ? "FAIL" : "PASS");
	sys_reboot(SYS_REBOOT_COLD);
	return fails ? 1 : 0;
}
