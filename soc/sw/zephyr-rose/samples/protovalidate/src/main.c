/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE protocol-layer validation: reproduce the reqrsp RX result using the
 * transport-neutral protocol API (rose_request_camera + rose_recv_reqrsp) rather
 * than raw tx/rx. The protocol layer strips the [header, num_bytes] framing, so
 * buf[0] is the first served data word.
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <rose/rose_proto.h>

#define PATTERN_BASE 0xC0DE0000u
#define N          16
#define RX_CHANNEL 2 /* reqrsp1 */

int main(void)
{
	const struct device *rose = DEVICE_DT_GET_ONE(ucbbar_roseadapter);
	uint32_t buf[N];
	int n;
	int ok;

	if (!device_is_ready(rose)) {
		printk("PROTOVALIDATE: FAIL rose device not ready\n");
		return 1;
	}

	rose_request_camera(rose, ROSE_CMD_CAMERA_LEFT);

	n = rose_recv_reqrsp(rose, RX_CHANNEL, buf, N);
	if (n < 2) {
		printk("PROTOVALIDATE: FAIL recv rc/n=%d\n", n);
		return 1;
	}

	ok = 1;
	for (int i = 0; i < n; i++) {
		if (buf[i] != (PATTERN_BASE + (uint32_t)i)) {
			ok = 0;
		}
	}

	if (ok) {
		printk("PROTOVALIDATE: PASS n=%d buf[0]=0x%08x buf[%d]=0x%08x\n",
		       n, buf[0], n - 1, buf[n - 1]);
	} else {
		printk("PROTOVALIDATE: FAIL n=%d buf[0]=0x%08x buf[1]=0x%08x\n", n, buf[0], buf[1]);
	}

	return ok ? 0 : 1;
}
