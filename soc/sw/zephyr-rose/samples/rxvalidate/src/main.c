/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE reqrsp RX validation, ported to Zephyr using the rose driver API.
 *
 * Mirrors soc/sw/rose-images/airsim-packettest/airsim-packettest-rxvalidate.c
 * but with no register pokes: request a camera frame over TX, read words back
 * on reqrsp channel 2, and validate the known pattern PatternEnv serves.
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <rose/rose.h>

/* RoSE protocol command (see soc/sw/generated-src/rose_c_header/rose_packet.h). */
#define CS_CAMERA_LEFT 0x11u

#define PATTERN_BASE 0xC0DE0000u
#define NREAD 4 /* header + count + 2 pattern words */
#define NPAT  2 /* validate 2 consecutive pattern words */
#define RX_CHANNEL 2 /* reqrsp1 (ROSE_RX_DATA_2) */

int main(void)
{
	/* compatible "ucbbar,RoseAdapter" -> lowercased C token */
	const struct device *rose = DEVICE_DT_GET_ONE(ucbbar_roseadapter);
	uint32_t buf[NREAD];
	int ok = 0;
	int found_at = -1;

	if (!device_is_ready(rose)) {
		printk("RXVALIDATE: FAIL rose device not ready\n");
		return 1;
	}

	/* 1) request CS_CAMERA_LEFT (num_bytes = 0) */
	rose_tx(rose, CS_CAMERA_LEFT);
	rose_tx(rose, 0);

	/* 2) receive NREAD words from reqrsp channel 2 */
	for (int i = 0; i < NREAD; i++) {
		rose_rx(rose, RX_CHANNEL, &buf[i]);
	}

	/* 3) validate: PATTERN_BASE.. as NPAT consecutive words anywhere in buf */
	for (int s = 0; s + NPAT <= NREAD; s++) {
		int match = 1;

		for (int i = 0; i < NPAT; i++) {
			if (buf[s + i] != (PATTERN_BASE + (uint32_t)i)) {
				match = 0;
				break;
			}
		}
		if (match) {
			ok = 1;
			found_at = s;
			break;
		}
	}

	if (ok) {
		printk("RXVALIDATE: PASS pattern at offset %d (buf=0x%08x,0x%08x,0x%08x,0x%08x)\n",
		       found_at, buf[0], buf[1], buf[2], buf[3]);
	} else {
		printk("RXVALIDATE: FAIL buf=0x%08x,0x%08x,0x%08x,0x%08x\n",
		       buf[0], buf[1], buf[2], buf[3]);
	}

	return ok ? 0 : 1;
}
