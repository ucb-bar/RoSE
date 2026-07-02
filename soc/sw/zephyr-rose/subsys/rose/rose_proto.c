/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * RoSE co-simulation transport/protocol layer implementation.
 */

#include <errno.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>

#include <rose/rose.h>
#include <rose/rose_proto.h>

int rose_request(const struct device *dev, uint32_t cmd, uint32_t arg)
{
	int rc;

	rc = rose_tx(dev, cmd);
	if (rc != 0) {
		return rc;
	}
	return rose_tx(dev, arg);
}

int rose_recv_reqrsp(const struct device *dev, uint8_t channel, uint32_t *out, size_t max_words)
{
	uint32_t header;
	uint32_t num_bytes;
	size_t nwords;
	int rc;

	if (out == NULL) {
		return -EINVAL;
	}

	/* Framed response: [header (cmd echo)] [num_bytes] [data...] */
	rc = rose_rx(dev, channel, &header);
	if (rc != 0) {
		return rc;
	}
	rc = rose_rx(dev, channel, &num_bytes);
	if (rc != 0) {
		return rc;
	}

	nwords = num_bytes / sizeof(uint32_t);

	for (size_t i = 0; i < nwords; i++) {
		uint32_t w;

		rc = rose_rx(dev, channel, &w);
		if (rc != 0) {
			return rc;
		}
		if (i < max_words) {
			out[i] = w; /* store */
		}
		/* else: drain to keep the FIFO framed */
	}

	return (int)MIN(nwords, max_words);
}

int rose_recv_dma(const struct device *dev, uint8_t channel, uint32_t request_cmd,
		  size_t nbytes, const volatile uint32_t **buf, k_timeout_t timeout)
{
	int rc;

	if (buf == NULL) {
		return -EINVAL;
	}

	/* arm first so we never miss the completion, then issue the request */
	rc = rose_dma_arm(dev, channel, nbytes);
	if (rc != 0) {
		return rc;
	}
	if (request_cmd != 0U) {
		rc = rose_request(dev, request_cmd, 0U);
		if (rc != 0) {
			return rc;
		}
	}
	rc = rose_dma_wait(dev, channel, timeout);
	if (rc != 0) {
		return rc;
	}

	*buf = (const volatile uint32_t *)rose_dma_buffer(dev, channel);
	return 0;
}
