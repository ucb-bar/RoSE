/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual RoSE optical-flow sensor driver (PMW3901-equivalent). Fetches a 2-word
 * [vx, vy] float32 body-frame horizontal-velocity frame from the RoSE bridge and
 * exposes it on the private channels ROSE_SENSOR_CHAN_FLOW_VX/VY. A real Flow-deck
 * build binds DT_ALIAS(flow) to a real PMW3901 driver instead; app code is unchanged.
 */

#define DT_DRV_COMPAT ucbbar_rose_flow

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <rose/rose_proto.h>
#include <rose/rose_sensor.h>

LOG_MODULE_REGISTER(rose_flow, CONFIG_SENSOR_LOG_LEVEL);

#define FLOW_WORDS 2

struct rose_flow_config {
	const struct device *rose;
	uint32_t cmd;
	uint8_t channel;
};

struct rose_flow_data {
	float vx, vy;
	bool pending;
};

/* Two-phase (pipelined): sample_fetch issues the request (TX only); channel_get collects
 * (the blocking read) on first call. See rose_imu.c for the rationale. */
static int rose_flow_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	const struct rose_flow_config *cfg = dev->config;
	struct rose_flow_data *data = dev->data;

	if (chan != SENSOR_CHAN_ALL &&
	    chan != (enum sensor_channel)ROSE_SENSOR_CHAN_FLOW_VX &&
	    chan != (enum sensor_channel)ROSE_SENSOR_CHAN_FLOW_VY) {
		return -ENOTSUP;
	}
	rose_request(cfg->rose, cfg->cmd, 0U);
	data->pending = true;
	return 0;
}

static int rose_flow_channel_get(const struct device *dev, enum sensor_channel chan,
				 struct sensor_value *val)
{
	const struct rose_flow_config *cfg = dev->config;
	struct rose_flow_data *data = dev->data;

	if (data->pending) {
		uint32_t raw[FLOW_WORDS];
		int n = rose_recv_reqrsp(cfg->rose, cfg->channel, raw, FLOW_WORDS);
		if (n < FLOW_WORDS) {
			LOG_ERR("short FLOW read: %d/%d", n, FLOW_WORDS);
			return -EIO;
		}
		memcpy(&data->vx, &raw[0], sizeof(float));
		memcpy(&data->vy, &raw[1], sizeof(float));
		data->pending = false;
	}

	switch ((int)chan) {
	case ROSE_SENSOR_CHAN_FLOW_VX:
		sensor_value_from_double(val, (double)data->vx);
		return 0;
	case ROSE_SENSOR_CHAN_FLOW_VY:
		sensor_value_from_double(val, (double)data->vy);
		return 0;
	default:
		return -ENOTSUP;
	}
}

static const struct sensor_driver_api rose_flow_api = {
	.sample_fetch = rose_flow_sample_fetch,
	.channel_get = rose_flow_channel_get,
};

static int rose_flow_init(const struct device *dev)
{
	const struct rose_flow_config *cfg = dev->config;

	if (!device_is_ready(cfg->rose)) {
		LOG_ERR("rose adapter not ready");
		return -ENODEV;
	}
	return 0;
}

#define ROSE_FLOW_DEFINE(inst)                                                        \
	static struct rose_flow_data rose_flow_data_##inst;                           \
	static const struct rose_flow_config rose_flow_config_##inst = {              \
		.rose = DEVICE_DT_GET(DT_INST_PHANDLE(inst, rose)),                   \
		.cmd = DT_INST_PROP(inst, rose_cmd),                                  \
		.channel = DT_INST_PROP(inst, rose_channel),                          \
	};                                                                           \
	SENSOR_DEVICE_DT_INST_DEFINE(inst, rose_flow_init, NULL,                      \
				     &rose_flow_data_##inst, &rose_flow_config_##inst, \
				     POST_KERNEL, CONFIG_SENSOR_INIT_PRIORITY,        \
				     &rose_flow_api);

DT_INST_FOREACH_STATUS_OKAY(ROSE_FLOW_DEFINE)
