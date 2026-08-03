/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual RoSE downward Time-of-Flight rangefinder (VL53L1x-equivalent). Fetches a
 * 1-word [height] float32 above-ground distance from the RoSE bridge and exposes it
 * as the standard SENSOR_CHAN_DISTANCE, so a real build binds DT_ALIAS(tof) to
 * st,vl53l1x unchanged.
 *
 * A real ToF samples slowly (~20-40 ms). This driver models that low rate: it only
 * issues a fresh request and returns 0 every `decimation` sample_fetch() calls, and
 * returns -EAGAIN otherwise. The application passes tof_valid = (fetch == 0) to the
 * estimator, which fuses altitude only on fresh samples (multi-rate estimation) and
 * dead-reckons on the IMU in between.
 */

#define DT_DRV_COMPAT ucbbar_rose_tof

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <rose/rose_proto.h>

LOG_MODULE_REGISTER(rose_tof, CONFIG_SENSOR_LOG_LEVEL);

struct rose_tof_config {
	const struct device *rose;
	uint32_t cmd;
	uint8_t channel;
	uint16_t decimation;
};

struct rose_tof_data {
	float height;
	uint32_t ctr;
	bool pending;
};

/* Low-rate + two-phase (pipelined). sample_fetch: on a fresh step issue the request (TX
 * only) and return 0; on other steps return -EAGAIN (no new sample -> the estimator skips
 * altitude fusion this step). channel_get collects the response on first call. */
static int rose_tof_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	const struct rose_tof_config *cfg = dev->config;
	struct rose_tof_data *data = dev->data;

	if (chan != SENSOR_CHAN_ALL && chan != SENSOR_CHAN_DISTANCE) {
		return -ENOTSUP;
	}

	bool fresh = (data->ctr % cfg->decimation) == 0U;
	data->ctr++;
	if (!fresh) {
		return -EAGAIN;
	}
	rose_request(cfg->rose, cfg->cmd, 0U);   /* issue only; collected in channel_get */
	data->pending = true;
	return 0;
}

static int rose_tof_channel_get(const struct device *dev, enum sensor_channel chan,
				struct sensor_value *val)
{
	const struct rose_tof_config *cfg = dev->config;
	struct rose_tof_data *data = dev->data;

	if (chan != SENSOR_CHAN_DISTANCE) {
		return -ENOTSUP;
	}
	if (data->pending) {
		uint32_t raw;
		int n = rose_recv_reqrsp(cfg->rose, cfg->channel, &raw, 1U);
		if (n < 1) {
			LOG_ERR("short ToF read");
			return -EIO;
		}
		memcpy(&data->height, &raw, sizeof(float));
		data->pending = false;
	}
	sensor_value_from_double(val, (double)data->height);
	return 0;
}

static const struct sensor_driver_api rose_tof_api = {
	.sample_fetch = rose_tof_sample_fetch,
	.channel_get = rose_tof_channel_get,
};

static int rose_tof_init(const struct device *dev)
{
	const struct rose_tof_config *cfg = dev->config;

	if (!device_is_ready(cfg->rose)) {
		LOG_ERR("rose adapter not ready");
		return -ENODEV;
	}
	return 0;
}

#define ROSE_TOF_DEFINE(inst)                                                       \
	static struct rose_tof_data rose_tof_data_##inst;                           \
	static const struct rose_tof_config rose_tof_config_##inst = {              \
		.rose = DEVICE_DT_GET(DT_INST_PHANDLE(inst, rose)),                 \
		.cmd = DT_INST_PROP(inst, rose_cmd),                                \
		.channel = DT_INST_PROP(inst, rose_channel),                        \
		.decimation = DT_INST_PROP(inst, decimation),                      \
	};                                                                         \
	SENSOR_DEVICE_DT_INST_DEFINE(inst, rose_tof_init, NULL,                     \
				     &rose_tof_data_##inst, &rose_tof_config_##inst, \
				     POST_KERNEL, CONFIG_SENSOR_INIT_PRIORITY,      \
				     &rose_tof_api);

DT_INST_FOREACH_STATUS_OKAY(ROSE_TOF_DEFINE)
