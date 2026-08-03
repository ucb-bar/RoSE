/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual RoSE multizone Time-of-Flight (ST VL53L5CX-equivalent). Fetches an N=zones*zones
 * grid of float32 above-surface distances (metres) from the RoSE bridge and exposes:
 *   - SENSOR_CHAN_DISTANCE      -> the MIN zone distance (closest obstacle in the FoV; the
 *                                  key "distance to the wall in this direction" nav signal),
 *   - ROSE_SENSOR_CHAN_TOF_ZONE_MEAN -> the mean zone distance (private channel).
 * The full grid is cached and reachable via rose_tof_zone_grid() for callers that want it.
 *
 * A real VL53L5CX ranges slowly (8x8 @ ~15 Hz), so this driver decimates: it issues a fresh
 * request and returns 0 only every `decimation` sample_fetch() calls (-EAGAIN otherwise).
 * Two-phase like the other RoSE drivers: sample_fetch issues the reqrsp request (TX only),
 * channel_get performs the blocking collect on the first call -> the four sensors pipeline
 * in one grant. A real build binds the same alias to st,vl53l5cx unchanged.
 */

#define DT_DRV_COMPAT ucbbar_rose_tof_zone

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <rose/rose_proto.h>
#include <rose/rose_sensor.h>

LOG_MODULE_REGISTER(rose_tof_zone, CONFIG_SENSOR_LOG_LEVEL);

#define ROSE_TOF_ZONE_MAX 64          /* 8x8 */

struct rose_tof_zone_config {
	const struct device *rose;
	uint32_t cmd;
	uint8_t channel;
	uint16_t zones;                   /* number of words = zones (e.g. 64 for 8x8) */
	uint16_t decimation;
};

struct rose_tof_zone_data {
	float grid[ROSE_TOF_ZONE_MAX];
	float min_m, mean_m, center_m;
	uint16_t nzones;
	uint32_t ctr;
	bool pending;
};

/* Integer sqrt for the square grid dimension (n = dim*dim), so the center (bore) zone can be
 * located without linking libm. */
static uint16_t rose_isqrt(uint16_t n)
{
	uint16_t r = 0;
	while ((uint32_t)(r + 1) * (r + 1) <= n) {
		r++;
	}
	return r;
}

static int rose_tof_zone_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	const struct rose_tof_zone_config *cfg = dev->config;
	struct rose_tof_zone_data *data = dev->data;

	if (chan != SENSOR_CHAN_ALL && chan != SENSOR_CHAN_DISTANCE &&
	    chan != (enum sensor_channel)ROSE_SENSOR_CHAN_TOF_ZONE_MEAN &&
	    chan != (enum sensor_channel)ROSE_SENSOR_CHAN_TOF_ZONE_CENTER) {
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

static void rose_tof_zone_collect(const struct rose_tof_zone_config *cfg,
				  struct rose_tof_zone_data *data)
{
	uint32_t raw[ROSE_TOF_ZONE_MAX];
	uint16_t n = cfg->zones;

	if (n > ROSE_TOF_ZONE_MAX) {
		n = ROSE_TOF_ZONE_MAX;
	}
	int got = rose_recv_reqrsp(cfg->rose, cfg->channel, raw, n);
	if (got < (int)n) {
		LOG_ERR("short ToF-zone read: %d/%u", got, n);
		data->pending = false;
		return;
	}
	float mn = 1e30f, sum = 0.0f;
	for (uint16_t i = 0; i < n; i++) {
		float v;
		memcpy(&v, &raw[i], sizeof(float));
		data->grid[i] = v;
		if (v < mn) {
			mn = v;
		}
		sum += v;
	}
	data->nzones = n;
	data->min_m = mn;
	data->mean_m = sum / (float)n;
	/* center (bore) zone of the square grid = row dim/2, col dim/2 */
	uint16_t dim = rose_isqrt(n);
	uint16_t ci = (dim / 2) * dim + (dim / 2);
	data->center_m = (ci < n) ? data->grid[ci] : data->mean_m;
	data->pending = false;
}

static int rose_tof_zone_channel_get(const struct device *dev, enum sensor_channel chan,
				     struct sensor_value *val)
{
	const struct rose_tof_zone_config *cfg = dev->config;
	struct rose_tof_zone_data *data = dev->data;

	if (data->pending) {
		rose_tof_zone_collect(cfg, data);
	}
	switch ((int)chan) {
	case SENSOR_CHAN_DISTANCE:
		sensor_value_from_double(val, (double)data->min_m);
		return 0;
	case ROSE_SENSOR_CHAN_TOF_ZONE_MEAN:
		sensor_value_from_double(val, (double)data->mean_m);
		return 0;
	case ROSE_SENSOR_CHAN_TOF_ZONE_CENTER:
		sensor_value_from_double(val, (double)data->center_m);
		return 0;
	default:
		return -ENOTSUP;
	}
}

/* Extra accessor for callers that want the full zone grid (not part of the sensor API). */
const float *rose_tof_zone_grid(const struct device *dev, uint16_t *nzones)
{
	struct rose_tof_zone_data *data = dev->data;

	if (nzones) {
		*nzones = data->nzones;
	}
	return data->grid;
}

static const struct sensor_driver_api rose_tof_zone_api = {
	.sample_fetch = rose_tof_zone_sample_fetch,
	.channel_get = rose_tof_zone_channel_get,
};

static int rose_tof_zone_init(const struct device *dev)
{
	const struct rose_tof_zone_config *cfg = dev->config;

	if (!device_is_ready(cfg->rose)) {
		LOG_ERR("rose adapter not ready");
		return -ENODEV;
	}
	return 0;
}

#define ROSE_TOF_ZONE_DEFINE(inst)                                                        \
	static struct rose_tof_zone_data rose_tof_zone_data_##inst;                        \
	static const struct rose_tof_zone_config rose_tof_zone_config_##inst = {           \
		.rose = DEVICE_DT_GET(DT_INST_PHANDLE(inst, rose)),                        \
		.cmd = DT_INST_PROP(inst, rose_cmd),                                       \
		.channel = DT_INST_PROP(inst, rose_channel),                              \
		.zones = DT_INST_PROP(inst, zones),                                        \
		.decimation = DT_INST_PROP(inst, decimation),                            \
	};                                                                                \
	SENSOR_DEVICE_DT_INST_DEFINE(inst, rose_tof_zone_init, NULL,                       \
				     &rose_tof_zone_data_##inst, &rose_tof_zone_config_##inst, \
				     POST_KERNEL, CONFIG_SENSOR_INIT_PRIORITY,           \
				     &rose_tof_zone_api);

DT_INST_FOREACH_STATUS_OKAY(ROSE_TOF_ZONE_DEFINE)
