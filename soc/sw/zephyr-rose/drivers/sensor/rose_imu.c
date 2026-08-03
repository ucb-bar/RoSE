/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual RoSE IMU sensor driver (BMI088-equivalent). Implements the standard
 * Zephyr sensor_driver_api on top of the RoSE bridge transport (rose_request /
 * rose_recv_reqrsp), so application/estimator code that reads a BMI088 via
 * DT_ALIAS(bmi088_accel)/(bmi088_gyro) + sensor_sample_fetch/sensor_channel_get
 * runs UNCHANGED in the RoSE co-sim; only the devicetree binding differs from the
 * real bosch,bmi08x-* driver.
 *
 * One reqrsp packet (rose-cmd, e.g. 0x12) carries the full 6-axis frame
 * [ax,ay,az, gx,gy,gz] as float32 words, matching the env's IMU packet. Each
 * logical node (accel or gyro, selected by the `is-gyro` DT flag) fetches the
 * frame and exposes its half.
 */

#define DT_DRV_COMPAT ucbbar_rose_imu

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <rose/rose_proto.h>

LOG_MODULE_REGISTER(rose_imu, CONFIG_SENSOR_LOG_LEVEL);

#define IMU_WORDS 6

struct rose_imu_config {
	const struct device *rose;
	uint32_t cmd;
	uint8_t channel;
	bool is_gyro;
};

struct rose_imu_data {
	int _unused;
};

/* Accel and gyro ride in ONE reqrsp frame, so fetch it once per tick into a shared cache:
 * the accelerometer node issues the request, the gyroscope node reuses the cache (no second
 * reqrsp round-trip, which matters for the control-loop cycle budget). Single IMU per
 * vehicle, so one shared frame suffices. */
static float s_imu_frame[IMU_WORDS];   /* [ax,ay,az, gx,gy,gz] */
static bool s_imu_valid;

static int rose_imu_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	const struct rose_imu_config *cfg = dev->config;
	uint32_t raw[IMU_WORDS];

	if (chan != SENSOR_CHAN_ALL &&
	    chan != SENSOR_CHAN_ACCEL_XYZ && chan != SENSOR_CHAN_GYRO_XYZ) {
		return -ENOTSUP;
	}

	/* Gyro node: reuse the frame the accel node already fetched this tick. */
	if (cfg->is_gyro) {
		return s_imu_valid ? 0 : -EAGAIN;
	}

	rose_request(cfg->rose, cfg->cmd, 0U);
	int n = rose_recv_reqrsp(cfg->rose, cfg->channel, raw, IMU_WORDS);
	if (n < IMU_WORDS) {
		LOG_ERR("short IMU read: %d/%d", n, IMU_WORDS);
		return -EIO;
	}
	memcpy(s_imu_frame, raw, sizeof(float) * IMU_WORDS);   /* words are float32 bits */
	s_imu_valid = true;
	return 0;
}

static int rose_imu_channel_get(const struct device *dev, enum sensor_channel chan,
				struct sensor_value *val)
{
	const struct rose_imu_config *cfg = dev->config;
	const float *v = cfg->is_gyro ? &s_imu_frame[3] : &s_imu_frame[0];

	if (cfg->is_gyro) {
		if (chan != SENSOR_CHAN_GYRO_XYZ) {
			return -ENOTSUP;
		}
	} else {
		if (chan != SENSOR_CHAN_ACCEL_XYZ) {
			return -ENOTSUP;
		}
	}
	for (int i = 0; i < 3; i++) {
		sensor_value_from_double(&val[i], (double)v[i]);
	}
	return 0;
}

static const struct sensor_driver_api rose_imu_api = {
	.sample_fetch = rose_imu_sample_fetch,
	.channel_get = rose_imu_channel_get,
};

static int rose_imu_init(const struct device *dev)
{
	const struct rose_imu_config *cfg = dev->config;

	if (!device_is_ready(cfg->rose)) {
		LOG_ERR("rose adapter not ready");
		return -ENODEV;
	}
	return 0;
}

#define ROSE_IMU_DEFINE(inst)                                                        \
	static struct rose_imu_data rose_imu_data_##inst;                            \
	static const struct rose_imu_config rose_imu_config_##inst = {               \
		.rose = DEVICE_DT_GET(DT_INST_PHANDLE(inst, rose)),                  \
		.cmd = DT_INST_PROP(inst, rose_cmd),                                 \
		.channel = DT_INST_PROP(inst, rose_channel),                         \
		.is_gyro = DT_INST_PROP(inst, is_gyro),                              \
	};                                                                          \
	SENSOR_DEVICE_DT_INST_DEFINE(inst, rose_imu_init, NULL,                      \
				     &rose_imu_data_##inst, &rose_imu_config_##inst, \
				     POST_KERNEL, CONFIG_SENSOR_INIT_PRIORITY,       \
				     &rose_imu_api);

DT_INST_FOREACH_STATUS_OKAY(ROSE_IMU_DEFINE)
