/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual RoSE IMU sensor driver (BMI088-equivalent). Implements the standard Zephyr
 * sensor_driver_api on top of the RoSE bridge, so app/estimator code that reads a BMI088
 * via DT_ALIAS(bmi088_accel)/(bmi088_gyro) + sensor_sample_fetch/sensor_channel_get runs
 * UNCHANGED in the RoSE co-sim; only the devicetree binding differs from the real
 * bosch,bmi08x-* driver.
 *
 * Accelerometer and gyroscope are SEPARATE nodes with their own reqrsp command (like the
 * real BMI088's two I2C devices at 0x18 / 0x68); each fetches its own 3-word frame.
 *
 * PIPELINED / async transport: on the lockstep Spike tier a reqrsp response is served one
 * grant after the request, and rose_rx blocks (spins) on the read. To avoid paying a grant
 * of latency PER sensor, this driver splits the two phases across the sensor API:
 *   - sample_fetch()  ISSUES the request (TX only, non-blocking) and marks a read pending;
 *   - channel_get()   COLLECTS (the blocking read) on first call, then serves from cache.
 * So an app that batches all sample_fetch() calls, then all channel_get() calls, streams
 * every sensor's request in one grant and collects them together in the next -> one grant
 * for the whole sensor set. On real hardware the split is invisible (the real driver does
 * the transaction in sample_fetch); the batched call pattern works identically.
 */

#define DT_DRV_COMPAT ucbbar_rose_imu

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>
#include <string.h>

#include <rose/rose_proto.h>

LOG_MODULE_REGISTER(rose_imu, CONFIG_SENSOR_LOG_LEVEL);

#define IMU_AXES 3

struct rose_imu_config {
	const struct device *rose;
	uint32_t cmd;
	uint8_t channel;
	bool is_gyro;
};

struct rose_imu_data {
	float v[IMU_AXES];
	bool pending;   /* a request was issued (fetch) but not yet collected (get) */
};

static int rose_imu_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	const struct rose_imu_config *cfg = dev->config;
	struct rose_imu_data *data = dev->data;

	if (chan != SENSOR_CHAN_ALL &&
	    chan != SENSOR_CHAN_ACCEL_XYZ && chan != SENSOR_CHAN_GYRO_XYZ) {
		return -ENOTSUP;
	}
	/* Phase 1: issue the request only (non-blocking TX); the read happens in channel_get
	 * so multiple sensors can be fetched back-to-back and stream in one grant. */
	rose_request(cfg->rose, cfg->cmd, 0U);
	data->pending = true;
	return 0;
}

static int rose_imu_channel_get(const struct device *dev, enum sensor_channel chan,
				struct sensor_value *val)
{
	const struct rose_imu_config *cfg = dev->config;
	struct rose_imu_data *data = dev->data;

	if ((cfg->is_gyro  && chan != SENSOR_CHAN_GYRO_XYZ) ||
	    (!cfg->is_gyro && chan != SENSOR_CHAN_ACCEL_XYZ)) {
		return -ENOTSUP;
	}
	/* Phase 2: collect the response on the first get after a fetch (blocks until the
	 * grant delivers it); subsequent gets read the cache. */
	if (data->pending) {
		uint32_t raw[IMU_AXES];
		int n = rose_recv_reqrsp(cfg->rose, cfg->channel, raw, IMU_AXES);
		if (n < IMU_AXES) {
			LOG_ERR("short IMU read: %d/%d", n, IMU_AXES);
			return -EIO;
		}
		memcpy(data->v, raw, sizeof(float) * IMU_AXES);   /* words are float32 bits */
		data->pending = false;
	}
	for (int i = 0; i < IMU_AXES; i++) {
		sensor_value_from_double(&val[i], (double)data->v[i]);
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
