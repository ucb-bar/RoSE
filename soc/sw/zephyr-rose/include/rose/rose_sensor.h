/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Custom sensor channels for the RoSE virtual sensor drivers. Zephyr has no
 * standard optical-flow channel, so the flow driver reports body-frame horizontal
 * velocity on these private channels. IMU (accel/gyro) and ToF (distance) use the
 * standard SENSOR_CHAN_ACCEL_XYZ / SENSOR_CHAN_GYRO_XYZ / SENSOR_CHAN_DISTANCE, so
 * those bind to real bosch,bmi08x-* / st,vl53l1x drivers unchanged.
 */

#ifndef ROSE_ROSE_SENSOR_H_
#define ROSE_ROSE_SENSOR_H_

#include <zephyr/drivers/sensor.h>

/* Body-frame optical-flow horizontal velocity (m/s) + multizone-ToF aggregates. */
enum rose_sensor_channel {
	ROSE_SENSOR_CHAN_FLOW_VX = SENSOR_CHAN_PRIV_START,
	ROSE_SENSOR_CHAN_FLOW_VY,
	/* Multizone ToF: mean zone distance (m). The MIN zone distance is reported on the
	 * standard SENSOR_CHAN_DISTANCE; the full grid is read via rose_tof_zone_grid(). */
	ROSE_SENSOR_CHAN_TOF_ZONE_MEAN,
};

/* Full zone grid accessor for the ucbbar,rose-tof-zone driver (not part of the sensor API);
 * returns a pointer to `*nzones` cached float32 distances (m), valid after channel_get. */
const float *rose_tof_zone_grid(const struct device *dev, uint16_t *nzones);

#endif /* ROSE_ROSE_SENSOR_H_ */
