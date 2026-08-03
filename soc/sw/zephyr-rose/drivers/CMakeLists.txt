add_subdirectory_ifdef(CONFIG_ROSE rose)

if(CONFIG_ROSE_SENSOR_IMU OR CONFIG_ROSE_SENSOR_FLOW OR CONFIG_ROSE_SENSOR_TOF)
  add_subdirectory(sensor)
endif()
