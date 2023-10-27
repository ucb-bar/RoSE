#ifndef __ROSE_H__
#define __ROSE_H__

#include <stdint.h>
#include <stdio.h>
#include "mmio.h"

#define ROSE_STATUS 0x2000
#define ROSE_WRITTEN_COUNTER_MAX 0x2014
#define ROSE_IN 0x2008
#define ROSE_OUT 0x200C

//TODO: verify this is the correct address and cacheblockbytes
#define ROSE_DMA 0x88000000
#define CacheBlockBytes 64

#define ROSE_RESET 0x01

void send_sim_reset() {
	printf("Sending reset cmd...\n");
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, ROSE_RESET);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, 0);
}

void send_obs_req(uint8_t cmd) {
    printf("Requesting cmd %02x...\n", cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, 0);
}

void read_obs_rsp(void * buf) {
	uint32_t i;
	uint8_t status;
	uint32_t raw_result;
	float result;

	uint32_t * raw_buf = (uint32_t*) buf;
	printf("Receiving obs...\n");
	while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
	uint32_t cmd = reg_read32(ROSE_OUT);
	printf("Got cmd: %x\n", cmd);
	while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
	uint32_t num_bytes = reg_read32(ROSE_OUT);
	printf("Got num bytes: %d\n", num_bytes);
	for(i = 0; i < num_bytes / 4; i++) {
		while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
    	raw_buf[i] = reg_read32(ROSE_OUT);
    	// printf("(%d, %x) ", i, buf[i]);
	}
	
}

void send_action(void * buf, uint32_t cmd, uint32_t num_bytes) {
	uint32_t i;
	uint32_t * raw_buf = (uint32_t*) buf;

	printf("Sending action %x...\n", cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, num_bytes);
	for (i = 0; i < num_bytes / 4; i++) {
		while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
		reg_write32(ROSE_IN, raw_buf[i]);
	}

}

#endif
