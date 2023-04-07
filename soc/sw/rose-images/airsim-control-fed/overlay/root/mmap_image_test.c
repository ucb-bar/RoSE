#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "mmio.h"

#define AIRSIM_STATUS (ptr + 0x00)
#define AIRSIM_IN     (ptr + 0x08)
#define AIRSIM_OUT    (ptr + 0x0C)

#define CS_GRANT_TOKEN  0x80
#define CS_REQ_CYCLES   0x81
#define CS_RSP_CYCLES   0x82
#define CS_DEFINE_STEP  0x83

#define CS_REQ_WAYPOINT 0x01
#define CS_RSP_WAYPOINT 0x02
#define CS_SEND_IMU     0x03
#define CS_REQ_ARM      0x04
#define CS_REQ_DISARM   0x05
#define CS_REQ_TAKEOFF  0x06

#define CS_REQ_IMG      0x10
#define CS_RSP_IMG      0x11

#define CS_SET_TARGETS  0x20

volatile void * ptr;
uint32_t buf[54656];

void send_arm() {
    printf("Sending arm...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_ARM);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_takeoff() {
    printf("Sending takeoff...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_TAKEOFF);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_img_req() {
    printf("Requesting image...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_IMG);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void send_target(float zcoord, float xvel, float yvel, float yawrate) {
    printf("Setting target %f, %f, %f, %f...\n", zcoord, xvel, yvel, yawrate);

    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_SET_TARGETS);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 16);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &zcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &xvel));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &yvel));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &yawrate ));
}

void send_waypoint(float xcoord, float ycoord, float zcoord, float vel) {
    printf("Navigating to waypoint...\n",xcoord, ycoord, zcoord);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, CS_RSP_WAYPOINT);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, 16);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &xcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &ycoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &zcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0);
    reg_write32(AIRSIM_IN, *((uint32_t *) &vel));
}

unsigned long long read_cycles()
{
    unsigned long long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

void recv_img(int count) {
  uint32_t i;
  uint8_t status;

  printf("Receiving image %d...\n", count);
  // printf("about to enter loop...\n");
  do {
    // printf("about to read status...\n");
    status = reg_read8(AIRSIM_STATUS);
    //printf("status: %x\n", status);
  } while ((status & 0x1) == 0);
  // while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  uint32_t cmd = reg_read32(AIRSIM_OUT);
  printf("Cmd: %x\n", cmd);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  uint32_t num_bytes = reg_read32(AIRSIM_OUT);
  printf("Num_bytes: %d\n", num_bytes);
  for(i = 0; i < num_bytes / 4; i++) {
    while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
    buf[i] = reg_read32(AIRSIM_OUT);
    printf("(%d, %x) ", i, buf[i]);
  }
  printf("\n");

}

int main() {
    int mem_fd;
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    ptr = mmap(NULL, 16, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, mem_fd, 0x2000);
    printf("Ptr: %x\n", ptr);

	// reg_write32(ptr, 53);
	// printf("Written val: %d\n", reg_read32(ptr));

	uint32_t result, ref;
	uint32_t data0 = 0x00000001;
	uint32_t data1 = 0x00000004;
	uint32_t data2 = 111;
    uint8_t status;
	int i, j;

    printf("Starting Test Code\n");
    send_arm();
    send_takeoff();

    while(1){
      printf("Measuring Cycles...\n");
      unsigned long long pre_inference_cycles = read_cycles();
      send_img_req();
      printf("In between cmds...\n");
      for(i = 0; i < 56; i++) {
        recv_img(i);
      }
      unsigned long long post_inference_cycles = read_cycles();
      printf("Cycles Elapsed: %llu\n", post_inference_cycles - pre_inference_cycles);
      sleep(1);

      //send_target(-1, 1, 1.5, 4);
      //send_waypoint(-10.0, 10.0, -10.0, 5.0);
      //send_waypoint(-10.0,-10.0, -10.0, 5.0);
      //send_waypoint( 10.0,-10.0, -10.0, 5.0);
      //send_waypoint( 10.0, 10.0, -10.0, 5.0);
  }
}
