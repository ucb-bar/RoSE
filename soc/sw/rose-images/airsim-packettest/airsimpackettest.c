#include "mmio.h"

#include<stdio.h>

#define AIRSIM_STATUS 0x2000
#define AIRSIM_WRITTEN_COUNTER_MAX 0x2004
#define AIRSIM_IN 0x2008
#define AIRSIM_OUT 0x200C

//TODO: verify this is the correct address and cacheblockbytes
#define AIRSIM_DMA 0x88000000
#define CacheBlockBytes 64

#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES  0x81
#define CS_RSP_CYCLES  0x82
#define CS_DEFINE_STEP 0x83

#define CS_REQ_WAYPOINT 0x01
#define CS_RSP_WAYPOINT 0x02
#define CS_SEND_IMU 0x03
#define CS_REQ_ARM  0x04
#define CS_REQ_DISARM  0x05
#define CS_REQ_TAKEOFF 0x06

#define CS_REQ_IMG      0x10
#define CS_RSP_IMG      0x11

#define CS_REQ_DEPTH     0x12
#define CS_RSP_DEPTH     0x13

#define CS_SET_TARGETS  0x20

uint32_t buf[56 * 56 * 3];

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

void send_waypoint(float xcoord, float ycoord, float zcoord, float vel) {
    printf("Navigating to waypoint...\n",xcoord, ycoord, zcoord);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_RSP_WAYPOINT);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 16);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &xcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &ycoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &zcoord));
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, *((uint32_t *) &vel));
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

void send_depth_req() {
    printf("Requesting depth...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_DEPTH);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

float load_depth() {
  uint32_t i;
  uint8_t status;
  uint32_t raw_result;
  float result;

  printf("Receiving depth ...\n");
  do {
    status = reg_read8(AIRSIM_STATUS);
  } while ((status & 0x1) == 0);

  uint32_t cmd = reg_read32(AIRSIM_OUT);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  uint32_t num_bytes = reg_read32(AIRSIM_OUT);
  while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  raw_result = reg_read32(AIRSIM_OUT);
  result = *((float *) &raw_result);
  return result;
}

void configure_counter(){
  printf("Configuring counter...\n");
  reg_write32(AIRSIM_WRITTEN_COUNTER_MAX, 56*56*3);
}

void send_img_req() {
    printf("Requesting image...\n");
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, CS_REQ_IMG);
    while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
    reg_write32(AIRSIM_IN, 0);
}

void recv_img_dma(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = AIRSIM_DMA + offset * 56*56*3;
  printf("offset for this access is: %d", offset);
  memcpy(buf, pointer, 56*56*3);
}

void recv_img() {
  uint32_t i;
  uint8_t status;

  printf("Receiving image...\n");
  // printf("about to enter loop...\n");
  do {
    // printf("about to read status...\n");
    status = reg_read8(AIRSIM_STATUS);
    printf("status: %x\n", status);
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



// DOC include start: AIRSIM test
int main(void)
{
  uint32_t result, ref;
  // uint32_t data0 = 0x00000001;
  // uint32_t data1 = 0x00000004;
  // uint32_t data2 = 111;
  uint8_t status, status_prev;
  int i, j;

  printf("Starting Test Code\n");
  configure_counter();
  printf("Configured counter...\n");
  send_arm();
  printf("Armed...\n");
  send_takeoff();
  printf("Took off...\n");

  // while (1){
  //   send_depth_req();
  //   float depth = load_depth();
  //   printf("Depth received: %f\n", depth);
  // }


  // This is a hack to get the first image, and the following while do while loop is for verifying
  // that the image is being received correctly
  send_img_req();
  status = 0x0;
  status_prev = 0x0;
  printf("Requested first image...\n");

  while(1){
    send_img_req();
    printf("Requested next image...\n");
    do
    {
      status_prev = status;
      status = reg_read8(AIRSIM_STATUS);
      printf("status: %x\n", status);
      //while the cam buffer has not advanced, wait
    } while ((status & 0x4) == (status_prev & 0x4));

    // TODO: see if it works the other way round
    recv_img_dma((status_prev & 0x4)>>2);
    printf("Finished receiving one image...\n");

    for (size_t i = 0; i < 10; i++)
    {
      printf("img[%d]: %x\n", i, buf[i]);
    }
    
  }

    // send_img_req();
    // printf("In between cmds...\n");
    // for(i = 0; i < 180; i++) {
    //   recv_img();
    // }
    //send_target(-1, 1, 1.5, 4);
    //send_waypoint(-10.0, 10.0, -10.0, 5.0);
    //send_waypoint(-10.0,-10.0, -10.0, 5.0);
    //send_waypoint( 10.0,-10.0, -10.0, 5.0);
    //send_waypoint( 10.0, 10.0, -10.0, 5.0);

  //for(i = 0; i < 100; i++) {
  //  // wait for peripheral to be ready
  //  while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
  //  reg_write32(AIRSIM_IN, data0);
  //  printf("SoC SENT DATA0\n");
  //  while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
  //  reg_write32(AIRSIM_IN, data1);
  //  printf("SoC SENT DATA1\n");
  //  while ((reg_read8(AIRSIM_STATUS) & 0x2) == 0) ;
  //  reg_write32(AIRSIM_IN, data2);
  //  printf("SoC SENT DATA2\n");
  //  data2++;
  //  // wait for peripheral to complete
  //  for(j = 0; j < 3; j++) {
  //    while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  //    result = reg_read32(AIRSIM_OUT);
  //    printf("SoC Got Data: 0x%x\n", result);
  //  }
  //}


  // if (result != data0) {
  //   printf("Hardware result %x does not match reference value %x\n", result, data0);
  //   return 1;
  // }

  // while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  // result = reg_read32(AIRSIM_OUT);
  // if (result != data1) {
  //   printf("Hardware result %x does not match reference value %x\n", result, data1);
  //   return 1;
  // }

  // while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  // result = reg_read32(AIRSIM_OUT);
  // if (result != data2) {
  //   printf("Hardware result %x does not match reference value %x\n", result, data2);
  //   return 1;
  // }

  // while ((reg_read8(AIRSIM_STATUS) & 0x1) == 0) ;
  // result = reg_read32(AIRSIM_OUT);
  // if (result != data3) {
  //   printf("Hardware result %x does not match reference value %x\n", result, data3);
  //   return 1;
  // }
  // printf("Completed AirSim Test~\n");
  // return 0;
}

