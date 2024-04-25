#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "nic.h"

#include "rose_port.h"
#include "rose_packet.h"

#define IMG_WIDTH 128
#define IMG_HEIGHT 128
#define SEARCH_RANGE 16
#define BLOCK_SIZE 8

// byte addressed size
#define STEREO_IMG_SIZE ((IMG_WIDTH-SEARCH_RANGE)*(IMG_HEIGHT-BLOCK_SIZE+1))

uint32_t buf[STEREO_IMG_SIZE/4];

void send_arm() {
    printf("Sending arm...\n");
    while (ROSE_TX_ENQ_READY == 0);
    reg_write32(ROSE_TX_DATA_ADDR, CS_ARM);
    while (ROSE_TX_ENQ_READY == 0);
    printf("Sent arm...\n");
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_takeoff() {
    printf("Sending takeoff...\n");
    while (ROSE_TX_ENQ_READY == 0);
    reg_write32(ROSE_TX_DATA_ADDR, CS_TAKEOFF);
    while (ROSE_TX_ENQ_READY == 0);
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, 56*56*3);
}

void send_img_req() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_STEREO);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

int recv_img(int start_byte) {
  uint32_t i;
  uint8_t status;

  do {
    status = ROSE_RX_DEQ_VALID_1;
  } while (status == 0);
  uint32_t cmd = ROSE_RX_DATA_1;
  // printf("Got cmd: %d\n", cmd);

  while (ROSE_RX_DEQ_VALID_1 == 0) ;
  uint32_t num_bytes = ROSE_RX_DATA_1;
  // printf("Got num_bytes: %d\n", num_bytes);
  
  for (i = 0; i < num_bytes / 4; i++) {
    while (ROSE_RX_DEQ_VALID_1 == 0) ;
    buf[i + start_byte] = ROSE_RX_DATA_1;
  }
  return num_bytes;
}

int recv_img_1_MMIO(int start_byte) {
  while (ROSE_RX_DEQ_VALID_1 == 0) ;
  buf[start_byte] = ROSE_RX_DATA_1;
  return 1;
}

int main(void) {
  int i;
  int img_rcvd = 0;
  uint64_t cycles_measured[32] = {0};
  int byte_read = 0;

  printf("Starting Test Code\n");
  configure_counter();
  // printf("Configured counter...\n");
  // send_arm();
  // printf("Armed...\n");
  // send_takeoff();
  // printf("Took off...\n");

  while (img_rcvd < 2) {
    uint64_t start = rdcycle();
    send_img_req();

    while (byte_read < STEREO_IMG_SIZE/4) {
      byte_read += recv_img_1_MMIO(byte_read);
    }
    uint64_t end = rdcycle();

    cycles_measured[img_rcvd] = end - start;
    img_rcvd++;
    printf("Received image %d\n", img_rcvd);

    // write image to nic
    nic_send(buf, STEREO_IMG_SIZE);

    byte_read = 0;
  }

  for (i = 0; i < 32; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
}

