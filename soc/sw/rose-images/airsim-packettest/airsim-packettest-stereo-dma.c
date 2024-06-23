#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include <stdlib.h>

// #include "nic.h"

#include "rose_port.h"
#include "rose_packet.h"

#define IMG_WIDTH 256
#define IMG_HEIGHT 256
#define SEARCH_RANGE 32
#define BLOCK_SIZE 8

#define NUM_ITERS 1

#define STEREO_IMG_WIDTH (IMG_WIDTH-SEARCH_RANGE-BLOCK_SIZE)
#define STEREO_IMG_HEIGHT (IMG_HEIGHT-BLOCK_SIZE+1)

// byte addressed size
#define STEREO_IMG_SIZE (STEREO_IMG_WIDTH*STEREO_IMG_HEIGHT)

uint32_t buf[STEREO_IMG_SIZE/4];

// void send_arm() {
//     printf("Sending arm...\n");
//     while (ROSE_TX_ENQ_READY == 0);
//     reg_write32(ROSE_TX_DATA_ADDR, CS_ARM);
//     while (ROSE_TX_ENQ_READY == 0);
//     printf("Sent arm...\n");
//     reg_write32(ROSE_TX_DATA_ADDR, 0);
// }

// void send_takeoff() {
//     printf("Sending takeoff...\n");
//     while (ROSE_TX_ENQ_READY == 0);
//     reg_write32(ROSE_TX_DATA_ADDR, CS_TAKEOFF);
//     while (ROSE_TX_ENQ_READY == 0);
//     reg_write32(ROSE_TX_DATA_ADDR, 0);
// }

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_1, STEREO_IMG_SIZE);
}

void send_img_req() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_STEREO);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_img_loopback_1_row(int row) {
    printf("Sending 1 row...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LOOPBACK);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, STEREO_IMG_WIDTH);
    for (int i = 0; i < STEREO_IMG_WIDTH/4; i++) {
      while (ROSE_TX_ENQ_READY == 0) ;
      reg_write32(ROSE_TX_DATA_ADDR, buf[row * STEREO_IMG_WIDTH/4 + i]);
    }
}

void send_img_loopback(uint32_t *img) {
    // printf("Requesting image...\n");
  for (int j = 0; j < (IMG_HEIGHT-BLOCK_SIZE); j++) {
    send_img_loopback_1_row(j);
  }
  printf("Sent %d rows\n", IMG_HEIGHT-BLOCK_SIZE);
}

void recv_img_dma_stereo(int offset){
  uint32_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_1 + offset * STEREO_IMG_SIZE;
  memcpy(buf, pointer, STEREO_IMG_SIZE);
}

int main(void) {
  int i;
  int img_rcvd = 0;
  uint64_t cycles_measured[32] = {0};

  int status = 0;
  int status_prev = 0;

  printf("Starting Test Code\n");
  configure_counter();
  // printf("Configured counter...\n");
  // send_arm();
  // printf("Armed...\n");
  // send_takeoff();
  // printf("Took off...\n");

  while (img_rcvd < NUM_ITERS) {
    send_img_req();
    uint64_t start = rdcycle();

    do
    {
      status_prev = status;
      status = ROSE_DMA_BUFFER_1;
    } while (status == status_prev);
    recv_img_dma_stereo(status_prev);
    // printf("byte_read: %d\n", byte_read);
    uint64_t end = rdcycle();

    cycles_measured[img_rcvd] = end - start;
    img_rcvd++;
    printf("Received image %d\n", img_rcvd);

    // write image loopback
    // send_img_loopback(buf);
    // send_img_loopback_1_row(buf); 
    // byte_read = 0;
  }
  for (i = 0; i < NUM_ITERS; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
  exit(0); 
}

