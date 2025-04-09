#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"
#include <math.h>
#include <stdlib.h>

#define IMG_HEIGHT 128 
#define IMG_WIDTH 128
#define ORIGIN_IMG_SIZE (IMG_HEIGHT*IMG_WIDTH)

#define NUM_ITERS 8

uint8_t origin_left_buf[ORIGIN_IMG_SIZE];

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, ORIGIN_IMG_SIZE);
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_1, ORIGIN_IMG_SIZE);
}

void send_img_req_left() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LEFT);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void recv_img_dma_left(int offset){
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_0 + offset * ORIGIN_IMG_SIZE;
  // printf("offset for this access is: %d\n", offset);
  // memcpy(origin_left_buf, pointer, ORIGIN_IMG_SIZE);
}

void cache_warmup(int offset) {
  memcpy(ROSE_DMA_BASE_ADDR_0 + offset * ORIGIN_IMG_SIZE, origin_left_buf, ORIGIN_IMG_SIZE);
}

void cache_warmup_both() {
  cache_warmup(0);
  cache_warmup(1);
}

int main(void) {
  int i;
  int img_rcvd = 0;
  uint8_t l_status = 0;
  uint8_t l_status_prev = 0;

  uint64_t cycles_measured[32] = {0};
  printf("Starting Test Code\n");
  configure_counter();

  // cache_warmup_both();

  send_img_req_left();

while(img_rcvd < NUM_ITERS){
    send_img_req_left();
    uint64_t start = rdcycle();
    do
    {
      l_status_prev = l_status;
      l_status = ROSE_DMA_BUFFER_0;
      // uint32_t curr_counter = ROSE_DMA_CURR_COUNTER_0;
      // printf("curr_counter: %d\n", curr_counter);
    } while (l_status == l_status_prev);
    recv_img_dma_left(l_status_prev);
    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end - start;
    // printf("Received left image\n");
    img_rcvd++;
  }
  
  for (i = 0; i < NUM_ITERS; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }

  exit(0);
}

