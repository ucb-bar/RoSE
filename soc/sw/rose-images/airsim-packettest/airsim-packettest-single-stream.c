#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"
#include <math.h>
#include <stdlib.h>

#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define ORIGIN_IMG_SIZE (IMG_HEIGHT*IMG_WIDTH)

#define PACKT_SIZE 32
#define STREAMING_INTERVAL 1

#define NUM_ITERS 8

uint8_t origin_left_buf[ORIGIN_IMG_SIZE];

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, PACKT_SIZE);
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_1, PACKT_SIZE);
}

void send_dummy_req_left() {
  // printf("Requesting image...\n");
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, CS_DUMMY_STREAM);
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, 4);
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, STREAMING_INTERVAL);
}

void recv_img_dma_left(int offset){
  volatile uint8_t *pointer;
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
  uint64_t singularity = rdcycle();
  int i;
  int img_rcvd = 0;
  uint8_t l_status = 0;
  uint8_t l_status_prev = 0;

  uint64_t cycles_measured[NUM_ITERS] = {0};
  configure_counter();
  send_dummy_req_left();

  while(img_rcvd < NUM_ITERS){
    // uint64_t start = rdcycle();
    do
    {
      l_status_prev = l_status;
      l_status = ROSE_DMA_BUFFER_0;
    } while (l_status == l_status_prev);
    // recv_img_dma_left(l_status_prev);
    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end;
    img_rcvd++;
  }

  printf("singularity: %" PRIu64 "\n", singularity);
  
  for (i = 0; i < NUM_ITERS; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }

  exit(0);
}

