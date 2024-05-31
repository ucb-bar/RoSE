#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"

#include "rocc.h"

#define IMG_WIDTH 256
#define IMG_HEIGHT 256
#define SEARCH_RANGE 32
#define BLOCK_SIZE 8

#define STEREO_IMG_WIDTH (IMG_WIDTH-SEARCH_RANGE-BLOCK_SIZE)
#define STEREO_IMG_HEIGHT (IMG_HEIGHT-BLOCK_SIZE)

#define rocc_fence() asm volatile("fence")

// byte addressed size
#define ORIGIN_IMG_SIZE (IMG_WIDTH*IMG_HEIGHT*2)
#define STEREO_IMG_SIZE (STEREO_IMG_WIDTH*STEREO_IMG_HEIGHT)

void CONFIG_DMA_R_ADDR (uint32_t* addr) {
  ROCC_INSTRUCTION_S(0, addr, 1);
}

void CONFIG_DMA_W_ADDR (uint32_t* addr) {
  ROCC_INSTRUCTION_S(0, addr, 2);
}

void COMPUTE_STEREO (void) {
  ROCC_INSTRUCTION_S(0, 0, 0);
}

uint32_t origin_buf[ORIGIN_IMG_SIZE/4];
uint32_t stereo_buf[STEREO_IMG_SIZE/4];

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, ORIGIN_IMG_SIZE);
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
      reg_write32(ROSE_TX_DATA_ADDR, stereo_buf[row * STEREO_IMG_WIDTH/4 + i]);
    }
}

void send_img_loopback(uint32_t *img) {
    // printf("Requesting image...\n");
  for (int j = 0; j < (IMG_HEIGHT-BLOCK_SIZE); j++) {
    send_img_loopback_1_row(j);
  }
  printf("Sent %d rows\n", IMG_HEIGHT-BLOCK_SIZE);
}

void recv_img_dma(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_0 + offset * ORIGIN_IMG_SIZE;
  printf("offset for this access is: %d\n", offset);
  memcpy(origin_buf, pointer, ORIGIN_IMG_SIZE);
}

int main(void) {
  int i;
  int img_rcvd = 0;
  uint8_t status, status_prev;
  uint64_t cycles_measured[32] = {0};
  int byte_read = 0;

  printf("Starting Test Code\n");
  configure_counter();

  CONFIG_DMA_R_ADDR(origin_buf);
  CONFIG_DMA_W_ADDR(stereo_buf);

while(img_rcvd < 1){
    send_img_req();
    uint64_t start = rdcycle();
    do
    {
      status_prev = status;
      status = ROSE_DMA_BUFFER_0;
    } while (status == status_prev);

    recv_img_dma(status_prev);
    printf("Received image\n");

    COMPUTE_STEREO();
    rocc_fence();
    printf("Computed stereo\n");

    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end - start;
    send_img_loopback(stereo_buf);
    img_rcvd++;
  }
  while(1);
  
  // for (i = 0; i < 32; i++) {
  //   printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  // }
}

