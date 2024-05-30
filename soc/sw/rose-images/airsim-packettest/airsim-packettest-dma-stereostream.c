#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include <rose_port.h>
#include <rose_packet.h>

#define IMG_WIDTH 256
#define IMG_HEIGHT 256
#define SEARCH_RANGE 32
#define BLOCK_SIZE 8

#define STEREO_IMG_WIDTH (IMG_WIDTH-SEARCH_RANGE-BLOCK_SIZE)
#define STEREO_IMG_HEIGHT (IMG_HEIGHT-BLOCK_SIZE)

// byte addressed size
#define STEREO_IMG_SIZE (STEREO_IMG_WIDTH*STEREO_IMG_HEIGHT)

uint32_t buf[STEREO_IMG_SIZE/4];

void send_arm() {
    printf("Sending arm...\n");
    while (ROSE_TX_ENQ_READY == 0);
    reg_write32(ROSE_TX_DATA_ADDR, CS_REQ_ARM);
    while (ROSE_TX_ENQ_READY == 0);
    printf("Sent arm...\n");
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_takeoff() {
    printf("Sending takeoff...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_REQ_TAKEOFF);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void configure_counter(){
  printf("Configuring counter...\n");
  reg_write32(AIRSIM_WRITTEN_COUNTER_MAX, STEREO_IMG_SIZE);
}

void send_img_req() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_STEREO);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void recv_img_dma(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_0 + offset * STEREO_IMG_SIZE;
  printf("offset for this access is: %d", offset);
  memcpy(buf, pointer, STEREO_IMG_SIZE);
}

int main(void)
{
  uint32_t result, ref;
  uint8_t status, status_prev;
  int i, j;

  printf("Starting Test Code\n");
  configure_counter();
  printf("Configured counter...\n");
  send_arm();
  printf("Armed...\n");
  send_takeoff();
  printf("Took off...\n");

  int img_rcvd = 0;
  uint64_t cycles_measured[32] = {0};

  send_img_req();
  status = 0x0;
  status_prev = 0x0;

  while(img_rcvd < 32){
    send_img_req();
    uint64_t start = rdcycle();
    do
    {
      status_prev = status;
      status = ROSE_DMA_BUFFER_0;
    } while (status == status_prev);

    recv_img_dma(status_prev);
    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end - start;
    img_rcvd++;
  }
  for (i = 0; i < 32; i++){
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
}