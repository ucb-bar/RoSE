#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include <rose_port.h>
#include <rose_packet.h>

uint32_t buf[56 * 56 * 3];

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
  reg_write32(AIRSIM_WRITTEN_COUNTER_MAX, 56*56*3);
}

void send_img_req() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void recv_img_dma(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_0 + offset * 56*56*3;
  printf("offset for this access is: %d", offset);
  memcpy(buf, pointer, 56*56*3);
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

  while(img_rcvd < 32){
    send_img_req();
    uint64_t start = rdcycle();
    printf("Requested next image...\n");
    do
    {
      status_prev = status;
      status = ROSE_DMA_BUFFER_0
      printf("status: %x\n", status);
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

