#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"

uint32_t buf[56 * 56 * 3];

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
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_TAKEOFF);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void configure_counter(){
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, 56*56*3);
}

void send_img_req() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void recv_img() {
  uint32_t i;
  uint8_t status;

  do {
    status = ROSE_RX_DEQ_VALID_1;
  } while (status == 0);
  uint32_t cmd = ROSE_RX_DATA_1;

  while (ROSE_RX_DEQ_VALID_1 == 0) ;
  uint32_t num_bytes = ROSE_RX_DATA_1;
  
  for(i = 0; i < num_bytes / 4; i++) {
    while (ROSE_RX_DEQ_VALID_1 == 0) ;
    buf[i] = ROSE_RX_DATA_1;
  }
}

int main(void)
{
  uint32_t result, ref;

  uint8_t status, status_prev;
  int i, j;
  int img_rcvd = 0;
  uint64_t cycles_measured[32] = {0};

  printf("Starting Test Code\n");
  configure_counter();
  printf("Configured counter...\n");
  // send_arm();
  // printf("Armed...\n");
  // send_takeoff();
  // printf("Took off...\n");

  while(img_rcvd < 32){
    uint64_t start = rdcycle();
    send_img_req();
    for(i = 0; i < 180; i++) {
      recv_img();
    }
    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end - start;
    img_rcvd++;
  }

  for (i = 0; i < 32; i++){
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
}

