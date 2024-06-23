#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include <stdlib.h>
// #include "nic.h"

#include "rose_port.h"
#include "rose_packet.h"

#define NUM_ITERS 20 
#define STREAMING_INTERVAL 1
#define NUM_BYTES 8

uint64_t cycles_measured[NUM_ITERS] = {0};
int packet_rcvd = 0;

void send_test_req() {
  // printf("Requesting image...\n");
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, CS_LAT_TEST);
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, 4);
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, STREAMING_INTERVAL);
}

int recv_packet() {
  while (ROSE_RX_DEQ_VALID_2 == 0) ;
  cycles_measured[packet_rcvd] = rdcycle();
  uint32_t header = ROSE_RX_DATA_2;
  // printf("Received packet with header: %d\n", header);
  while (ROSE_RX_DEQ_VALID_2 == 0) ;
  uint32_t size = ROSE_RX_DATA_2;
  int byte_read = 0;
  while (byte_read < size) {
    while (ROSE_RX_DEQ_VALID_2 == 0) ;
    uint32_t data = ROSE_RX_DATA_2;
    byte_read += 4;
  }
}

int main(void) {
  int i;
  send_test_req();

  while (packet_rcvd < NUM_ITERS) {
    recv_packet();
    packet_rcvd++;
  }

  for (i = 0; i < NUM_ITERS; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
  exit(0); 
}

