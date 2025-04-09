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

#define NUM_ITERS 5

#define ORIGIN_IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)

uint32_t buf[ORIGIN_IMG_SIZE/4];

void send_img_req_left() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LEFT);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

int recv_img_1_MMIO(int start_byte) {
  while (ROSE_RX_DEQ_VALID_2 == 0) ;
  buf[start_byte] = ROSE_RX_DATA_2;
  return 1;
}

int main(void) {
  int i;
  int img_rcvd = 0;
  int byte_read = 0;
  uint64_t cycles_measured[NUM_ITERS] = {0};

  printf("Starting Test Code\n");

  while (img_rcvd < NUM_ITERS) {
    uint64_t start = rdcycle();
    send_img_req_left();

    while (byte_read < ORIGIN_IMG_SIZE/4) {
      byte_read += recv_img_1_MMIO(byte_read);
    }
    printf("byte_read: %d\n", byte_read);
    uint64_t end = rdcycle();

    cycles_measured[img_rcvd] = end - start;
    img_rcvd++;
    byte_read = 0;
  }

  for (i = 0; i < NUM_ITERS; i++) {
    printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  }
  exit(0); 
}

