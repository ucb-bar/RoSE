#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"
#include <math.h>
#include <stdlib.h>

#define IMG_WIDTH 256
#define IMG_HEIGHT 256
#define SEARCH_RANGE 32
#define BLOCK_SIZE 8

#define STEREO_IMG_WIDTH (IMG_WIDTH-SEARCH_RANGE-BLOCK_SIZE)
#define STEREO_IMG_HEIGHT (IMG_HEIGHT-BLOCK_SIZE)

#define rocc_fence() asm volatile("fence")

// byte addressed size
#define ORIGIN_IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)
#define STEREO_IMG_SIZE (STEREO_IMG_WIDTH*STEREO_IMG_HEIGHT)

uint8_t origin_left_buf[ORIGIN_IMG_SIZE];
uint8_t origin_right_buf[ORIGIN_IMG_SIZE];
uint8_t stereo_buf[STEREO_IMG_SIZE];

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

void send_img_stream_req_left(uint32_t interval) {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LEFT_STREAM);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 4);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, interval) ;
}

void send_img_req_right() {
    // printf("Requesting image...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_RIGHT);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_img_loopback_1_row(int row) {
    printf("Sending 1 row...\n");
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LOOPBACK);
    while (ROSE_TX_ENQ_READY == 0) ;
    reg_write32(ROSE_TX_DATA_ADDR, STEREO_IMG_WIDTH);
    for (int i = 0; i < STEREO_IMG_WIDTH; i+=4) {
      while (ROSE_TX_ENQ_READY == 0) ;
      uint32_t data = stereo_buf[row * STEREO_IMG_WIDTH + i]<<24 | stereo_buf[row * STEREO_IMG_WIDTH + i+1]<<16 | stereo_buf[row * STEREO_IMG_WIDTH + i+2]<<8 | stereo_buf[row * STEREO_IMG_WIDTH + i+3];
      reg_write32(ROSE_TX_DATA_ADDR, data);
    }
}

void send_img_loopback(uint32_t *img) {
    // printf("Requesting image...\n");
  for (int j = 0; j < (IMG_HEIGHT-BLOCK_SIZE); j++) {
    send_img_loopback_1_row(j);
  }
  printf("Sent %d rows\n", IMG_HEIGHT-BLOCK_SIZE);
}

void recv_img_dma_left(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_0 + offset * ORIGIN_IMG_SIZE;
  printf("offset for this access is: %d\n", offset);
  memcpy(origin_left_buf, pointer, ORIGIN_IMG_SIZE);
}

void recv_img_dma_right(int offset){
  uint32_t i;
  uint8_t *pointer;
  pointer = ROSE_DMA_BASE_ADDR_1 + offset * ORIGIN_IMG_SIZE;
  printf("offset for this access is: %d\n", offset);
  memcpy(origin_right_buf, pointer, ORIGIN_IMG_SIZE);
}

// Core function computing stereoBM
void compute_dispartiy(uint8_t *left, uint8_t *right, uint8_t *stereo_buf, int min_disparity, int max_disparity, int half_block_size) {
  // allocate data for disparity, use calloc for 0 initialization
  int SAD = 0;
  int min_SAD = INT32_MAX;
  int l_r, l_c, r_r, r_c;
  int height = IMG_HEIGHT;
  int width = IMG_WIDTH;

  int sad_iop = 0;

  signed char *disparity = (signed char *)calloc(width*height, sizeof(signed char));
  if (!disparity) {
      printf("Error: Memory allocation failed\n");
      return NULL;
  }

  // compute disparity
  // outer loop iterating over blocks
  for (int i=0+half_block_size; i<height-half_block_size; i++) {
      for (int j=0+half_block_size-min_disparity; j<width-half_block_size-max_disparity; j++) {
          // middle loop per block
          min_SAD = INT32_MAX;
          for (int offset=min_disparity; offset<max_disparity; offset++) {
              SAD = 0;
              // inner loop per pixel: compute SAD
              for (l_r = i-half_block_size; l_r < half_block_size+i; l_r++) {
                  for (l_c = j-half_block_size; l_c < half_block_size+j; l_c++) {
                      r_r = l_r;
                      r_c = l_c + offset;
                      SAD += abs(left[l_r*width+l_c] - right[r_r*width+r_c]);
                      sad_iop++;
                  }
              }
              // reduction step
              if (SAD < min_SAD) {
                // for debugging
                // if (i == half_block_size) {
                //     printf("Updated min_SAD: %x, SAD: %x, j: %d, offset: %d\n", min_SAD, SAD, j, offset);
                // }
                min_SAD = SAD;
                stereo_buf[(i-half_block_size)*STEREO_IMG_WIDTH + (j-half_block_size)] = offset;
              }
          }
      }
  }
  return disparity;
}

int main(void) {
  int i;
  int img_rcvd = 0;
  uint8_t l_status = 0;
  uint8_t l_status_prev = 0;
  uint8_t r_status = 0;
  uint8_t r_status_prev = 0;

  uint64_t cycles_measured[32] = {0};
  int byte_read = 0;

  printf("Starting Test Code\n");
  configure_counter();

while(img_rcvd < 1){
    send_img_stream_req_left(20);
    uint64_t start = rdcycle();
    do
    {
      l_status_prev = l_status;
      l_status = ROSE_DMA_BUFFER_0;
      uint32_t curr_counter = ROSE_DMA_CURR_COUNTER_0;
      printf("curr_counter: %d\n", curr_counter);
    } while (l_status == l_status_prev);
    recv_img_dma_left(l_status_prev);
    printf("Received left image\n");

    uint64_t end = rdcycle();
    cycles_measured[img_rcvd] = end - start;
    printf("Cycle count: %d\n", cycles_measured[img_rcvd]);
    img_rcvd++;
  }
  while(1);
  
  // for (i = 0; i < 32; i++) {
  //   printf("cycle[%d], %" PRIu64 " cycles\n", i, cycles_measured[i]);
  // }
}

