#include "mmio.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>

#include "rose_port.h"
#include "rose_packet.h"
#include <math.h>
#include <stdlib.h>

#include "arch.h"
#include "plic.h"

#define IMG_WIDTH               256
#define IMG_HEIGHT              256
#define SEARCH_RANGE            32
#define BLOCK_SIZE              8

#define STEREO_IMG_WIDTH        (IMG_WIDTH-SEARCH_RANGE-BLOCK_SIZE)
#define STEREO_IMG_HEIGHT       (IMG_HEIGHT-BLOCK_SIZE)

#define MIE                     0x304
#define MIP                     0x344

#define MEIE                    (1 << 11)

#define rocc_fence()            (asm volatile("fence"))

// byte addressed size
#define ORIGIN_IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)
#define STEREO_IMG_SIZE (STEREO_IMG_WIDTH*STEREO_IMG_HEIGHT)

uint8_t origin_left_buf[ORIGIN_IMG_SIZE];
uint8_t origin_right_buf[ORIGIN_IMG_SIZE];
uint8_t stereo_buf[STEREO_IMG_SIZE];

extern void trap_entry(void);

void setup_trap_vector(void) {
  uint64_t trap_vector = (uint64_t) &trap_entry;
  WRITE_CSR("mtvec", trap_vector);
}

void external_interrupt_handler(void) {
  uint64_t hart_id = READ_CSR("mhartid");
  uint32_t irq_id = PLIC_claimIRQ(hart_id);
  uint64_t mip = READ_CSR("mip");
  printf("MIP: 0x%lx\n", mip);
  uint32_t device_pending = reg_read32(0x2024);
  printf("External interrupt received\n");
      // Clear the interrupt (specific to your hardware)
      // Example: write to a register in your interrupt controller
      // *(volatile uint32_t *)YOUR_INTERRUPT_CONTROLLER_ADDRESS = 1 << 11;
  reg_write32(0x2024, 1 << 1);
  PLIC_completeIRQ(hart_id, irq_id);
  HAL_CORE_clearIRQ(11);
}

void trap_handler(void) {
  uint64_t hart_id = READ_CSR("mhartid");
  uint64_t m_cause = READ_CSR("mcause");
  uint64_t mepc = READ_CSR("mepc");
  printf("Trap handler: hart_id = %ld, cause = 0x%lx, mepc: 0x%lx\n", hart_id, m_cause, mepc);
  // if the first bit is set 
  // if ((cause >> (MXLEN-1)) && ((cause & 0xFF) == 11)) { // External interrupt (interrupt bit set and cause code 11)
  //   external_interrupt_handler();
  // }
  // else {
  //   // Handle other traps or exceptions if needed
  //   printf("Unhandled trap: cause = 0x%lx\n", cause);
  // }
  switch (m_cause) {
    case 0x00000000UL:      // instruction address misaligned
      printf("instructionAddressMisalignedException\n");
      break;
    case 0x00000001UL:      // instruction access fault
      printf("instructionAccessFaultException\n");
      break;
    case 0x00000002UL:      // illegal instruction
      printf("illegalInstructionException\n");
      break;
    case 0x00000003UL:      // breakpoint
      printf("breakpointException\n");
      break;
    case 0x00000004UL:      // load address misaligned
      printf("loadAddressMisalignedException\n");
      break;
    case 0x00000005UL:      // load access fault
      printf("loadAccessFaultException\n");
      break;
    case 0x00000006UL:      // store/AMO address misaligned
      printf("storeAMOAddressMisalignedException\n");
      break;
    case 0x00000007UL:      // store/AMO access fault
      printf("storeAMOAccessFaultException\n");
      break;
    case 0x00000008UL:      // environment call from U-mode
      printf("environmentCallUModeException\n");
      break;
    case 0x0000000BUL:      // environment call from M-mode
      printf("environmentCallMModeException\n");
      break;
    case 0x0000000CUL:      // instruction page fault
      printf("instructionPageFaultException\n");
      break;
    case (1UL << (RISCV_XLEN-1)) | 0x00000003UL:      // machine software interrupt
      printf("machineSoftwareInterruptCallback\n");
      break;
    case (1UL << (RISCV_XLEN-1)) | 0x00000007UL:      // machine timer interrupt
      printf("machineTimerInterruptCallback\n");
      break;
    case (1UL << (RISCV_XLEN-1)) | 0x0000000BUL:      // machine external interrupt
      printf("machineExternalInterruptCallback\n");
      external_interrupt_handler();
      break;
    default:
      break;
  }
}

void enable_external_interrupts(void) {
  uint32_t mie = READ_CSR("mie");
  mie |= MEIE;
  WRITE_CSR("mie", mie); 

  uint32_t mstatus = READ_CSR("mstatus");
  mstatus |= (1 << 3); // Enable MIE
  WRITE_CSR("mstatus", mstatus); 
}

void configure_counter() {
  printf("Configuring counter...\n");
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, ORIGIN_IMG_SIZE);
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_1, ORIGIN_IMG_SIZE);
}

void send_img_req_left() {
  // printf("Requesting image...\n");
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LEFT);
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_img_req_right() {
  // printf("Requesting image...\n");
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_RIGHT);
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, 0);
}

void send_img_loopback_1_row(int row) {
  printf("Sending 1 row...\n");
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, CS_CAMERA_LOOPBACK);
  while (ROSE_TX_ENQ_READY == 0) {}
  reg_write32(ROSE_TX_DATA_ADDR, STEREO_IMG_WIDTH);
  for (int i = 0; i < STEREO_IMG_WIDTH; i += 4) {
    while (ROSE_TX_ENQ_READY == 0) {}
    uint32_t data = (stereo_buf[row * STEREO_IMG_WIDTH + i] << 24)
                  | (stereo_buf[row * STEREO_IMG_WIDTH + i + 1] << 16)
                  | (stereo_buf[row * STEREO_IMG_WIDTH + i + 2] << 8)
                  | stereo_buf[row * STEREO_IMG_WIDTH + i + 3];
    reg_write32(ROSE_TX_DATA_ADDR, data);
  }
}

void send_img_loopback(uint32_t *img) {
    // printf("Requesting image...\n");
  for (int j = 0; j < (IMG_HEIGHT - BLOCK_SIZE); j += 1) {
    send_img_loopback_1_row(j);
  }
  printf("Sent %d rows\n", IMG_HEIGHT - BLOCK_SIZE);
}

void recv_img_dma_left(int offset){
  uint8_t *pointer = ROSE_DMA_BASE_ADDR_0 + offset * ORIGIN_IMG_SIZE;
  printf("offset for this access is: %d\n", offset);
  memcpy(origin_left_buf, pointer, ORIGIN_IMG_SIZE);
}

void recv_img_dma_right(int offset){
  uint8_t *pointer = ROSE_DMA_BASE_ADDR_1 + offset * ORIGIN_IMG_SIZE;
  printf("offset for this access is: %d\n", offset);
  memcpy(origin_right_buf, pointer, ORIGIN_IMG_SIZE);
}


int main(void) {
  uint32_t hart_id = READ_CSR("mhartid");
  printf("Hello World from hart %d!\n", hart_id);

  PLIC_enable(hart_id, 4);
  PLIC_enable(hart_id, 3);
  PLIC_setPriority(4, 1);
  PLIC_setPriority(3, 2);

  setup_trap_vector();
  enable_external_interrupts();
  configure_counter();
  send_img_req_left();
  
  while (1) {

  }
}

