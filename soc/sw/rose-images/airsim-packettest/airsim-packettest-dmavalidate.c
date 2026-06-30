// Bridge RX VALIDATION via the DMA path (RoSE's real camera-data path).
//   1) configure the DMA buffer size,
//   2) request CS_CAMERA_LEFT (routed to channel 0 = DMA0),
//   3) wait for the DMA engine to fill a buffer (cam_buffer bit flips),
//   4) read the buffer from memory (0x88000000) and VALIDATE it equals the known
//      pattern (PATTERN_BASE + i) that PatternEnv serves,
//   5) re-echo a PASS/FAIL marker + the data over TX (header 0x42) so the result is
//      verifiable in the synchronizer log (buffering-immune).
#include "mmio.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "rose_port.h"
#include "rose_packet.h"

#define PATTERN_BASE 0xC0DE0000u
#define N 16                       // PatternEnv serves 16 words
#define BUF_BYTES (N * 4)
#define ECHO_HEADER 0x42
#define PASS_MARKER 0x600D600Du
#define FAIL_MARKER 0xBAD0BAD0u

static inline void tx_word(uint32_t w) {
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, w);
}

uint32_t buf[N];

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered so PASS/FAIL flushes immediately

  // 1) DMA buffer size = N words
  reg_write32(ROSE_DMA_CONFIG_COUNTER_ADDR_0, BUF_BYTES);

  // 2) request (routed to channel 0 = DMA0 by the synchronizer config)
  tx_word(CS_CAMERA_LEFT);
  tx_word(0);

  // 3) wait for the DMA engine to finish a buffer (cam_buffer bit flips)
  int status_prev = ROSE_DMA_BUFFER_0;
  int status = status_prev;
  do { status = ROSE_DMA_BUFFER_0; } while (status == status_prev);

  // 4) read buffer 0 from memory and validate against the known pattern
  volatile uint32_t *dma = (volatile uint32_t *)(uintptr_t)ROSE_DMA_BASE_ADDR_0;
  int ok = 1;
  for (int i = 0; i < N; i++) {
    buf[i] = dma[i];
    if (buf[i] != (PATTERN_BASE + (uint32_t)i)) ok = 0;
  }

  // 5) report result + exit cleanly (exit flushes; sim ends -> fast, no infinite loop)
  if (ok) {
    printf("DMAVALIDATE: PASS rx[0]=0x%08x rx[%d]=0x%08x\n", buf[0], N - 1, buf[N - 1]);
  } else {
    printf("DMAVALIDATE: FAIL rx[0]=0x%08x rx[1]=0x%08x\n", buf[0], buf[1]);
  }
  exit(ok ? 0 : 1);
  return 0;
}
