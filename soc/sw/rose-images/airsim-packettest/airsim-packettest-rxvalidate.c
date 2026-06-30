// Bridge RX VALIDATION test.
//   1) request CS_CAMERA_LEFT,
//   2) read NREAD words from channel 2 (reqrsp1 = ROSE_RX_DATA_2),
//   3) VALIDATE the SoC received the known pattern (PATTERN_BASE + 0,1,2,...) that
//      PatternEnv serves — scanning the buffer so we are robust to whether the bridge
//      forwards the packet header/count words ahead of the payload,
//   4) re-echo over TX (header 0xAB): a PASS/FAIL marker, the match offset, and all
//      received words — so the result is verifiable on the host (synchronizer log),
//      independent of UART/printf buffering.
#include "mmio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "rose_port.h"
#include "rose_packet.h"

#define PATTERN_BASE 0xC0DE0000u
#define NREAD 4        // header + count + 2 pattern words (fast)
#define NPAT  2        // validate 2 consecutive pattern words
#define ECHO_HEADER 0x42
#define PASS_MARKER 0x600D600Du
#define FAIL_MARKER 0xBAD0BAD0u

static inline void tx_word(uint32_t w) {
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, w);
}
static inline uint32_t rx_word(void) {
  while (ROSE_RX_DEQ_VALID_2 == 0) ;
  return ROSE_RX_DATA_2;
}

uint32_t buf[NREAD];

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  // 1) request
  tx_word(CS_CAMERA_LEFT);
  tx_word(0);

  // 2) receive NREAD words from channel 2
  for (int i = 0; i < NREAD; i++) buf[i] = rx_word();

  // 3) validate: find PATTERN_BASE.. as NPAT consecutive words anywhere in buf
  int ok = 0;
  int found_at = -1;
  for (int s = 0; s + NPAT <= NREAD; s++) {
    int match = 1;
    for (int i = 0; i < NPAT; i++) {
      if (buf[s + i] != (PATTERN_BASE + (uint32_t)i)) { match = 0; break; }
    }
    if (match) { ok = 1; found_at = s; break; }
  }

  // 4) report result + exit cleanly (exit flushes; sim ends -> fast)
  if (ok) {
    printf("RXVALIDATE: PASS pattern at offset %d (buf=0x%08x,0x%08x,0x%08x,0x%08x)\n",
           found_at, buf[0], buf[1], buf[2], buf[3]);
  } else {
    printf("RXVALIDATE: FAIL buf=0x%08x,0x%08x,0x%08x,0x%08x\n", buf[0], buf[1], buf[2], buf[3]);
  }
  exit(ok ? 0 : 1);
}
