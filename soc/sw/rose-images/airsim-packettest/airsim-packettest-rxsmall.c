// RX-delivery probe (non-blocking, observability-robust):
//   1) send a CS_CAMERA_LEFT request,
//   2) continuously echo the STATUS register back over TX (header 0xBB).
// The host synchronizer sends a response routed to channel 2 and watches the
// echoed status words: if the DEQ_VALID_2 bit (0x2) ever sets, the RX/arbiter
// delivered the response to channel 2. Never blocks on a hanging RX read.
#include "mmio.h"
#include <stdint.h>
#include "rose_port.h"
#include "rose_packet.h"

#define STATUS_HEADER 0xBB

static inline void tx_word(uint32_t w) {
  while (ROSE_TX_ENQ_READY == 0) ;
  reg_write32(ROSE_TX_DATA_ADDR, w);
}

int main(void) {
  // 1) request
  tx_word(CS_CAMERA_LEFT);
  tx_word(0);

  // 2) continuously report the status register
  while (1) {
    uint32_t status = reg_read32(ROSE_STATUS_ADDR);   // 0x2000
    tx_word(STATUS_HEADER);
    tx_word(4);
    tx_word(status);
    for (volatile int d = 0; d < 20000; d++) ;        // pace
  }
  return 0;
}
