#include <stdio.h>
#include <inttypes.h>

#define __IO volatile
#define __I volatile const

#define MXLEN 32

#define SET_BITS(REG, BIT)                    ((REG) |= (BIT))
#define CLEAR_BITS(REG, BIT)                  ((REG) &= ~(BIT))

#define READ_CSR(REG) ({                          \
  unsigned long __tmp;                            \
  asm volatile ("csrr %0, " REG : "=r"(__tmp));  \
  __tmp; })

#define WRITE_CSR(REG, VAL) ({                    \
  asm volatile ("csrw " REG ", %0" :: "rK"(VAL)); })

/* Peripheral Struct Definition */
typedef struct {
  __IO uint32_t priority_threshold;
  __IO uint32_t claim_complete;
} PLIC_ContextControlEntry_TypeDef;

typedef struct {
  __IO uint32_t priorities[1024];
  __I  uint32_t pendings[1024];
  __IO uint32_t enables[1024];
} PLIC_TypeDef;

// because the maximum struct size is 65535, we need to split PLIC content
typedef struct {
  PLIC_ContextControlEntry_TypeDef context_controls[1024];
} PLIC_ContextControl_TypeDef;

#ifndef PLIC_BASE
  #define PLIC_BASE                 0x0C000000U
  #define PLIC                      ((PLIC_TypeDef *)PLIC_BASE)
  #define PLIC_CC                   ((PLIC_ContextControl_TypeDef *)(PLIC_BASE + 0x00200000U))
#endif

void PLIC_disable(uint32_t hart_id, uint32_t irq_id) {
  uint32_t bit_index = irq_id & 0x1F;
  CLEAR_BITS(PLIC->enables[hart_id], 1 << bit_index);
}

void PLIC_enable(uint32_t hart_id, uint32_t irq_id) {
  uint32_t bit_index = irq_id & 0x1F;
  SET_BITS(PLIC->enables[hart_id], 1 << bit_index);
}

void PLIC_setPriority(uint32_t irq_id, uint32_t priority) {
  PLIC->priorities[irq_id] = priority;
}

void PLIC_setPriorityThreshold(uint32_t hart_id, uint32_t priority) {
  PLIC_CC->context_controls[hart_id].priority_threshold = priority;
}

uint32_t PLIC_claimIRQ(uint32_t hart_id) {
  return PLIC_CC->context_controls[hart_id].claim_complete;
}

void PLIC_completeIRQ(uint32_t hart_id, uint32_t irq_id) {
  PLIC_CC->context_controls[hart_id].claim_complete = irq_id;
}

void HAL_CORE_clearIRQ(uint32_t IRQn) {
  uint32_t mask = (1U << (uint32_t)IRQn);
  asm volatile("csrc mip, %0" :: "r"(mask));
}