/**
 * @file rv/arch.h
 * @brief RISC-V Architecture Configuration
 *
 * This header file provides architecture-specific configuration macros for the RISC-V architecture.
 * It defines macros and constants based on the target RISC-V architecture's word length (xlen),
 * allowing easy adaptation of code to different RISC-V architectures.
 *
 * The macros provided include load and store instructions, register byte size, and other architecture-specific settings.
 * These macros enable conditional compilation of code sections based on the target architecture,
 * ensuring compatibility and efficient utilization of resources across different RISC-V architectures.
 *
 * @note This file should be included to configure architecture-specific settings and enable conditional compilation
 *       based on the target RISC-V architecture.
 *
 * @author -T.K.-
 * @date 2024-05-29
 */

#ifndef __RV_ARCH_H
#define __RV_ARCH_H

#ifdef __riscv_xlen
  #define RISCV_XLEN __riscv_xlen
#else
  #warning "__riscv_xlen not defined, defaulting to 64"
  #define RISCV_XLEN 64
#endif

#if RISCV_XLEN == 64
  #define LREG ld
  #define SREG sd
  #define REGBYTES 8
#elif RISCV_XLEN == 32
  #define LREG lw
  #define SREG sw
  #define REGBYTES 4
#else
  #error "Unsupported RISCV_XLEN"
#endif

#endif /* __RV_ARCH_H */