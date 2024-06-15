.section .text
.global trap_entry
.global restore_context

trap_entry:
    # Save the context
    addi sp, sp, -128              # Adjust stack pointer
    sw ra, 124(sp)                 # Save return address
    sw t0, 120(sp)                 # Save temporary registers
    sw t1, 116(sp)
    sw t2, 112(sp)
    sw s0, 108(sp)                 # Save saved registers
    sw s1, 104(sp)
    sw a0, 100(sp)                 # Save argument registers
    sw a1, 96(sp)
    sw a2, 92(sp)
    sw a3, 88(sp)
    sw a4, 84(sp)
    sw a5, 80(sp)
    sw a6, 76(sp)
    sw a7, 72(sp)
    sw t3, 68(sp)
    sw t4, 64(sp)
    sw t5, 60(sp)
    sw t6, 56(sp)
    csrr t0, mcause                # Read the cause of the trap
    sw t0, 52(sp)                  # Save the cause register
    csrr t0, mepc                  # Read the program counter at the time of the trap
    sw t0, 48(sp)                  # Save the program counter

    # Call the C trap handler
    la t0, trap_handler            # Load address of trap_handler
    jr t0                          # Jump to trap_handler

restore_context:
    # Restore the context
    lw t0, 52(sp)                  # Restore the cause register (in case needed)
    lw t0, 48(sp)                  # Restore the program counter
    csrw mepc, t0                  # Write back the program counter
    lw ra, 124(sp)                 # Restore return address
    lw t0, 120(sp)                 # Restore temporary registers
    lw t1, 116(sp)
    lw t2, 112(sp)
    lw s0, 108(sp)                 # Restore saved registers
    lw s1, 104(sp)
    lw a0, 100(sp)                 # Restore argument registers
    lw a1, 96(sp)
    lw a2, 92(sp)
    lw a3, 88(sp)
    lw a4, 84(sp)
    lw a5, 80(sp)
    lw a6, 76(sp)
    lw a7, 72(sp)
    lw t3, 68(sp)
    lw t4, 64(sp)
    lw t5, 60(sp)
    lw t6, 56(sp)
    addi sp, sp, 128               # Adjust stack pointer back

    mret                            # Return from trap
