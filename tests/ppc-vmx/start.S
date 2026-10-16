/* Reset entry: MSR[VEC] and MSR[FP] on, stack in RAM, call main, spin. */
    .section .text.start, "ax"
    .globl _start
_start:
    lis     %r3, 0x0200         /* MSR_VEC */
    ori     %r3, %r3, 0x2000    /* MSR_FP */
    mtmsr   %r3
    isync
    lis     %r1, 0x0080         /* stack at 8 MB, real mode */
    li      %r0, 0
    stwu    %r0, -64(%r1)
    bl      main
1:  b       1b

/*
 * Alignment (0x600) and program (0x700) interrupts for the dead-store
 * check in vmx_test.c: save v19 and count at EXC_SAVE (0x30000), then
 * return past the faulting instruction.
 */
    .text
    .globl exc_handler
exc_handler:
    mtspr   272, %r3            /* SPRG0 */
    mtspr   273, %r4            /* SPRG1 */
    mfmsr   %r3
    oris    %r3, %r3, 0x0200    /* MSR_VEC: interrupts clear it */
    mtmsr   %r3
    isync
    lis     %r3, 3
    stvx    %v19, 0, %r3
    lwz     %r4, 16(%r3)
    addi    %r4, %r4, 1
    stw     %r4, 16(%r3)
    mfspr   %r3, 26             /* SRR0 */
    addi    %r3, %r3, 4
    mtspr   26, %r3
    mfspr   %r3, 272
    mfspr   %r4, 273
    rfi
