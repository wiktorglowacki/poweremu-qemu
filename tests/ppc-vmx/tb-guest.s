/*
 * Timebase read check for the fast mftb path in hw/ppc/ppc.c.
 *
 * Runs from the mac99 reset vector in real mode, waits for the host to
 * write a control block at 0x300000, then spins `count` times in one of
 * several loop shapes and reports the timebase at both ends:
 *
 *   0x300000  GO (host writes 1)
 *   0x300004  mode
 *   0x300008  count
 *   0x30000c  wait ticks (mode 6)
 *   0x300020  t0, timebase low word before the loop
 *   0x300018  iterations done (mode 6)
 *   0x300024  t1, timebase low word after it
 *   0x300028  monotonic violations (mode 3)
 *   0x30002c  DONE, 0xdeadbeef
 *
 * modes: 0 empty loop (loop cost), 1 mftb, 2 mftbu/mftb/mftbu (what
 * MorphOS does), 3 mftb compared against the previous read, 4 mfspr 268
 * (the same SPR by its other encoding), 5 mftb + mftbu, 6 spin until
 * the timebase has advanced by the given number of ticks.
 *
 * Built by tb-qtest.py with clang --target=powerpc-unknown-none-elf.
 * 25 MHz timebase (mac99 TBFREQ), so one tick is 40 ns.
 */
    .section .text.start, "ax"
    .globl _start
_start:
    lis     %r3, 0x0200             /* MSR_VEC */
    ori     %r3, %r3, 0x2000        /* MSR_FP */
    mtmsr   %r3
    isync
    lis     %r1, 0x0080             /* stack at 8 MB, real mode */
    li      %r0, 0
    stwu    %r0, -64(%r1)

    lis     %r4, 0x0030             /* control block */
1:  lwz     %r5, 0(%r4)
    cmpwi   %r5, 0
    beq     1b

    mftb    %r20                    /* t0 */
    lwz     %r6, 4(%r4)             /* mode */
    lwz     %r7, 8(%r4)             /* count */
    mtctr   %r7
    mr      %r8, %r20               /* previous value */
    li      %r9, 0                  /* violations */

    cmpwi   %r6, 0
    beq     mode0
    cmpwi   %r6, 1
    beq     mode1
    cmpwi   %r6, 2
    beq     mode2
    cmpwi   %r6, 3
    beq     mode3
    cmpwi   %r6, 4
    beq     mode4
    cmpwi   %r6, 5
    beq     mode5
    cmpwi   %r6, 6
    beq     mode6
    b       done

mode0:
2:  bdnz    2b
    b       done
mode1:
2:  mftb    %r3
    bdnz    2b
    b       done
mode2:
2:  mftbu   %r10
    mftb    %r11
    mftbu   %r12
    bdnz    2b
    b       done
mode3:
2:  mftb    %r3
    cmplw   %r3, %r8
    bge     3f
    addi    %r9, %r9, 1
3:  mr      %r8, %r3
    bdnz    2b
    b       done
mode4:
2:  mfspr   %r3, 268
    bdnz    2b
    b       done
mode5:
2:  mftb    %r3
    mftbu   %r10
    bdnz    2b
    b       done
mode6:                              /* spin until the TB moved on */
    lwz     %r7, 0x0c(%r4)
    add     %r7, %r7, %r20          /* deadline in the low word */
    li      %r9, 0
6:  mftb    %r3
    addi    %r9, %r9, 1
    cmplw   %r3, %r7
    blt     6b
    stw     %r9, 0x18(%r4)

done:
    mftb    %r21                    /* t1 */
    stw     %r20, 0x20(%r4)
    stw     %r21, 0x24(%r4)
    stw     %r9, 0x28(%r4)
    lis     %r3, 0xdead
    ori     %r3, %r3, 0xbeef
    stw     %r3, 0x2c(%r4)
4:  b       4b
