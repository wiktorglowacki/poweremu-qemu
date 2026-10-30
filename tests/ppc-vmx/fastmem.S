#ifndef WATCH
#define WATCH 0
#endif
.section .text.start,"ax"
.globl _start
_start:
    bl      1f
1:  mflr    3
    addi    3,3,handler-1b
    rlwinm  3,3,0,6,29
    oris    3,3,0x4800
    ori     3,3,2
    li      4,0x300
    stw     3,0(4)
    dcbst   0,4
    sync
    icbi    0,4
    isync
    lis     3,0x0102
    ori     3,3,0x0304
    li      4,0x3ffc
    stw     3,0(4)
    lis     3,0x0506
    ori     3,3,0x0708
    li      4,0x4000
    stw     3,0(4)
    lis     3,4
    mtspr   25,3
    lis     4,0x2000
    ori     4,4,1
    mtsr    1,4
    lis     5,0x8000
    ori     5,5,0x80
    li      6,2
    stw     5,0x40(3)
    stw     6,0x44(3)
    addi    6,6,0x1000
    stw     5,0(3)
    stw     6,4(3)
    addi    6,6,0x1000
    stw     5,0xc0(3)
    stw     6,0xc4(3)
    addi    6,6,0x1000
    stw     5,0x80(3)
    stw     6,0x84(3)
    lis     3,3
    li      4,0
    stw     4,0(3)
    sync
    isync
    lis     9,0x1000
    li      3,0x4010
    mtmsr   3
    isync
#if WATCH
    lwz     3,0x3000(9)
    stw     3,0x3000(9)
    li      4,0x55
    li      5,0x66
    b       marker
.org 0x200
marker:
    stw     4,0x3000(9)
    stw     5,0x3004(9)
    b       done
#else
    lwz     3,0x3ff8(9)
    lis     6,0xdead
    ori     6,6,0xbeef
    lis     7,0x1122
    ori     7,7,0x3344
    addi    8,9,0x3ffe
    b       fault_load
.org 0x200
fault_load:
    lwz     6,0(8)
    b       fault_store
.org 0x210
fault_store:
    stw     7,0(8)
    b       done
#endif
.org 0x220
done:
    b       done

.org 0x300
handler:
    mtspr   272,3
    mtspr   273,4
    mtspr   274,5
    lis     3,3
    lwz     4,0(3)
    mulli   5,4,24
    add     5,5,3
    addi    5,5,0x10
    mfspr   4,18
    stw     4,0(5)
    mfspr   4,19
    stw     4,4(5)
    mfspr   4,26
    stw     4,8(5)
    stw     6,12(5)
    li      4,0x3ffc
    lwz     4,0(4)
    stw     4,16(5)
    li      4,0x4000
    lwz     4,0(4)
    stw     4,20(5)
    lwz     4,0(3)
    addi    4,4,1
    stw     4,0(3)
    mfspr   4,26
    addi    4,4,4
    mtspr   26,4
    mfspr   3,272
    mfspr   4,273
    mfspr   5,274
    rfi
