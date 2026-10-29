/* Restart load/store faults using BATs or hashed pages (FAULT_MODE 1/2). */
#ifndef FAULT_MODE
#define FAULT_MODE 0
#endif
#ifndef ELEMENT_SIZE
#define ELEMENT_SIZE 0
#endif
.section .text.start,"ax"
.globl _start
_start:
    lis     1,0x0080
    lis     3,handler@h
    ori     3,3,handler@l
    rlwinm  3,3,0,6,29
    oris    3,3,0x4800
    ori     3,3,2
    li      4,0x300
    stw     3,0(4)
    dcbst   0,4
    sync
    icbi    0,4
    isync
    lis     3,0x1234
    ori     3,3,0x5678
    stw     3,0x1000(0)
    lis     3,0xaabb
    ori     3,3,0xccdd
    stw     3,0x1004(0)
    /* DBAT0 maps the test's first 128 MiB of RAM. */
    li      3,2
    mtspr   537,3
    li      3,0x1ffe
    mtspr   536,3
    sync
    isync
#if FAULT_MODE
    /* SR1 key=1, VSID=1: EA 0x10001000 hashes to PTEG 0x40000. */
    lis     3,4
    mtspr   25,3
    lis     4,0x4000
    ori     4,4,1
    mtsr    1,4
#if FAULT_MODE == 2
    lis     4,0x8000
    ori     4,4,0x80
    stw     4,0(3)
    li      4,0x1000         /* PP=0 denies the first load. */
    stw     4,4(3)
#endif
    sync
    isync
#endif
    lis     3,0x0200
    ori     3,3,0x2010
    mtmsr   3
    isync
    vspltisb 19,7
    li      8,0x2000
    stvx    19,0,8
    lvx     19,0,8
    lis     9,0x1000
    lis     6,0xdead
    ori     6,6,0xbeef
#if ELEMENT_SIZE
    addi    8,9,0x1000 + ELEMENT_SIZE - 1
#endif
.globl fault_load
fault_load:
#if ELEMENT_SIZE == 1
    lvebx   19,0,8
#elif ELEMENT_SIZE == 2
    lvehx   19,0,8
#elif ELEMENT_SIZE == 4
    lvewx   19,0,8
#else
    lwz     6,0x1000(9)
#endif
    vaddubm 19,19,19
    lis     3,3
    addi    4,3,32
    lvx     20,0,4
    addi    4,3,48
    stvx    20,0,4
#if FAULT_MODE == 0
    li      4,0
    mtspr   538,4
#elif FAULT_MODE == 1
    lis     4,4
    li      5,0
    stw     5,0(4)           /* Remove the PTE before the store. */
#endif
#if FAULT_MODE
    addi    4,9,0x1000
    tlbie   4
#endif
    sync
    isync
#if ELEMENT_SIZE
    addi    8,9,0x1004 + ELEMENT_SIZE - 1
#endif
.globl fault_store
fault_store:
#if ELEMENT_SIZE == 1
    stvebx  19,0,8
#elif ELEMENT_SIZE == 2
    stvehx  19,0,8
#elif ELEMENT_SIZE == 4
    stvewx  19,0,8
#else
    stw     6,0x1004(9)
#endif
    vaddubm 19,19,19
    lwz     7,0x1004(9)
    stvx    19,0,3
    stw     6,16(3)
    stw     7,28(3)
    lis     4,0x564d
    ori     4,4,0x5846
    stw     4,24(3)
1:  b       1b

.text
handler:
    mtspr   272,3
    mtspr   273,4
    mfmsr   3
    oris    3,3,0x0200
    mtmsr   3
    isync
    lis     3,3
    addi    3,3,32
    stvx    19,0,3
    lwz     4,-12(3)
    addi    4,4,1
    stw     4,-12(3)
    /* Keep each fault's DSISR, DAR, SRR0 and uncommitted load result. */
    slwi    4,4,4
    add     3,3,4
    addi    3,3,16
    mfspr   4,18
    stw     4,0(3)
    mfspr   4,19
    stw     4,4(3)
    mfspr   4,26
    stw     4,8(3)
    stw     6,12(3)
#if FAULT_MODE
    lis     3,4
    li      4,0x1002
#if FAULT_MODE == 2
    /* First fault grants reads; the second also grants writes. */
    mfspr   4,18
    srwi    4,4,25
    andi.   4,4,1
    xori    4,4,1
    ori     4,4,0x1002
#endif
    stw     4,4(3)
    lis     4,0x8000
    ori     4,4,0x80
    stw     4,0(3)
    mfspr   4,19
    tlbie   4
#else
    /* DBAT1 maps EA 0x10000000 to physical address 0. */
    li      4,2
    mtspr   539,4
    lis     4,0x1000
    ori     4,4,2
    mtspr   538,4
#endif
    sync
    isync
    mfspr   3,272
    mfspr   4,273
    rfi
