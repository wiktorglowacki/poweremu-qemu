/* Grouped loads/stores: fast path, a run across a page, a DSI inside a run, a run of stores to a code page. */
.section .text.start,"ax"
.globl _start
_start:
    lis     3,handler@h
    ori     3,3,handler@l
    rlwinm  3,3,0,6,29
    oris    3,3,0x4800
    ori     3,3,2
    li      4,0x300
    stw     3,0(4)
    lis     30,3
    li      4,0x5000
    li      5,256
    bl      fill
    li      4,0x5ff0
    li      5,8
    bl      fill
    lis     4,0x0200
    addi    4,4,-16
    li      5,8
    bl      fill
    lis     4,4
    lis     5,0x3860
    ori     5,5,1
    stw     5,0(4)
    lis     5,0x4e80
    ori     5,5,0x20
    stw     5,4(4)
    li      3,2
    mtspr   537,3
    li      3,0x3fe
    mtspr   536,3
    li      3,0x2010
    mtmsr   3
    isync
    li      9,0x5000
    lwzx    0,0,9
    lwz     10,0(9)
    lhz     11,6(9)
    lha     12,24(9)
    lbz     13,27(9)
    lfs     1,8(9)
    lfd     2,16(9)
    addi    9,9,32
    stw     10,0(9)
    sth     12,6(9)
    stb     13,7(9)
    stfs    2,8(9)
    stfd    1,16(9)
    stwu    11,24(9)
    lwzu    14,4(9)
    lwz     9,-60(9)
    stw     9,20(30)
    stw     10,24(30)
    stw     11,28(30)
    stw     12,32(30)
    stw     13,36(30)
    stw     14,40(30)
    stfd    1,48(30)
    stfd    2,56(30)
    li      9,0x5ff8
    lwz     15,0(9)
    lwz     16,4(9)
    lwz     17,8(9)
    lwz     18,12(9)
    stw     18,0(9)
    stw     17,4(9)
    stw     16,8(9)
    stw     15,12(9)
    lis     9,0x0200
    addi    9,9,-8
    lwz     19,0(9)
    lwz     20,4(9)
.globl fault_mid
fault_mid:
    lwz     21,8(9)
    lwz     22,12(9)
    lis     4,4
    mtctr   4
    bctrl
    stw     3,64(30)
    lwzx    0,0,4
    lis     23,0x3860
    ori     23,23,2
    lis     24,0x4e80
    ori     24,24,0x20
    lis     25,0x6000
    stw     23,0(4)
    stw     25,4(4)
    stw     25,8(4)
    stw     24,12(4)
    dcbst   0,4
    sync
    icbi    0,4
    isync
    bctrl
    stw     3,68(30)
    stw     19,72(30)
    stw     20,76(30)
    stw     21,80(30)
    stw     22,84(30)
    lis     4,0x4c44
    ori     4,4,0x5347
    stw     4,0(30)
1:  b       1b

fill:
    mtctr   5
    lis     6,0x9e37
    ori     6,6,0x79b1
1:  mullw   7,4,6
    xoris   7,7,0x8001
    stw     7,0(4)
    addi    4,4,4
    bdnz    1b
    blr

.text
handler:
    mtspr   272,3
    mtspr   273,4
    lis     3,3
    lwz     4,4(3)
    addi    4,4,1
    stw     4,4(3)
    mfspr   4,26
    stw     4,8(3)
    mfspr   4,19
    stw     4,12(3)
    mfspr   4,18
    stw     4,16(3)
    li      4,2
    mtspr   537,4
    li      4,0x7fe
    mtspr   536,4
    sync
    isync
    mfspr   3,272
    mfspr   4,273
    rfi
