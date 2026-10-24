/*
 * Guest for lfs-qtest.py: 4096 lfs + stfd and 4096 lfd + stfs, results
 * left in memory for the harness to read back over qtest.
 *
 * Runs as mac99 firmware, so it starts at 0xfff00100 in real mode with
 * MSR[FP] clear; the first two instructions are the mtmsr that turns the
 * FPU on.  It waits for the harness to set the word at 0x00230000 before
 * it starts, because the tables are written after QEMU has started.
 *
 * Everything is an absolute address in the low 32 bits, so the image is
 * position independent and has no relocations.
 */
	.set	IN_S,  0x00100000	/* 4096 singles in  */
	.set	OUT_D, 0x00200000	/* 4096 doubles out */
	.set	IN_D,  0x00110000	/* 4096 doubles in  */
	.set	OUT_S, 0x00210000	/* 4096 singles out */
	.set	DONE,  0x00220000	/* 0xdeadbeef when finished */
	.set	GO,    0x00230000	/* harness sets this to 1 */
	.set	N,     4096

	.text
	.globl	_start
_start:
	li	0, 0x2000
	mtmsr	0
	lis	3, IN_S@h
	lis	4, OUT_D@h
	lis	5, IN_D@h
	lis	6, OUT_S@h
	lis	11, GO@h
	li	7, N
wait:
	lwz	12, 0(11)
	cmpwi	12, 0
	beq	wait
loop:
	lfs	1, 0(3)
	stfd	1, 0(4)
	lfd	2, 0(5)
	stfs	2, 0(6)
	addi	3, 3, 4
	addi	4, 4, 8
	addi	5, 5, 8
	addi	6, 6, 4
	addi	7, 7, -1
	cmpwi	7, 0
	bne	loop
	lis	9, 0xdead
	ori	9, 9, 0xbeef
	lis	10, DONE@h
	stw	9, 0(10)
1:	b	1b
