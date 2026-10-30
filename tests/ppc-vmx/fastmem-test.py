#!/usr/bin/env python3
"""Fastmem (PPC_FASTMEM=1) guest-visible checks on a user-mode page table.

1. A page-crossing lwz and stw into an unmapped page raise a DSI with the
   same registers and untouched memory with PPC_FASTMEM=0 and =1.
2. A gdb watchpoint on a page already in the fastmem window fires once.
Needs clang with the PowerPC target and gdb; no linker or objcopy.
"""
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run import Monitor, tool

here = Path(__file__).resolve().parent
qemu = Path(sys.argv[1] if len(sys.argv) > 1 else here / '../../build/qemu-system-ppc').resolve()
BASE = 0xfff00100
FAULT_LOAD, FAULT_STORE, MARKER, DONE = BASE + 0x200, BASE + 0x210, BASE + 0x200, BASE + 0x220
MEM = [0x01020304, 0x05060708]


def rom(tmp, watch):
    obj = tmp / f'fastmem-{watch}.o'
    subprocess.run([tool('clang'), '--target=powerpc-unknown-none-elf', '-mcpu=7450',
                    f'-DWATCH={watch}', '-c', str(here / 'fastmem.S'), '-o', str(obj)], check=True)
    d = obj.read_bytes()
    shoff, = struct.unpack('>I', d[0x20:0x24])
    shentsize, shnum, shstrndx = struct.unpack('>HHH', d[0x2e:0x34])
    sh = lambda i: struct.unpack('>IIIIII', d[shoff + i * shentsize:shoff + i * shentsize + 24])
    names = sh(shstrndx)[4]
    for i in range(shnum):
        name, _, _, _, off, size = sh(i)
        if d[names + name:].split(b'\0', 1)[0] == b'.text.start':
            out = tmp / f'fastmem-{watch}.bin'
            out.write_bytes(bytes(0x100) + d[off:off + size])
            return out
    raise RuntimeError('.text.start not found')


def qemu_args(image, extra=()):
    return [str(qemu), '-M', 'mac99', '-cpu', '7450', '-m', '64', '-bios', str(image),
            '-object', 'memory-backend-shm,id=m,size=64M', '-machine', 'memory-backend=m',
            '-display', 'none', '-serial', 'none', *extra]


def dsi_case(tmp, image, fastmem):
    sock = tmp / f'mon-{fastmem}.sock'
    vm = subprocess.Popen(qemu_args(image, ['-monitor', f'unix:{sock},server=on,wait=off']),
                          env=dict(os.environ, PPC_FASTMEM=fastmem), stdout=subprocess.DEVNULL)
    try:
        mon = Monitor(str(sock))
        for _ in range(200):
            if mon.words(0x30000, 1)[0] == 2:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f'PPC_FASTMEM={fastmem}: guest did not take two faults')
        return mon.words(0x30010, 12)
    finally:
        vm.kill()
        vm.wait()


def watch_case(tmp, image):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    vm = subprocess.Popen(qemu_args(image, ['-S', '-gdb', f'tcp:127.0.0.1:{port}']),
                          env=dict(os.environ, PPC_FASTMEM='1'), stdout=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        cmds = ['set pagination off', 'set confirm off', 'set architecture powerpc:common',
                'set endian big', f'target remote 127.0.0.1:{port}',
                f'break *{MARKER:#x}', 'continue', 'watch *(unsigned int *)0x10003000',
                f'break *{DONE:#x}', 'continue', 'continue', 'kill']
        gdb = subprocess.run(['gdb', '-batch', *sum((['-ex', c] for c in cmds), [])],
                             capture_output=True, text=True, timeout=120)
        return gdb.stdout + gdb.stderr
    finally:
        vm.kill()
        vm.wait()


with tempfile.TemporaryDirectory(prefix='ppc-fastmem-') as tmp:
    tmp = Path(tmp)
    image = rom(tmp, 0)
    got = {f: dsi_case(tmp, image, f) for f in ('0', '1')}
    want = [0x40000000, 0x10004000, FAULT_LOAD, 0xdeadbeef, *MEM,
            0x42000000, 0x10004000, FAULT_STORE, 0xdeadbeef, *MEM]
    for f, words in got.items():
        assert words == want, (f, [hex(w) for w in words])
    print('page-crossing lwz/stw into an unmapped page: DSI, memory untouched, '
          'identical with PPC_FASTMEM=0 and 1 PASS')
    if not shutil.which('gdb'):
        sys.exit('gdb not found: watchpoint case skipped')
    out = watch_case(tmp, rom(tmp, 1))
    hits = len(re.findall(r'^Old value = ', out, re.M))
    assert hits == 1 and f'Breakpoint 3, {DONE:#010x}' in out, out
    print('gdb watchpoint on a fastmem-mapped page fires once PASS')
