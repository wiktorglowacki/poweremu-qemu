#!/usr/bin/env python3
"""Check recoverable PPC load/store faults with live vector state."""
import os
import struct
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from run import Monitor, tool

here = Path(__file__).resolve().parent
qemu = Path(sys.argv[1] if len(sys.argv) > 1 else here / '../../build/qemu-system-ppc').resolve()

with tempfile.TemporaryDirectory(prefix='ppc-ldst-fault-') as tmp:
    tmp = Path(tmp)
    obj, elf, rom = (tmp / name for name in ('guest.o', 'guest.elf', 'guest.bin'))
    for mode, name, size in ((m, n, s) for m, n in enumerate(
            ('BAT missing', 'hash missing', 'hash protected')) for s in (0, 1, 2, 4)):
        subprocess.run([tool('clang'), '--target=powerpc-unknown-none-elf', '-mcpu=7450',
                        '-maltivec', f'-DFAULT_MODE={mode}', f'-DELEMENT_SIZE={size}', '-c', str(here / 'ldst-fault.S'),
                        '-o', str(obj)], check=True)
        subprocess.run([tool('ld.lld'), '-T', str(here / 'link.ld'), '-o', str(elf), str(obj)], check=True)
        subprocess.run([tool('llvm-objcopy'), '-O', 'binary', str(elf), str(rom)], check=True)
        symbols = subprocess.check_output([tool('llvm-nm'), str(elf)], text=True)
        pcs = {name: int(addr, 16) for addr, name in
               re.findall(r'^([0-9a-f]+) T (fault_\w+)$', symbols, re.MULTILINE)}
        dsisr = 0x08000000 if mode == 2 else 0x40000000
        for keep in ('0', '1'):
            sock = tmp / f'mon-{mode}-{size}-{keep}.sock'
            vm = subprocess.Popen([str(qemu), '-M', 'mac99', '-cpu', '7450', '-m', '64',
                                   '-bios', str(rom), '-display', 'none', '-serial', 'none',
                                   '-monitor', f'unix:{sock},server=on,wait=off'],
                                  env=dict(os.environ, PPC_LDST_KEEP=keep), stdout=subprocess.DEVNULL)
            try:
                mon = Monitor(str(sock))
                for _ in range(100):
                    words = mon.words(0x30000, 24)
                    if words[6] == 0x564d5846:
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError(f'{name} fault guest did not finish: {words}')
                loaded = bytes.fromhex('12345678')[:size] + bytes([7] * (16 - size))
                before_store = list(struct.unpack('>4I', bytes((x * 2) & 255 for x in loaded)))
                final = list(struct.unpack('>4I', bytes((x * 4) & 255 for x in loaded)))
                gpr6 = 0xdeadbeef if size else 0x12345678
                stored = int.from_bytes(bytes([14] * size) + bytes.fromhex('aabbccdd')[size:], 'big') if size else 0x12345678
                assert words[:4] == final, (size, words[:4], final)
                assert words[4:8] == [gpr6, 2, 0x564d5846, stored]
                assert words[8:12] == before_store
                assert words[12:16] == [0x07070707] * 4
                assert words[16:20] == [dsisr, 0x10001000, pcs['fault_load'], 0xdeadbeef]
                assert words[20:24] == [dsisr | 0x02000000, 0x10001004,
                                        pcs['fault_store'], gpr6]
                if mode:
                    assert mon.words(0x40000, 2) == [0x80000080, 0x1182]
                print(f'PPC_LDST_KEEP={keep}: {name} element={size} restart, fault registers and vector state PASS')
            finally:
                vm.kill()
                vm.wait()
