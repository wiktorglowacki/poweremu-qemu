#!/usr/bin/env python3
"""Check grouped PPC loads/stores (PPC_LDST_GROUP) against the per-access path."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from run import Monitor, tool

here = Path(__file__).resolve().parent
qemu = Path(sys.argv[1] if len(sys.argv) > 1 else here / '../../build/qemu-system-ppc').resolve()


def val(addr):
    return (addr * 0x9e3779b1 & 0xffffffff) ^ 0x80010000


with tempfile.TemporaryDirectory(prefix='ppc-ldst-group-') as tmp:
    tmp = Path(tmp)
    obj, elf, rom = (tmp / name for name in ('guest.o', 'guest.elf', 'guest.bin'))
    subprocess.run([tool('clang'), '--target=powerpc-unknown-none-elf', '-mcpu=7450', '-c',
                    str(here / 'ldst-group.S'), '-o', str(obj)], check=True)
    subprocess.run([tool('ld.lld'), '-T', str(here / 'link.ld'), '-o', str(elf), str(obj)], check=True)
    subprocess.run([tool('llvm-objcopy'), '-O', 'binary', str(elf), str(rom)], check=True)
    fault_mid = int(re.search(r'^([0-9a-f]+) T fault_mid$',
                              subprocess.check_output([tool('llvm-nm'), str(elf)], text=True),
                              re.MULTILINE).group(1), 16)
    seen = {}
    for group in ('0', '1'):
        sock = tmp / f'mon-{group}.sock'
        vm = subprocess.Popen([str(qemu), '-M', 'mac99', '-cpu', '7450', '-m', '64',
                               '-bios', str(rom), '-display', 'none', '-serial', 'none',
                               '-monitor', f'unix:{sock},server=on,wait=off'],
                              env=dict(os.environ, PPC_LDST_GROUP=group), stdout=subprocess.DEVNULL)
        try:
            mon = Monitor(str(sock))
            for _ in range(100):
                res = mon.words(0x30000, 22)
                if res[0] == 0x4c445347:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f'PPC_LDST_GROUP={group}: guest did not finish: {res}')
            seen[group] = (res, mon.words(0x5000, 16), mon.words(0x5ff0, 8), mon.words(0x40000, 4))
        finally:
            vm.kill()
            vm.wait()
        res, page, cross, smc = seen[group]
        assert res[1:5] == [1, fault_mid, 0x02000000, 0x40000000], [hex(x) for x in res[1:5]]
        assert res[18:22] == [val(a) for a in range(0x01fffff8, 0x02000008, 4)], res[18:22]
        assert res[16:18] == [1, 2] and smc == [0x38600002, 0x60000000, 0x60000000, 0x4e800020], (res[16:18], smc)
        assert cross[2:6] == [val(a) for a in (0x6004, 0x6000, 0x5ffc, 0x5ff8)], cross
        h = lambda a: val(a & ~3) >> (16 - (a & 2) * 8) & 0xffff
        assert res[5:11] == [val(0x5000), val(0x5000), h(0x5006), 0xffff0000 | h(0x5018) if h(0x5018) & 0x8000
                             else h(0x5018), val(0x5018) & 0xff, val(0x503c)], [hex(x) for x in res[5:11]]
        assert page[8] == val(0x5000) and page[14] == h(0x5006), [hex(x) for x in page]
        print(f'PPC_LDST_GROUP={group}: fast run, run across a page, DSI inside a run, stores to a code page PASS')
    assert seen['0'] == seen['1'], seen
    print('PPC_LDST_GROUP=0 and 1 leave the same registers and memory PASS')
