#!/usr/bin/env python3
"""Check the inline lfs/stfs conversion against the helpers inside QEMU.

Builds tests/ppc-vmx/lfs-guest.s for a 32-bit PowerPC, lays it out as the
mac99 firmware image (no linker needed: the program is relocatable-free),
plays 4096 single and 4096 double bit patterns through the guest's lfs and
stfs, reads the results back with qtest, and compares them with
ppc_todouble()/ppc_tosingle() from target/ppc/fp32-conv.h (lfs-conv-ref.c).
Runs the whole thing twice, once with PPC_NATIVE_FP=0, and also compares
the two runs with each other.

The two tables cover every exponent of both formats with edge mantissas,
both signs: single 0x00000001..0x7fffffff with mantissa 0, 1, 0x100,
0x200000, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff, and double exponent
0..2047 with mantissa 0 and all ones.

usage: tests/ppc-vmx/lfs-qtest.py [path/to/qemu-system-ppc]

Needs clang with the PowerPC target, gcc, python3, and objdump for the
optional disassembly dump.  Takes a few seconds.
"""

import base64
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QEMU = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(HERE, "..", "..", "build", "qemu-system-ppc")
)
N = 4096
IN_S, IN_D = 0x100000, 0x110000
OUT_D, OUT_S = 0x200000, 0x210000
DONE, GO = 0x220000, 0x230000


def build(tmp):
    """clang -c the guest, then flatten .text from the ELF object."""
    obj = os.path.join(tmp, "guest.o")
    subprocess.check_call(
        [
            "clang",
            "--target=powerpc-unknown-none-elf",
            "-mcpu=750",
            "-mbig-endian",
            "-c",
            os.path.join(HERE, "lfs-guest.s"),
            "-o",
            obj,
        ]
    )
    d = open(obj, "rb").read()
    assert d[:4] == b"\x7fELF" and d[5] == 2, "not a big-endian ELF object"
    (shoff,) = struct.unpack_from(">I", d, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from(">HHH", d, 0x2E)
    secs = [
        struct.unpack_from(">IIIIIIIIII", d, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names = secs[shstrndx][4]

    def name(off):
        s = d[names + off :]
        return s[: s.index(b"\0")].decode()

    for s in secs:
        if name(s[0]) == ".text":
            text = d[s[4] : s[4] + s[5]]
            break
    else:
        sys.exit(".text not found")
    for s in secs:
        if name(s[0]) in (".rel.text", ".rela.text") and s[5]:
            sys.exit("guest has relocations; it must be position independent")
    rom = os.path.join(tmp, "guest.bin")
    open(rom, "wb").write(bytes(0x100) + text)  # reset vector at 0xfff00100
    ref = os.path.join(tmp, "conv-ref")
    subprocess.check_call(
        ["cc", "-O2", "-o", ref, os.path.join(HERE, "lfs-conv-ref.c")]
    )
    return rom, ref


def tables():
    s, d = [], []
    for sign in (0, 1):
        for e in range(256):
            for m in (0, 1, 0x100, 0x200000, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF):
                s.append((sign << 31) | (e << 23) | m)
    for e in range(2048):
        d.append((0 << 63) | (e << 52))
        d.append((1 << 63) | (e << 52) | 0xFFFFFFFFFFFFF)
    assert len(s) == len(d) == N
    return s, d


class Qtest:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX)
        for _ in range(200):
            try:
                self.s.connect(path)
                break
            except OSError:
                time.sleep(0.05)
        else:
            sys.exit("could not connect to the qtest socket")
        self.f = self.s.makefile("rwb")
        self.cmd("endianness big")

    def cmd(self, line):
        self.f.write((line + "\n").encode())
        self.f.flush()
        r = self.f.readline().decode().strip()
        if not r.startswith("OK"):
            sys.exit("%s -> %s" % (line, r))
        return r

    def write(self, addr, data):
        self.cmd(
            "b64write 0x%x 0x%x %s" % (addr, len(data), base64.b64encode(data).decode())
        )

    def read(self, addr, n):
        r = self.cmd("b64read 0x%x 0x%x" % (addr, n))
        return base64.b64decode(r.split()[1])


def run(rom, s, d, env_extra):
    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "qtest")
    p = subprocess.Popen(
        [
            QEMU,
            "-M",
            "mac99",
            "-bios",
            rom,
            "-display",
            "none",
            "-serial",
            "none",
            "-monitor",
            "none",
            "-no-reboot",
            "-qtest",
            "unix:" + sock + ",server=on,wait=off",
        ],
        env=dict(os.environ, **env_extra),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        q = Qtest(sock)
        q.write(IN_S, struct.pack(">%dI" % N, *s))
        q.write(IN_D, struct.pack(">%dQ" % N, *d))
        q.write(GO, b"\0\0\0\1")
        for _ in range(400):
            if q.read(DONE, 4) == b"\xde\xad\xbe\xef":
                break
            time.sleep(0.05)
        else:
            sys.exit("guest did not finish")
        got_d = struct.unpack(">%dQ" % N, q.read(OUT_D, 8 * N))
        got_s = struct.unpack(">%dI" % N, q.read(OUT_S, 4 * N))
    finally:
        p.kill()
        p.wait()
    return got_d, got_s


def main():
    tmp = tempfile.mkdtemp()
    rom, ref = build(tmp)
    s, d = tables()
    out = subprocess.run(
        [ref],
        check=True,
        stdout=subprocess.PIPE,
        input=struct.pack("<%dI" % N, *s) + struct.pack("<%dQ" % N, *d),
    ).stdout
    want_d = struct.unpack("<%dQ" % N, out[: 8 * N])
    want_s = struct.unpack("<%dI" % N, out[8 * N :])

    bad = 0
    runs = []
    for label, env in (("", {}), ("PPC_NATIVE_FP=0", {"PPC_NATIVE_FP": "0"})):
        got_d, got_s = run(rom, s, d, env)
        runs.append((got_d, got_s))
        nbad = 0
        for x, g, w in zip(s, got_d, want_d):
            if g != w:
                nbad += 1
                if nbad < 6:
                    print("  lfs  0x%08x: guest 0x%016x want 0x%016x" % (x, g, w))
        for x, g, w in zip(d, got_s, want_s):
            if g != w:
                nbad += 1
                if nbad < 6:
                    print("  stfs 0x%016x: guest 0x%08x want 0x%08x" % (x, g, w))
        print(
            "%slfs/stfs %d patterns each: %d mismatches"
            % (label + ": " if label else "", N, nbad)
        )
        bad += nbad

    print("inline and helper runs identical:", runs[0] == runs[1])
    return 1 if bad or runs[0] != runs[1] else 0


if __name__ == "__main__":
    sys.exit(main())
