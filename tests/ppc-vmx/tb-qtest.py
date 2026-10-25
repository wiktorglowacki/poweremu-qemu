#!/usr/bin/env python3
"""Check the fast guest timebase (mftb) path of hw/ppc/ppc.c.

Builds tests/ppc-vmx/tb-guest.s for a 32-bit PowerPC, boots it from the
mac99 reset vector, and drives it over qtest.  Three things are checked,
each with the fast path on and with PPC_FAST_TB=0 (the exact
qemu_clock_get_ns path) as the reference:

  monotonic  a loop of 200M mftb reads, each compared with the previous
             one in the guest; zero backwards steps allowed.  At 25 MHz a
             low word wraps every 171.8 s and the run is a few seconds,
             so the low-word comparison is exact.

  rate       the timebase advance over a multi-second loop of mftb reads
             against the host's CLOCK_MONOTONIC elapsed time, both runs
             and both against the 25 MHz mac99 timebase frequency.  The
             host measures around the GO write and the DONE poll (1 ms
             apart), so the reference has a few ms of slack on seconds;
             the tolerance below is 0.5%.

  drift      a two second run with PPC_TB_STATS=1 also prints, once a
             second, the error of the extrapolated clock against the real
             virtual clock at the end of each 1 ms extrapolation window.
             The largest magnitude is checked against 1 us.

It also prints the cost of one read: the same loop with an empty body is
subtracted from the mftb loop's timebase advance.

usage: tests/ppc-vmx/tb-qtest.py [path/to/qemu-system-ppc]

Needs clang with the PowerPC target and python3.  Takes about a minute.
"""

import base64
import os
import re
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
TBFREQ = 25_000_000
CTL = 0x300000
DONE = 0xDEADBEEF

MODES = {
    0: "empty",
    1: "mftb",
    2: "mftbu/mftb/mftbu",
    3: "mftb monotonic",
    4: "mfspr 268",
    5: "mftb/mftbu",
}


def build(tmp):
    """clang -c the guest, then flatten .text from the ELF object."""
    obj = os.path.join(tmp, "guest.o")
    subprocess.check_call(
        [
            "clang",
            "--target=powerpc-unknown-none-elf",
            "-mcpu=7450",
            "-mbig-endian",
            "-c",
            os.path.join(HERE, "tb-guest.s"),
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

    text = b""
    for s in secs:
        if name(s[0]).startswith(".text") and s[5]:
            text = d[s[4] : s[4] + s[5]]
    assert text, ".text not found"
    for s in secs:
        if name(s[0]) in (".rel.text", ".rela.text") and s[5]:
            sys.exit("guest has relocations; it must be position independent")
    rom = os.path.join(tmp, "guest.bin")
    open(rom, "wb").write(bytes(0x100) + text)  # reset vector at 0xfff00100
    return rom


class Qtest:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX)
        for _ in range(400):
            try:
                self.s.connect(path)
                break
            except OSError:
                time.sleep(0.02)
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


def run(rom, mode, count, env_extra, wait_ticks=0):
    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "qtest")
    err = open(os.path.join(tmp, "stderr"), "w+b")
    p = subprocess.Popen(
        [
            QEMU,
            "-M",
            "mac99",
            "-cpu",
            "7450",
            "-m",
            "128",
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
        stderr=err,
    )
    try:
        q = Qtest(sock)
        q.write(CTL + 4, struct.pack(">IIII", mode, count, wait_ticks, 0))
        t0 = time.monotonic()
        q.write(CTL, struct.pack(">I", 1))
        for _ in range(120000):
            if q.read(CTL + 0x2C, 4) == struct.pack(">I", DONE):
                break
            time.sleep(0.001)
        else:
            sys.exit("guest did not finish")
        t1 = time.monotonic()
        iters = struct.unpack(">I", q.read(CTL + 0x18, 4))[0]
        lo0, lo1, fails = struct.unpack(">III", q.read(CTL + 0x20, 12))
    finally:
        p.kill()
        p.wait()
        err.seek(0)
        stats = err.read().decode(errors="replace")
        err.close()
    return (lo1 - lo0) & 0xFFFFFFFF, t1 - t0, fails, iters, stats


def stats_error(text):
    """Largest |error| reported by PPC_TB_STATS, in ns, or None."""
    worst = None
    for lo, hi in re.findall(r"error (-?\d+)\.\.(-?\d+) ns", text):
        for v in (int(lo), int(hi)):
            worst = abs(v) if worst is None else max(worst, abs(v))
    return worst


def main():
    tmp = tempfile.mkdtemp()
    rom = build(tmp)
    bad = 0

    print("cost of one read (25 MHz timebase, 40 ns per tick):")
    base = {}
    for fast in ("1", "0"):
        d, _, _, _, _ = run(rom, 0, 20_000_000, {"PPC_FAST_TB": fast})
        base[fast] = d / TBFREQ * 1e9 / 20_000_000
    for fast in ("1", "0"):
        n = 20_000_000
        d, _, _, _, _ = run(rom, 1, n, {"PPC_FAST_TB": fast})
        per = d / TBFREQ * 1e9 / n - base[fast]
        print("  PPC_FAST_TB=%s mftb %.2f ns, loop %.2f ns" % (fast, per, base[fast]))
        base["mftb" + fast] = per

    print("monotonic, 200M reads:")
    for fast in ("1", "0"):
        _, _, fails, _, _ = run(rom, 3, 200_000_000, {"PPC_FAST_TB": fast})
        print("  PPC_FAST_TB=%s %d backwards steps" % (fast, fails))
        bad += fails != 0

    print("rate over a long interval, against CLOCK_MONOTONIC:")
    # Two run lengths, differenced, so the fixed cost of starting the vCPU
    # drops out.  DONE is polled every millisecond, which puts a random
    # millisecond into each wall time; that is the 0.1% tolerance.  The
    # drift check below is the precise one (it compares against QEMU's own
    # virtual clock once per extrapolation window).
    measured = {}
    for fast in ("1", "0"):
        n = 400_000_000
        t1, h1, _, _, _ = run(rom, 1, n, {"PPC_FAST_TB": fast})
        t2, h2, _, _, _ = run(rom, 1, 2 * n, {"PPC_FAST_TB": fast})
        measured[fast] = (t2 - t1) / (h2 - h1)
        err = (measured[fast] / TBFREQ - 1) * 100
        ok = abs(err) < 0.1
        bad += not ok
        print(
            "  PPC_FAST_TB=%s %.0f Hz over %.2f s (%+.4f%%), %s"
            % (fast, measured[fast], h2 - h1, err, "ok" if ok else "FAIL")
        )
    skew = (measured["1"] / measured["0"] - 1) * 100
    ok = abs(skew) < 0.1
    bad += not ok
    print("  fast against exact: %+.4f%%, %s" % (skew, "ok" if ok else "FAIL"))

    print("extrapolation drift against the real clock (PPC_TB_STATS=1):")
    for fast in ("1",):
        _, _, _, _, stats = run(
            rom, 1, 150_000_000, {"PPC_FAST_TB": fast, "PPC_TB_STATS": "1"}
        )
        worst = stats_error(stats)
        ok = worst is not None and worst < 1000
        bad += not ok
        print(
            "  worst |error| at a window end: %s ns, %s"
            % (worst, "ok" if ok else "FAIL")
        )
        for line in stats.splitlines():
            if line.startswith("ppc-tb:"):
                print("   ", line)

    print("a 25 ms wait loop that polls mftb (its poll rate is elastic):")
    for fast in ("1", "0"):
        _, host, _, iters, _ = run(
            rom, 6, 0, {"PPC_FAST_TB": fast}, wait_ticks=625_000
        )
        print(
            "  PPC_FAST_TB=%s %d polls in %.3f s of host time, %.0f polls/s"
            % (fast, iters, host, iters / host)
        )

    print("FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
