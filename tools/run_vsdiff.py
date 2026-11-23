"""Run the vertex-shader differential test (shaderdiff --vs) over every translated vertex shader.

Reference: D3D9 software vertex processing (ProcessVertices). Candidate: the SPIR-V on Vulkan with transform
feedback. Usage: python -I tools/run_vsdiff.py [build/cache_rtx5070] [build/shaderdiff/shaderdiff.exe] [--viewvec]

--viewvec: only the shaders that got sdp2spv's view vector and read EyePosition (the SPECULAR ones), with LightData 0:
each one's view vector (location 12) must equal one of its texture coordinate outputs as D3D9 computes it (the half
vector, which is the view vector when no light is added).
"""
import collections
import csv
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d3ddisasm import disassemble  # noqa: E402


def written_masks(code: bytes) -> dict:
    """Component write masks per output, keyed 'P' (position) or interface location 1..11, from the D3D9
    disassembly. Only written components are compared: for unwritten ones the D3D9 software pipeline (the
    reference here) differs from hardware D3D9, and the translation follows hardware (verified separately)."""
    asm = [l.strip() for l in disassemble(code).splitlines() if l.strip() and not l.strip().startswith("//")]
    sem_of = {}  # vs_3_0: output register number -> location key
    for l in asm:
        m = re.match(r"dcl_(\w+?)(\d*)\s+o(\d+)", l)
        if m:
            usage, idx, reg = m.group(1), int(m.group(2) or 0), int(m.group(3))
            key = {"position": "P", "texcoord": str(idx + 1) if idx < 8 else None,
                   "color": str(9 + idx) if idx < 2 else None, "fog": "11"}.get(usage)
            if key:
                sem_of[reg] = key
    masks = collections.defaultdict(set)
    for l in asm:
        if l.startswith(("dcl", "def", "vs_")):
            continue
        parts = l.split(None, 1)
        if len(parts) < 2:
            continue
        dst = parts[1].split(",")[0].strip()
        m = re.match(r"(oPos|oFog|oPts|oT(\d)|oD(\d)|o(\d+))(?:\.([xyzw]+))?$", dst)
        if not m:
            continue
        reg, mask = m.group(1), m.group(5) or "xyzw"
        if reg == "oPos": key = "P"
        elif reg == "oFog": key, mask = "11", "x"
        elif m.group(2) is not None: key = str(int(m.group(2)) + 1)
        elif m.group(3) is not None: key = str(9 + int(m.group(3)))
        elif m.group(4) is not None: key = sem_of.get(int(m.group(4)))
        else: key = None
        if key:
            masks[key] |= set(mask)
    return {k: "".join(c for c in "xyzw" if c in v) for k, v in masks.items()}


def main():
    args = [a for a in sys.argv[1:] if a not in ("--viewvec", "--frame")]
    viewvec = "--viewvec" in sys.argv[1:]
    frame = "--frame" in sys.argv[1:]
    cache = args[0] if len(args) > 0 else "build/cache_rtx5070"
    exe = os.path.abspath(args[1] if len(args) > 1 else "build/shaderdiff/shaderdiff.exe")
    manifest = json.load(open(os.path.join(cache, "manifest.json"), encoding="utf-8"))
    jobs_path = os.path.join(cache, "vsdiff_jobs.txt")
    results_path = os.path.join(cache, "vsdiff_results.csv")

    count = 0
    with open(jobs_path, "w", encoding="utf-8") as f:
        for h, s in sorted(manifest["shaders"].items()):
            if not s["ok"] or not s["profile"].startswith("vs_"):
                continue
            if viewvec and not (s.get("viewVector") and any(c["name"] == "EyePosition" for c in s["ctab"])):
                continue
            if frame and not s.get("viewVector"):
                continue
            cbvs = sorted({b["binding"] for b in s["bindings"] if b["kind"] == "cbv"})
            ins = [v for v in s["io"] if v["dir"] == "in" and "location" in v]
            ins.sort(key=lambda v: v["location"])
            masks = written_masks(open(os.path.join(cache, "spv", h + ".dxso"), "rb").read())
            outs = [f"{k}:{masks[k]}" for k in sorted(masks, key=lambda k: -1 if k == "P" else int(k))]
            defs = ",".join("/".join(repr(float(x)) if i else str(int(x)) for i, x in enumerate(d)) for d in s.get("defs", []))
            f.write("%s %s %s %s %s %s\n" % (h, s["profile"], ",".join(map(str, cbvs)) or "-",
                                             ",".join("%s:%d:%d" % (v["semantic"], v["semanticIndex"], v["location"]) for v in ins),
                                             ",".join(outs) or "-", defs or "-"))
            count += 1
    print(f"{count} vertex-shader jobs")

    env = dict(os.environ)
    if viewvec:
        env["SHADERDIFF_VIEWVEC"] = "1"
    if frame:
        env["SHADERDIFF_FRAME"] = "1"
    rc = subprocess.run([exe, "--vs", jobs_path, os.path.join(cache, "spv"), results_path], env=env).returncode
    rows = list(csv.DictReader(open(results_path, encoding="utf-8")))
    status = collections.Counter(r["status"] for r in rows)
    print(f"results: {len(rows)}/{count}  " + "  ".join(f"{k}={v}" for k, v in sorted(status.items())))
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"max abs error among passing shaders: {max((float(r['maxAbs']) for r in ok), default=0):.3g}; "
          f"vertices skipped for |w|<1e-3: {sum(int(r['skippedW']) for r in rows)}")
    bad = sorted((r for r in rows if r["status"] != "ok"), key=lambda r: -int(r["mismatchedVerts"] or 0))
    out_names = {"-1": "position", "0": "normal"} | {str(i): f"texcoord{i-1}" for i in range(1, 9)} | \
                {"9": "color0", "10": "color1", "11": "fog", "12": "view vector"}
    for r in bad[:30]:
        names = manifest["shaders"][r["hash"]]["names"]
        print(f"  {r['status']:10} {names[0]:22} {r['profile']} verts={r['mismatchedVerts']:>3}/256 "
              f"first: v{r['firstVertex']} {out_names.get(r['firstOutput'], r['firstOutput'])} ref=[{r['ref']}] got=[{r['got']}] {r['error'][:160]}")
    return 0 if rc == 0 and len(rows) == count and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
