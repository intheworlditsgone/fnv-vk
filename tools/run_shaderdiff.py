"""Run the pixel-shader differential test (tools/shaderdiff) over every translated pixel shader.

Usage: python -I tools/run_shaderdiff.py [build/shadercache] [build/shaderdiff/shaderdiff.exe]
Writes <cache>/shaderdiff_jobs.txt, <cache>/shaderdiff_results.csv and prints a summary.
"""
import collections
import csv
import json
import os
import subprocess
import sys


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else "build/shadercache"
    exe = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "build/shaderdiff/shaderdiff.exe")
    manifest = json.load(open(os.path.join(cache, "manifest.json"), encoding="utf-8"))

    jobs_path = os.path.join(cache, "shaderdiff_jobs.txt")
    results_path = os.path.join(cache, "shaderdiff_results.csv")
    count = 0
    with open(jobs_path, "w", encoding="utf-8") as f:
        for h, s in sorted(manifest["shaders"].items()):
            if not s["ok"] or not s["profile"].startswith("ps_"):
                continue
            cbvs = sorted({b["binding"] for b in s["bindings"] if b["kind"] == "cbv"})
            tex = sorted({(t["reg"], t["kind"]) for t in s["textures"]})
            defs = ",".join("/".join(repr(float(x)) if i else str(int(x)) for i, x in enumerate(d)) for d in s.get("defs", []))
            f.write("%s %s %s %s %s %d\n" % (h, s["profile"], ",".join(map(str, cbvs)) or "-",
                                             ",".join("%d:%d" % t for t in tex) or "-", defs or "-", s.get("sunLight", -1)))
            count += 1
    print(f"{count} pixel-shader jobs")

    rc = subprocess.run([exe, jobs_path, os.path.join(cache, "spv"), results_path]).returncode
    if rc != 0:
        print(f"shaderdiff exited with {rc}")

    rows = list(csv.DictReader(open(results_path, encoding="utf-8")))
    status = collections.Counter(r["status"] for r in rows)
    print(f"results: {len(rows)}/{count}  " + "  ".join(f"{k}={v}" for k, v in sorted(status.items())))
    worst = max((float(r["maxAbs"]) for r in rows if r["status"] == "ok"), default=0.0)
    print(f"max abs error among passing shaders: {worst:.3g}")

    # Classification. Validation messages about depth-compare format support come from the SM3 frontend's
    # statically present (runtime-disabled) Dref sampling path; anything else is reported as-is.
    pixels = 64 * 64
    def klass(r):
        if r["status"] == "error":
            return "error"
        segments = [s for s in r["error"].split(" | ") if s.strip()]
        # VUID-vkCmdDraw-None-06479: image sampled with depth comparison lacks the DEPTH_COMPARISON format feature.
        dref_only = int(r["validation"]) > 0 and segments and all(s.startswith("VUID-vkCmdDraw-None-06479") for s in segments)
        other_val = int(r["validation"]) > 0 and not dref_only
        mism = int(r["mismatched"])
        if other_val:
            return "validation-other"
        if mism == 0:
            return "match" + ("+dref-static" if dref_only else "")
        # robust = mismatching pixels where the D3D9 reference itself is stable under +-1e-6 input perturbation.
        if int(r.get("robust") or 0) == 0:
            return "ill-conditioned" + ("+dref-static" if dref_only else "")
        return "ROBUST-MISMATCH" + ("+dref-static" if dref_only else "")
    classes = collections.Counter(klass(r) for r in rows)
    print("classes: " + "  ".join(f"{k}={v}" for k, v in sorted(classes.items())))
    with open(os.path.join(cache, "shaderdiff_classes.txt"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{klass(r)} {r['hash']} {manifest['shaders'][r['hash']]['names'][0]} {r['mismatched']} {r['maxAbs']}\n")
    nonfinite = sum(1 for r in rows if int(r.get("nonFinite") or 0) > 0)
    print(f"shaders with inf/NaN divergence only (counted as match): {nonfinite}")

    bad = [r for r in rows if not klass(r).startswith("match")]
    bad.sort(key=lambda r: -int(r["mismatched"]))

    # Tag each non-matching shader with the numerically sensitive constructs it contains.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from d3ddisasm import disassemble
    def features(h):
        asm = disassemble(open(os.path.join(cache, "spv", h + ".dxso"), "rb").read())
        lines = [l.strip() for l in asm.splitlines() if l.strip() and not l.strip().startswith("//")]
        ops = {l.split()[0] for l in lines}
        tags = []
        if "dcl_cube" in ops: tags.append("cube")
        if ops & {"rep", "loop"} and any(o.startswith("break") for o in ops): tags.append("loop-break")
        if "texldp" in ops or "texldp_pp" in ops: tags.append("texldp")
        if ops & {"dsx", "dsy", "dsx_pp", "dsy_pp", "texldd"}: tags.append("derivatives")
        if any(o.startswith("pow") for o in ops): tags.append("pow")
        if any(o.startswith("nrm") for o in ops): tags.append("nrm")
        if any(o.startswith("texkill") for o in ops): tags.append("texkill")
        if any(o.startswith(("cmp", "if", "setp")) for o in ops): tags.append("select")
        return tags
    tag_counts = collections.Counter()
    for r in bad:
        r["tags"] = features(r["hash"])
        tag_counts.update(r["tags"])
    print("constructs in non-matching shaders: " + "  ".join(f"{k}={v}" for k, v in tag_counts.most_common()))

    for r in bad[:40]:
        names = manifest["shaders"][r["hash"]]["names"]
        print(f"  {klass(r):18} {names[0]:22} {r['profile']} mism={r['mismatched']:>4} robust={r['robust']:>4} max={float(r['maxAbs']):<9.3g} "
              f"[{','.join(r['tags'])}] {r['error'][:120]}")
    return 0 if rc == 0 and len(rows) == count and not bad else 1  # any non-match needs a written explanation


if __name__ == "__main__":
    sys.exit(main())
