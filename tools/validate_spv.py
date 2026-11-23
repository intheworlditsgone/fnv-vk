"""Validate every SPIR-V module produced by sdp2spv with spirv-val (Vulkan 1.3 target env, scalar block layout).

The legacy-state constant buffers use scalar packing (vec3 at offset 12, 28-byte array strides), so the backend
requires the Vulkan 1.2 `scalarBlockLayout` feature -- the same hard requirement DXVK 3.x has
(dxvk_device_info.cpp: ENABLE_FEATURE(vk12, scalarBlockLayout, true)).

Usage: python -I tools/validate_spv.py build/shadercache
Exit code is non-zero if any module fails validation or the manifest and spv directory disagree.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys


def find_spirv_val():
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = os.path.join(sdk, "Bin", "spirv-val.exe")
        if os.path.exists(cand):
            return cand
    return "spirv-val"


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else "build/shadercache"
    manifest = json.load(open(os.path.join(cache, "manifest.json"), encoding="utf-8"))
    spv_dir = os.path.join(cache, "spv")
    val = find_spirv_val()

    hashes = sorted(h for h, s in manifest["shaders"].items() if s["ok"])
    missing = [h for h in hashes if not os.path.exists(os.path.join(spv_dir, h + ".spv"))]
    if missing:
        print(f"missing .spv for {len(missing)} manifest entries, e.g. {missing[:3]}")
        return 1

    def run(h):
        p = subprocess.run([val, "--target-env", "vulkan1.3", "--scalar-block-layout", os.path.join(spv_dir, h + ".spv")],
                           capture_output=True, text=True)
        return h, p.returncode, (p.stdout + p.stderr).strip()

    failures = []
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for h, rc, out in ex.map(run, hashes):
            if rc != 0:
                failures.append((h, out))

    print(f"spirv-val (vulkan1.3, scalar block layout): {len(hashes) - len(failures)}/{len(hashes)} valid")
    for h, out in failures[:20]:
        names = manifest["shaders"][h]["names"]
        print(f"FAIL {h} {names[0]} ({manifest['shaders'][h]['profile']})\n  {out[:600]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
