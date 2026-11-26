"""Differential test of a shader replacement against the game's original shaders on real D3D9.

Compiles the replacements exactly as the plugin does (backend_test --compile-replacements), pairs each target with every
build of that shader name in the sdp2spv cache (optionally one package), and runs tools/shaderdiff with the
replacement's SPIR-V in place of the translation: the original bytecode on D3D9 against the replacement on Vulkan, same
random inputs.

usage: python -I tools/replacediff.py <replacement id> [--package NNN] [--only NAME[,NAME...]] [--const REG=x,y,z,w;...]
  run from the repository root; uses build/cache_rtx5070, build/backend_test32/backend_test.exe,
  build/shaderdiff/shaderdiff.exe; work files in build/replacediff/<id>.
  --const pins float constants in both APIs (e.g. a replacement parameter at c223 to its vanilla setting).
  --sun MODE runs tools/shaderdiff's sun shadow test (off, shadow, blend, splitx, splity) on the targets that declare the
  sun shadow bindings (FNV_SUN); --sun-strength sets SHADERDIFF_SUN_STRENGTH.
"""
import argparse
import collections
import csv
import json
import os
import shutil
import struct
import subprocess
import sys


def spirv_bindings(path):
    """(set, binding) of every resource the module declares (OpDecorate DescriptorSet / Binding)."""
    w = open(path, 'rb').read()
    words = struct.unpack('<%dI' % (len(w) // 4), w)
    sets, binds = {}, {}
    i = 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        if op == 71 and n >= 4:  # OpDecorate target decoration literal
            target, deco = words[i + 1], words[i + 2]
            if deco == 34:
                sets[target] = words[i + 3]
            elif deco == 33:
                binds[target] = words[i + 3]
        i += max(n, 1)
    return sorted((sets.get(t, 0), b) for t, b in binds.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('id')
    ap.add_argument('--package')
    ap.add_argument('--only')
    ap.add_argument('--const', default='')
    ap.add_argument('--sun')
    ap.add_argument('--sun-strength')
    a = ap.parse_args()
    cache = 'build/cache_rtx5070'
    work = os.path.join('build', 'replacediff', a.id)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, 'spv'))
    rc = subprocess.run([os.path.abspath('build/backend_test32/backend_test.exe'), '--compile-replacements', 'replacements',
                         os.path.join(work, 'compiled')], capture_output=True, text=True)
    for line in rc.stdout.splitlines():
        if a.id in line or 'NOT loaded' in line:
            print(line)
    if rc.returncode != 0:
        raise SystemExit('replacements did not compile')
    compiled = os.path.join(work, 'compiled', a.id)
    targets = {f[:-4]: os.path.join(compiled, f) for f in os.listdir(compiled) if f.endswith('.spv')}
    if a.only:
        keep = set(a.only.split(','))
        targets = {k: v for k, v in targets.items() if k in keep}
    m = json.load(open(os.path.join(cache, 'manifest.json'), encoding='utf-8'))['shaders']
    import configparser
    ini = configparser.ConfigParser(strict=False, interpolation=None)
    ini.optionxform = str
    ini.read(os.path.join('replacements', a.id, 'replace.ini'))
    wanted = None
    if a.package:
        wanted = {a.package}
    else:
        pk = ini.get('Replacement', 'Packages', fallback='')
        if pk.strip():
            wanted = {'%03d' % int(x) for x in pk.replace(',', ' ').split()}
    jobs = []
    names = {}
    for h, s in sorted(m.items()):
        for full in s['names']:
            pkg, nm = full.split('/')
            if nm not in targets or (wanted and pkg not in wanted):
                continue
            spv = targets[nm]
            kinds = {t['reg']: t['kind'] for t in s['textures']}
            cbvs, tex = [], []
            binds = spirv_bindings(spv)
            # The sun shadow block (9; its cascades 48 / 49 are also the wet world's rain cover, block 12).
            sun = s.get('sunLight', -1) if (1, 9) in binds else -1
            if (1, 9) in binds and sun < 0:
                # Lighting30Shader's builds (the hair): no PSLightColor, so the sun shadow pass does not mark them; the
                # sun's colour is LightData[0] (docs/LIGHTING.md "Hair").
                sun = next((c['index'] for c in s['ctab'] if c['name'] == 'LightData' and c['set'] == 'float4'), -1)
            if (1, 9) in binds and sun < 0:
                raise SystemExit('%s declares the sun shadow bindings but the original has no sun light colour' % nm)
            for st, b in binds:
                if st != 1:
                    continue
                # The blocks (13: the skin buffer's, with 61 / 62), the skin's detail map (59 / 60), the hair's tangent
                # shift map (63 / 64).
                if b < 16 or b == 59 or b == 63:
                    cbvs.append(b)
                elif b < 32:
                    reg = b - 16
                    if reg not in kinds:
                        raise SystemExit('%s samples s%d, which the original does not declare' % (nm, reg))
                    tex.append((reg, kinds[reg]))
            # The original's defs, as the plugin overlays them: not at the target's GameConstants registers.
            game = {int(x) for x in ini.get('Shader.' + nm, 'GameConstants', fallback='').replace(',', ' ').split()}
            defs = ','.join('/'.join(repr(float(x)) if i else str(int(x)) for i, x in enumerate(d))
                            for d in s.get('defs', []) if int(d[0]) not in game)
            shutil.copy(spv, os.path.join(work, 'spv', h + '.spv'))
            shutil.copy(os.path.join(cache, 'spv', h + '.dxso'), os.path.join(work, 'spv', h + '.dxso'))
            jobs.append('%s %s %s %s %s %d' % (h, s['profile'], ','.join(map(str, sorted(set(cbvs)))) or '-',
                                                ','.join('%d:%d' % t for t in sorted(set(tex))) or '-', defs or '-', sun))
            names[h] = full
            break
    jobs_path = os.path.join(work, 'jobs.txt')
    open(jobs_path, 'w').write('\n'.join(jobs) + '\n')
    print('%d targets, %d builds to compare' % (len(targets), len(jobs)))
    env = dict(os.environ)
    if a.const:
        env['SHADERDIFF_CONST'] = a.const
    if a.sun:
        env['SHADERDIFF_SUN'] = a.sun
    if a.sun_strength:
        env['SHADERDIFF_SUN_STRENGTH'] = a.sun_strength
    results = os.path.join(work, 'results.csv')
    subprocess.run([os.path.abspath('build/shaderdiff/shaderdiff.exe'), jobs_path, os.path.join(work, 'spv'), results], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(results, encoding='utf-8')))
    status = collections.Counter(r['status'] for r in rows)
    print('results: %d  %s' % (len(rows), '  '.join('%s=%d' % kv for kv in sorted(status.items()))))
    for r in sorted(rows, key=lambda r: names.get(r['hash'], '')):
        if r['status'] != 'ok':
            print('  %-8s %-22s mism=%4s robust=%4s max=%-10s at (%s,%s) ref %s got %s %s' % (
                r['status'], names.get(r['hash'], r['hash']), r['mismatched'], r['robust'], r['maxAbs'], r['firstX'],
                r['firstY'], r['ref'], r['got'], r['error'][:200]))
    return 0 if status.get('ok', 0) == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
