"""The game's lighting-shader permutation tables, from the decomp: every compiled SLS pixel shader with the source file
and the defines it was built from.

ShadowLightShader::LoadPixelShaders (0x00B7..., src/bsshader/shadowlightshader.cpp in the byte-matching decomp) holds
two tables, shaderModel1Defs[48] and shaderModel2Defs[160]; row i of the first compiles to SLS1<iii>.pso, row i of
the second to SLS2<iii>.pso (rows 70..73 and 152: SLS2<iii>s<passes>.pso). Each row is a source file followed by
(name, value) define pairs. The shipped packages hold the compiled results, not the sources; this table is what each one
is.

The parallax, skin and SpeedTree branch families (PAR2<iii>, SKIN2<iii>, STB1<iii>) compile the same sources with
their own tables, which the decomp has not split out: tools/ida_shadertables.py reads them from the binary in IDA, and
--families <that JSON> adds them.

usage: python lightingtable.py <decomp src dir> <Data/Shaders dir> <package number> <out.json> [--families <json>]
"""
import json
import os
import re
import struct
import sys


def tables(src):
    text = open(os.path.join(src, 'bsshader', 'shadowlightshader.cpp'), encoding='utf-8', errors='replace').read()
    out = {}
    for prefix, name in (('SLS1', 'shaderModel1Defs'), ('SLS2', 'shaderModel2Defs')):
        m = re.search(r'const char\* ' + name + r'\[(\d+)\]\[\d+\] = \{(.*?)\n    \};', text, re.S)
        if not m:
            raise SystemExit('table %s not found' % name)
        rows = re.findall(r'\{([^{}]*)\}', m.group(2))
        if len(rows) != int(m.group(1)):
            raise SystemExit('%s: %d rows, expected %s' % (name, len(rows), m.group(1)))
        for i, row in enumerate(rows):
            toks = [t.strip() for t in row.split(',') if t.strip()]
            if not toks or toks[0] == '0':
                continue
            source = re.sub(r'^shader_', '', toks[0])
            source = re.sub(r'_hlsl$', '.hlsl', source)
            defines = {}
            rest = [t for t in toks[1:]]
            while rest and rest[0] != '0':
                key = re.sub(r'^shader_', '', rest[0])
                val = rest[1] if len(rest) > 1 else '0'
                rest = rest[2:]
                if val in ('0', 'shader_empty'):
                    val = ''
                elif val.startswith('shader_'):
                    val = val[len('shader_'):].replace('_', '.')
                defines[key] = val
            out['%s%03d' % (prefix, i)] = {'row': i, 'source': source, 'defines': defines}
    out.update(vertex_table(text))
    return out


def vertex_table(text):
    """ShadowLightShader::LoadVertexShaders: shaderEntries[156] of {f00 source, f01.. (name, value) pairs ending at 0};
    rows 0..102 compile to SLS2<iii>.vso, 103..124 to SLS1S<iii>.vso (i - 103), 125..155 to SLS1<iii>.vso (i - 125).
    Keyed by the full shader name."""
    body = text[text.index('ShadowLightShader_LoadVertexShaders()\n{'):]
    body = body[:body.index('sprintf_s')]
    fields = {}
    for m in re.finditer(r'shaderEntries\[(\d+)\]\.f(\d\d) = ("(?:[^"\\]|\\.)*"|0);', body):
        fields.setdefault(int(m.group(1)), {})[int(m.group(2))] = m.group(3)
    out = {}
    for i, f in fields.items():
        if 0 not in f or f[0] == '0':
            continue
        source = f[0].strip('"').replace('\\\\', '_').replace('.v.hlsl', '_v.hlsl')
        defines = {}
        k = 1
        while f.get(k, '0') != '0':
            defines[f[k].strip('"')] = f.get(k + 1, '0').strip('"') if f.get(k + 1, '0') != '0' else ''
            k += 2
        name = 'SLS2%03d.vso' % i if i < 103 else 'SLS1S%03d.vso' % (i - 103) if i < 125 else 'SLS1%03d.vso' % (i - 125)
        out[name] = {'row': i, 'source': source, 'defines': defines}
    return out


def family_tables(path):
    """tools/ida_shadertables.py output -> {shader name: row}: each row under each of its family's names (a format with
    the row number, and the defines that build adds)."""
    out = {}
    for family, t in json.load(open(path)).items():
        for i, row in t['rows'].items():
            source = row['source'].replace(chr(92), '_').replace('.p.hlsl', '_p.hlsl')
            for fmt, extra in t['names']:
                defines = dict(row['defines'])
                defines.update(dict(extra))
                out[fmt % int(i)] = {'row': int(i), 'source': source, 'defines': defines}
    return out


def package_names(shaders, pkg):
    b = open(os.path.join(shaders, 'shaderpackage%03d.sdp' % pkg), 'rb').read()
    n = struct.unpack_from('<I', b, 4)[0]
    o = 12
    names = []
    for _ in range(n):
        names.append(b[o:o + 256].split(b'\0')[0].decode())
        o += 260 + struct.unpack_from('<I', b, o + 256)[0]
    return names


def main():
    args = sys.argv[1:]
    families = None
    if '--families' in args:
        i = args.index('--families')
        families = args[i + 1]
        del args[i:i + 2]
    src, shaders, pkg, out = args[0], args[1], int(args[2]), args[3]
    t = tables(src)
    fam = family_tables(families) if families else {}
    t.update(fam)
    names = package_names(shaders, pkg)
    shipped = {}
    for nm in names:
        m = re.match(r'(SLS[12]\d{3})(s\d)?\.pso$', nm)
        if m:
            shipped[nm] = m.group(1)
        elif nm in fam or (nm.startswith('SLS') and nm.endswith('.vso')):
            shipped[nm] = nm
    rows = {}
    missing = []
    for nm, key in sorted(shipped.items()):
        if key not in t:
            missing.append(nm)
            continue
        rows[nm] = t[key]
    unshipped = sorted(k for k in t if not any(v == k for v in shipped.values()))
    json.dump({'package': pkg, 'shaders': rows, 'notInTable': missing, 'tableRowsNotShipped': unshipped},
              open(out, 'w'), indent=1)
    by = {}
    for r in rows.values():
        by[r['source']] = by.get(r['source'], 0) + 1
    print('%d shipped lighting pixel shaders mapped, %d not in the table, %d table rows not shipped' %
          (len(rows), len(missing), len(unshipped)))
    for s, c in sorted(by.items(), key=lambda x: -x[1]):
        print('  %4d %s' % (c, s))
    if missing:
        print('not in the table:', ' '.join(missing))


if __name__ == '__main__':
    main()
