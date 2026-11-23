"""Run inside IDA (FalloutNV.exe 1.4.0.525, unpacked): the permutation tables of the shader families the decomp has not
split out yet, read from Hex-Rays output of each LoadPixelShaders (a local array of BSShaderMacro rows: the source
file, then (name, definition) pairs ending at a null name). Row i compiles to each of the family's names: a format
with the row number (e.g. PAR2%03i.pso) and the defines that build adds (TRANSPARENCYMS for the transparency
multisampling builds, which the loaders append to the row).

Writes JSON {family: {"function": ea, "names": [[format, [[define, value]...]]...], "rows": {i: {"source",
"defines"}}}} to the path in FNV_SHADERTABLES_OUT. Consumed by tools/lightingtable.py (--families).

  PAR2     ParallaxShader::LoadPixelShaders        0x00BCD7C0
  SKIN2    SkinShader::LoadPixelShaders            0x00BCF320
  STB1     SpeedTreeBranchShader::LoadPixelShaders 0x00BD3C60 (rows as separate locals; stride 19 dwords)
  GRASS2   TallGrassShader::LoadShaders            0x00BAAFC0 (pixel rows: the array v41; 2.x and 3.0 builds, each
           also with TRANSPARENCYMS)
  STLEAF2  SpeedTreeLeafShader::LoadShaders        0x00BB13D0 (pixel rows: the leaf.p.hlsl locals; each also TMS)
  STFROND2 SpeedTreeFrondShader::LoadShaders       0x00BD2730 (pixel rows: kMacroArray_PS)
"""
import json
import os
import re

import ida_hexrays

TMS = [['TRANSPARENCYMS', '']]
# family, function, row layout, the array holding the pixel rows (None: any), names
FUNCS = [
    ('PAR2', 0x00BCD7C0, 'array', None, [['PAR2%03i.pso', []]]),
    ('SKIN2', 0x00BCF320, 'array', None, [['SKIN2%03i.pso', []]]),
    ('STB1', 0x00BD3C60, 'locals', None, [['STB1%03i.pso', []]]),
    ('GRASS2', 0x00BAAFC0, 'array', 'v41', [['GRASS2%03i.pso', []], ['GRASS2%03iTMS.pso', TMS],
                                             ['GRASS23x%03i.pso', []], ['GRASS23x%03iTMS.pso', TMS]]),
    ('STLEAF2', 0x00BB13D0, 'locals', None, [['STLEAF2%03i.pso', []], ['STLEAF2%03iTMS.pso', TMS]]),
    ('STFROND2', 0x00BD2730, 'array', 'kMacroArray_PS', [['STFROND2%03i.pso', []]]),
]


def unquote(v):
    v = v.strip()
    if v in ('&EmptyString', '0'):
        return ''
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1].replace('\\\\', '\\')
    return '<' + v + '>'  # computed at run time (e.g. SPECMULT from a float)


def array_rows(c, array=None):
    a = re.escape(array) if array else r'\w+'
    rows = {}
    for m in re.finditer(a + r'\[(\d+)\]\.strPath = "([^"]*)"', c):
        rows.setdefault(int(m.group(1)), {})['source'] = m.group(2).replace('\\\\', '\\')
    macros = {}
    for m in re.finditer(a + r'\[(\d+)\]\.kMacros\[(\d+)\]\.(Name|Definition) = ([^;]*);', c):
        macros.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {})[m.group(3)] = m.group(4)
    for i, r in rows.items():
        defs = []
        for j in sorted(macros.get(i, {})):
            d = macros[i][j]
            if d.get('Name', '0').strip() == '0':
                break
            defs.append([unquote(d['Name']), unquote(d.get('Definition', '0'))])
        r['defines'] = defs
    return rows


def local_rows(c):
    # v14[0] = "file"; then consecutive locals: name, definition, ..., 0 terminator; each row 19 dwords.
    offs = {}
    for m in re.finditer(r'(?:const char \*|char \*|int |_DWORD )\s*(v\d+)(?:\[\d+\])?; // \[esp\+\w+\] \[ebp-([0-9A-F]+)h\]', c):
        offs[m.group(1)] = -int(m.group(2), 16)
    stores = []
    for m in re.finditer(r'(v\d+)(\[(\d+)\])? = ([^;]*);', c):
        v, idx, val = m.group(1), m.group(3), m.group(4)
        if v not in offs:
            continue
        stores.append((offs[v] + 4 * int(idx or 0), val))
    files = sorted(o for o, val in stores if val.endswith('.p.hlsl"'))  # the pixel rows
    first = files[0]
    rows = {}
    for o, val in sorted(stores):
        k, r = divmod(o - first, 76)
        if not 0 <= k < 64:
            continue
        rows.setdefault(k, {})[r // 4] = val
    out = {}
    for k, r in rows.items():
        if 0 not in r or not r[0].endswith('.p.hlsl"'):
            continue
        defs = []
        j = 1
        while j in r and r[j].strip() != '0':
            defs.append([unquote(r[j]), unquote(r.get(j + 1, '0'))])
            j += 2
        out[k] = {'source': unquote(r[0]), 'defines': defs}
    return out


def main():
    res = {}
    for family, ea, kind, array, names in FUNCS:
        c = str(ida_hexrays.decompile(ea))
        rows = array_rows(c, array) if kind == 'array' else local_rows(c)
        res[family] = {'function': '0x%08X' % ea, 'names': names, 'rows': {str(k): rows[k] for k in sorted(rows)}}
    json.dump(res, open(os.environ.get('FNV_SHADERTABLES_OUT', 'shadertables.json'), 'w'), indent=1)
    return {k: len(v['rows']) for k, v in res.items()}


if __name__ == '__main__':
    print(main())
