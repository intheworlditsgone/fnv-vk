"""Pack the sdp2spv cache (manifest.json + spv/<md5>.spv) into the runtime shader pack the plugin loads.

usage: packshaders.py <cache dir> <out.bin>

Format (little endian), version 2:
  char[4] 'FVKS'; u32 version; u32 count
  per shader:
    u8[16] md5 of the D3D9 bytecode (the key the game's bytecode is hashed to at runtime)
    u8 stage (0 vertex, 1 pixel); u8 nDefs; u8 nTextures; u8 nInputs
    u32 cbvMask      bit b = uniform binding b (0..8) is declared by the module
    u32 textureMask  bit s = sampler register s has image binding 16+s and sampler binding 32+s
    u32 spvBytes
    nDefs    x { u32 register; f32 value[4] }          `def` constants, overlaid on binding 0 before each draw
    nTextures x { u8 register; u8 kind }               kind: 5 = 2D, 9 = cube (dxbc-spirv resource kinds)
    nInputs  x { u8 usage; u8 usageIndex; u8 location; u8 components }   VS only; usage = D3DDECLUSAGE
    zero padding to a 4-byte file offset
    spv words
"""
import json
import os
import struct
import sys

USAGE = {'position': 0, 'weight': 1, 'blend': 2, 'normal': 3, 'psize': 4, 'texcoord': 5, 'tangent': 6,
         'binormal': 7, 'tessfactor': 8, 'positiont': 9, 'color': 10, 'fog': 11, 'depth': 12, 'sample': 13}


def main():
    cache, out = sys.argv[1], sys.argv[2]
    m = json.load(open(os.path.join(cache, 'manifest.json'), encoding='utf-8'))
    shaders = m['shaders']
    blob = bytearray()
    n = 0
    for h in sorted(shaders):
        s = shaders[h]
        if not s.get('ok'):
            raise SystemExit('shader %s did not translate' % h)
        prof = s['profile']
        stage = 0 if prof.startswith('vs') else 1
        spv = open(os.path.join(cache, 'spv', h + '.spv'), 'rb').read()
        assert len(spv) % 4 == 0 and spv[:4] == b'\x03\x02\x23\x07', h
        defs = s.get('defs', [])
        tex = s.get('textures', [])
        ins = [i for i in s['io'] if i['dir'] == 'in' and 'location' in i] if stage == 0 else []
        cbv = tmask = 0
        for b in s.get('bindings', []):
            if b['kind'] == 'cbv':
                cbv |= 1 << b['binding']
            elif b['kind'] == 'texture':
                tmask |= 1 << (b['binding'] - 16)
        for t in tex:
            assert t['kind'] in (5, 9), (h, t)
            assert tmask >> t['reg'] & 1, (h, t)
        blob += bytes.fromhex(h)
        blob += struct.pack('<BBBBIII', stage, len(defs), len(tex), len(ins), cbv, tmask, len(spv))
        for d in defs:
            blob += struct.pack('<I4f', d[0], *d[1:5])
        for t in tex:
            blob += struct.pack('<BB', t['reg'], t['kind'])
        for i in ins:
            blob += struct.pack('<BBBB', USAGE[i['semantic']], i['semanticIndex'], i['location'], i['components'])
        while (12 + len(blob)) % 4:
            blob += bytes(1)
        blob += spv
        n += 1
    with open(out, 'wb') as f:
        f.write(b'FVKS' + struct.pack('<II', 2, n) + blob)
    print('packed %d shaders, %d bytes -> %s' % (n, len(blob) + 12, out))


if __name__ == '__main__':
    main()
