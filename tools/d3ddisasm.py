"""Disassemble D3D9 shader bytecode with the system d3dcompiler_47 (D3DDisassemble). Usage: d3ddisasm.py file.dxso"""
import ctypes
import sys


def disassemble(code: bytes) -> str:
    d3dc = ctypes.WinDLL("d3dcompiler_47.dll")
    blob = ctypes.c_void_p()
    hr = d3dc.D3DDisassemble(code, len(code), 0, None, ctypes.byref(blob))
    if hr != 0:
        raise OSError(f"D3DDisassemble failed: 0x{hr & 0xffffffff:08x}")
    vtbl = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    get_ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vtbl[3])
    get_size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vtbl[4])
    text = ctypes.string_at(get_ptr(blob), get_size(blob)).decode("ascii", "replace").rstrip("\0")
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(blob)
    return text


if __name__ == "__main__":
    print(disassemble(open(sys.argv[1], "rb").read()))
