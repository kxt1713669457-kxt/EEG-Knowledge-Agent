"""Windows凭据用当前用户DPAPI加密；浏览器和任务快照只接触连接ID。"""
import base64
import ctypes
import os


def _dpapi(value: bytes, *, decrypt=False):
    if os.name != "nt":
        raise RuntimeError("本机凭据存储需要Windows DPAPI，不允许退回明文保存。")
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    target = Blob()
    if decrypt:
        ok = crypt.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
    else:
        ok = crypt.CryptProtectData(ctypes.byref(source), "EEG研究空间", None, None, None, 1, ctypes.byref(target))
    if not ok:
        raise RuntimeError("本机凭据加密或读取失败，请重新填写密钥。")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))


def protect(value):
    return base64.b64encode(_dpapi(value.encode("utf-8"))).decode("ascii")


def reveal(value):
    return _dpapi(base64.b64decode(value), decrypt=True).decode("utf-8")
