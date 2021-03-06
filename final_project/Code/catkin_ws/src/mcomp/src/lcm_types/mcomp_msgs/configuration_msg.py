"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class configuration_msg(object):
    __slots__ = ["timestamp", "configuration"]

    def __init__(self):
        self.timestamp = 0
        self.configuration = ""

    def encode(self):
        buf = BytesIO()
        buf.write(configuration_msg._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">q", self.timestamp))
        __configuration_encoded = self.configuration.encode('utf-8')
        buf.write(struct.pack('>I', len(__configuration_encoded)+1))
        buf.write(__configuration_encoded)
        buf.write(b"\0")

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != configuration_msg._get_packed_fingerprint():
            raise ValueError("Decode error")
        return configuration_msg._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = configuration_msg()
        self.timestamp = struct.unpack(">q", buf.read(8))[0]
        __configuration_len = struct.unpack('>I', buf.read(4))[0]
        self.configuration = buf.read(__configuration_len)[:-1].decode('utf-8', 'replace')
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if configuration_msg in parents: return 0
        tmphash = (0x362f6d313196ad3c) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if configuration_msg._packed_fingerprint is None:
            configuration_msg._packed_fingerprint = struct.pack(">Q", configuration_msg._get_hash_recursive([]))
        return configuration_msg._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

