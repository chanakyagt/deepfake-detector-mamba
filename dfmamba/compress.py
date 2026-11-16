import torch
import torch.nn.functional as F

CODES = {"int8": 3, "int4": 4, "topk": 5}
NAMES = {v: k for k, v in CODES.items()}
GROUPS = {"int8": 256, "int4": 64}
QMAX = {"int8": 127, "int4": 7}
TOPK_ENTRY_BYTES = 6
SCALE_BYTES = 2


def is_wire_code(code):
    return code in NAMES


def _bytes_of(t):
    return t.contiguous().cpu().numpy().tobytes()


def _writable(buf):
    return buf if isinstance(buf, bytearray) else bytearray(buf)


class Codec:
    def __init__(self, name, frac=0.1, stochastic=False):
        if name not in CODES:
            raise ValueError(f"unknown codec {name!r}, expected one of {sorted(CODES)}")
        if name == "topk" and not 0 < frac <= 1:
            raise ValueError(f"topk fraction must be in (0, 1], got {frac}")
        self.name = name
        self.frac = float(frac)
        self.stochastic = bool(stochastic)

    @property
    def code(self):
        return CODES[self.name]

    @classmethod
    def from_wire(cls, code, numel, nbytes):
        if code not in NAMES:
            raise ValueError(f"unknown codec code {code}")
        c = cls(NAMES[code])
        if c.name == "topk":
            if nbytes % TOPK_ENTRY_BYTES:
                raise ValueError("topk payload size is not a multiple of 6 bytes")
            c.frac = max(nbytes // TOPK_ENTRY_BYTES, 1) / max(numel, 1)
        c.validate(numel, nbytes)
        return c

    def count(self, n):
        return min(n, max(1, int(round(self.frac * n))))

    def payload_nbytes(self, n):
        if self.name == "topk":
            return self.count(n) * TOPK_ENTRY_BYTES
        g = GROUPS[self.name]
        ng = -(-n // g)
        return ng * SCALE_BYTES + (ng * g if self.name == "int8" else ng * g // 2)

    def validate(self, n, nbytes):
        if self.name == "topk":
            if nbytes % TOPK_ENTRY_BYTES or not 0 < nbytes // TOPK_ENTRY_BYTES <= max(n, 1):
                raise ValueError("bad topk payload size")
        elif nbytes != self.payload_nbytes(n):
            raise ValueError(f"{self.name} payload of {nbytes} bytes does not match {n} values")

    def encode(self, x):
        x = x.detach().float().flatten().contiguous()
        if self.name == "topk":
            return self._encode_topk(x)
        return self._encode_int(x)

    def _encode_int(self, x):
        g, q, n = GROUPS[self.name], QMAX[self.name], x.numel()
        xp = F.pad(x, (0, (-n) % g)).view(-1, g)
        scale = (xp.abs().amax(1) / q).to(torch.bfloat16)
        s = scale.float()
        div = torch.where(s > 0, s, torch.ones_like(s))[:, None]
        y = xp / div
        y = torch.floor(y + torch.rand_like(y)) if self.stochastic else torch.round(y)
        y = y.clamp(-q, q)
        dec = (y * s[:, None]).flatten()[:n]
        qi = y.to(torch.int8).flatten()
        if self.name == "int4":
            u = (qi + 8).to(torch.uint8)
            qi = u[0::2] | (u[1::2] << 4)
        return _bytes_of(scale.view(torch.int16)) + _bytes_of(qi), dec

    def _encode_topk(self, x):
        n = x.numel()
        k = self.count(n)
        idx = torch.topk(x.abs(), k, sorted=False).indices.sort().values
        vals = x[idx].to(torch.bfloat16)
        dec = torch.zeros_like(x)
        dec[idx] = vals.float()
        return _bytes_of(idx.to(torch.int32)) + _bytes_of(vals.view(torch.int16)), dec

    def decode(self, buf, n, device="cpu"):
        self.validate(n, len(buf))
        buf = _writable(buf)
        if self.name == "topk":
            k = len(buf) // TOPK_ENTRY_BYTES
            idx = torch.frombuffer(buf, dtype=torch.int32, count=k).long()
            vals = torch.frombuffer(buf, dtype=torch.int16, offset=4 * k, count=k).view(torch.bfloat16).float()
            out = torch.zeros(n)
            out[idx] = vals
            return out.to(device)
        g = GROUPS[self.name]
        ng = -(-n // g)
        s = torch.frombuffer(buf, dtype=torch.int16, count=ng).view(torch.bfloat16).float()
        off = ng * SCALE_BYTES
        if self.name == "int8":
            qv = torch.frombuffer(buf, dtype=torch.int8, offset=off, count=ng * g).float()
        else:
            u = torch.frombuffer(buf, dtype=torch.uint8, offset=off, count=ng * g // 2)
            qv = torch.stack([u & 15, u >> 4], 1).flatten().float() - 8
        return (qv.view(ng, g) * s[:, None]).flatten()[:n].to(device)
