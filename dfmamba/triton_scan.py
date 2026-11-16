import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def available():
    return HAVE_TRITON and torch.cuda.is_available()


if HAVE_TRITON:
    @triton.jit
    def _scan_fwd(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, y_ptr, L, E,
                  sxb, sxl, sbb, sbl, N: tl.constexpr, BE: tl.constexpr):
        b = tl.program_id(0)
        e = tl.program_id(1) * BE + tl.arange(0, BE)
        n = tl.arange(0, N)
        me = e < E
        A = tl.load(A_ptr + e[:, None] * N + n[None, :], mask=me[:, None], other=0.0)
        Dv = tl.load(D_ptr + e, mask=me, other=0.0)
        h = tl.zeros((BE, N), dtype=tl.float32)
        for t in range(L):
            off = b * sxb + t * sxl + e
            xv = tl.load(x_ptr + off, mask=me, other=0.0)
            dv = tl.load(dt_ptr + off, mask=me, other=0.0)
            Bv = tl.load(B_ptr + b * sbb + t * sbl + n)
            Cv = tl.load(C_ptr + b * sbb + t * sbl + n)
            h = tl.exp(dv[:, None] * A) * h + (dv * xv)[:, None] * Bv[None, :]
            tl.store(y_ptr + off, tl.sum(h * Cv[None, :], axis=1) + Dv * xv, mask=me)


def scan_fused(x, delta, A, Bm, Cm, D, block_e=64):
    x, delta, Bm, Cm = (t.float().contiguous() for t in (x, delta, Bm, Cm))
    A, D = A.float().contiguous(), D.float().contiguous()
    b, L, E = x.shape
    y = torch.empty_like(x)
    _scan_fwd[(b, triton.cdiv(E, block_e))](x, delta, A, Bm, Cm, D, y, L, E,
                                            x.stride(0), x.stride(1), Bm.stride(0), Bm.stride(1),
                                            N=A.shape[1], BE=block_e, num_warps=4)
    return y
