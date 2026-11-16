import torch
import torch.nn as nn
import torch.nn.functional as F


def scan_reference(x, delta, A, Bm, Cm, D):
    h = x.new_zeros(x.shape[0], x.shape[2], A.shape[1])
    ys = []
    for t in range(x.shape[1]):
        dt = delta[:, t, :, None]
        h = torch.exp(dt * A) * h + dt * Bm[:, t, None, :] * x[:, t, :, None]
        ys.append((h * Cm[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + x * D


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        e, n = self.d_inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * e, bias=False)
        self.conv1d = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, 2 * n, bias=False)
        self.dt_proj = nn.Linear(e, e)
        dt = torch.exp(torch.rand(e) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
                       + torch.log(torch.tensor(dt_min)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d_model, bias=False)
        self.scan_impl = "auto"

    def ssm_inputs(self, u):
        acc = torch.promote_types(u.dtype, torch.float32)
        delta = F.softplus(self.dt_proj(u).to(acc))
        Bm, Cm = self.x_proj(u).to(acc).chunk(2, -1)
        A = -torch.exp(self.A_log.to(acc))
        return u.to(acc), delta, A, Bm, Cm, self.D.to(acc)

    def use_triton(self, u):
        if self.scan_impl == "ref":
            return False
        if torch.is_grad_enabled() and (u.requires_grad or any(p.requires_grad for p in self.parameters())):
            return False
        if self.d_state & (self.d_state - 1):
            return False
        if not u.is_cuda:
            return False
        try:
            from . import triton_scan
            return triton_scan.available()
        except ImportError:
            return False

    def scan(self, u):
        args = self.ssm_inputs(u)
        if self.use_triton(u):
            from .triton_scan import scan_fused
            y = scan_fused(*args)
        else:
            y = scan_reference(*args)
        return y.to(u.dtype)

    def forward(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2))
        return self.out_proj(self.scan(u) * F.silu(z))


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(d_model, d_state)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
