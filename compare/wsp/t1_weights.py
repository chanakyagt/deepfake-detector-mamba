import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session, weight_stats

S = Session("mamba_diloco")
print("ckpt", S.t.ckpt, "params", S.total_params)
other = torch.load(os.path.join(CK, "img_runs", "single_H1/model.pt"), map_location="cpu", weights_only=False)["model"]
init_alog = torch.log(torch.arange(1, 17, dtype=torch.float32)).repeat(1024, 1)
for name, shape, n, isp in S.names():
    t = S.state(name).float()
    o = other[name].float()
    cos = float(torch.nn.functional.cosine_similarity(t.flatten(), o.flatten(), 0))
    ws = weight_stats(t)
    print(f"{name:32s} {str(shape):14s} l2={ws['l2']:.3f} std={ws['std']:.4f} maxabs={ws['max_abs']:.3f} "
          f"single_l2={o.norm():.3f} cos(diloco,single)={cos:.3f}" + (f" eff_rank={ws['eff_rank']:.1f} top_sv={[round(v,2) for v in ws['top_sv']]}" if 'eff_rank' in ws else ""))
for l in range(4):
    a = S.state(f"layers.{l}.mixer.A_log")
    print("A_log layer", l, "ratio to init log(1..16) (cols 1..15):", (a[:, 1:] / init_alog[:, 1:]).mean().item(), (a[:, 1:] / init_alog[:, 1:]).std().item(), "col0 maxabs", a[:, 0].abs().max().item())
    c = S.state(f"layers.{l}.mixer.conv1d.weight")[:, 0, :]
    print("conv taps l2 per position", [round(float(c[:, j].norm()), 3) for j in range(4)])
    dtb = S.state(f"layers.{l}.mixer.dt_proj.bias")
    print("dt bias softplus range", float(torch.nn.functional.softplus(dtb).min()), float(torch.nn.functional.softplus(dtb).median()), float(torch.nn.functional.softplus(dtb).max()))
    D = S.state(f"layers.{l}.mixer.D")
    print("D", float(D.mean()), float(D.std()), float(D.min()), float(D.max()))
hb = S.state("head.3.bias"); hw = S.state("head.3.weight")
print("head.3.bias", hb.tolist(), "logit diff weight norm", float((hw[1] - hw[0]).norm()))
print("budget used", S.budget.data["used"])
