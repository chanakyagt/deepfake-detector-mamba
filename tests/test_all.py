import os
import sys
import tempfile
import threading

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfmamba.compress import GROUPS, QMAX, Codec
from dfmamba.data import FeatureCache
from dfmamba.diloco import AllReduceEveryStep, DiLoCo
from dfmamba.hub import Hub, HubClient
from dfmamba.metrics import auc
from dfmamba.model import TemporalClassifier
from dfmamba.ssm import SelectiveSSM, scan_reference


def unrolled(x, delta, A, Bm, Cm, D):
    b, L, E = x.shape
    y = torch.zeros_like(x)
    for t in range(L):
        h = 0
        for s in range(t + 1):
            decay = torch.exp(delta[:, s + 1:t + 1, :, None] * A).prod(1) if s < t else 1
            h = h + decay * delta[:, s, :, None] * Bm[:, s, None, :] * x[:, s, :, None]
        y[:, t] = (h * Cm[:, t, None, :]).sum(-1)
    return y + x * D


def test_scan_equals_unrolled_sum():
    torch.manual_seed(0)
    m = SelectiveSSM(8, 4).double()
    u = torch.randn(2, 7, 16, dtype=torch.float64)
    args = m.ssm_inputs(u)
    assert torch.allclose(scan_reference(*args), unrolled(*args), atol=1e-10)
    print("ok  scan recurrence == unrolled sum")


def test_causal():
    torch.manual_seed(0)
    m = TemporalClassifier(d_model=32, n_layers=2).eval()
    layer = m.layers[0]
    x = torch.randn(1, 12, 32)
    x2 = x.clone()
    x2[:, 8:] = torch.randn(1, 4, 32)
    with torch.no_grad():
        a, b = layer(x), layer(x2)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6) and not torch.allclose(a[:, 8:], b[:, 8:])
    print("ok  output at frame t does not depend on later frames")


def test_triton_matches_reference():
    if not torch.cuda.is_available():
        print("skip triton test (no GPU)")
        return
    from dfmamba.triton_scan import available, scan_fused
    if not available():
        print("skip triton test (no triton)")
        return
    torch.manual_seed(0)
    for b, L, d in ((1, 30, 512), (3, 7, 40), (2, 300, 256)):
        m = SelectiveSSM(d).cuda()
        u = torch.randn(b, L, 2 * d, device="cuda")
        with torch.no_grad():
            args = m.ssm_inputs(u)
            ref, fused = scan_reference(*args), scan_fused(*args)
        assert torch.allclose(ref, fused, rtol=1e-4, atol=1e-4), (ref - fused).abs().max()
    m.eval()
    with torch.no_grad():
        assert m.use_triton(u)
    m.train()
    assert not m.use_triton(u)
    print("ok  fused Triton scan == reference; used only for inference")


def test_gradients_flow():
    torch.manual_seed(0)
    m = TemporalClassifier(d_model=32, n_layers=2)
    loss = m(torch.randn(4, 10, 1280)).logsumexp(1).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print("ok  every parameter gets a finite gradient")


def test_auc():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 400), rng.random(400).round(2)
    pos, neg = p[y == 1], p[y == 0]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert abs(auc(y, p) - brute) < 1e-12
    print("ok  AUC with ties == brute force")


def test_sharding():
    with tempfile.TemporaryDirectory() as d:
        torch.save({"feats": torch.arange(10.0)[:, None, None].half(), "labels": torch.arange(10)}, d + "/x.pt")
        parts = [FeatureCache(d + "/x.pt", r, 3).labels.tolist() for r in range(3)]
    assert sorted(sum(parts, [])) == list(range(10)) and not set(parts[0]) & set(parts[1])
    print("ok  workers get disjoint shards that cover the data")


def test_diloco_h1_equals_plain():
    torch.manual_seed(0)
    X, Y = torch.randn(400, 8), torch.randn(400, 1)
    a = torch.nn.Linear(8, 1)
    b = torch.nn.Linear(8, 1)
    b.load_state_dict(a.state_dict())
    oa, ob = torch.optim.AdamW(a.parameters(), 1e-2), torch.optim.AdamW(b.parameters(), 1e-2)
    s = DiLoCo(b.parameters(), inner_steps=1, outer_lr=1.0, outer_momentum=0.0, comm_dtype=torch.float32)
    for i in range(40):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            torch.nn.functional.mse_loss(m(X[i * 10:i * 10 + 10]), Y[i * 10:i * 10 + 10]).backward()
            o.step()
        s.step()
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))
    print("ok  DiLoCo with H=1, outer lr 1, no momentum == plain AdamW")


def _worker(rank, n, f, q, mode):
    dist.init_process_group("gloo", init_method=f"file://{f}", rank=rank, world_size=n)
    g = torch.Generator().manual_seed(1)
    w = torch.randn(8, 1, generator=g)
    X = torch.randn(800, 8, generator=g)
    Y = X @ w
    X, Y = X[rank::n], Y[rank::n]
    torch.manual_seed(10 + rank)
    m = torch.nn.Linear(8, 1)
    o = torch.optim.AdamW(m.parameters(), 1e-2)
    s = DiLoCo(m.parameters(), 25, 0.7, 0.9) if mode == "diloco" else AllReduceEveryStep(m.parameters(), torch.float32)
    first = None
    for i in range(200):
        j = (i * 10) % len(X)
        loss = torch.nn.functional.mse_loss(m(X[j:j + 10]), Y[j:j + 10])
        first = first or loss.item()
        o.zero_grad()
        loss.backward()
        if mode != "diloco":
            s.before_step()
        o.step()
        if s.step():
            v = torch.cat([p.detach().flatten() for p in m.parameters()])
            vs = [torch.zeros_like(v) for _ in range(n)]
            dist.all_gather(vs, v)
            assert all(torch.equal(vs[0], t) for t in vs)
    q.put((rank, first, torch.nn.functional.mse_loss(m(X), Y).item(), s.syncs, s.bytes))
    dist.destroy_process_group()


def test_multi_worker_sync():
    for mode, syncs in (("diloco", 8), ("allreduce", 200)):
        with tempfile.TemporaryDirectory() as d:
            q = mp.get_context("spawn").SimpleQueue()
            mp.spawn(_worker, args=(3, (d + "/init").replace("\\", "/"), q, mode), nprocs=3)
            res = [q.get() for _ in range(3)]
        for rank, first, final, s, by in res:
            assert s == syncs and final < 0.15 * first, (mode, first, final)
        print(f"ok  3 workers, {mode}: identical weights after every sync, loss {res[0][1]:.2f} -> {res[0][2]:.4f}, "
              f"{syncs} syncs, {res[0][4]} bytes sent per worker")


def _start_hub(n, **kw):
    hub = Hub(0, n, host="127.0.0.1", timeout=kw.pop("timeout", 30), **kw)
    err = []

    def run():
        try:
            hub.serve()
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return hub, t, err


def _run_threads(fns):
    out, errs = [None] * len(fns), []

    def wrap(i, f):
        try:
            out[i] = f()
        except Exception as e:
            errs.append(e)

    ts = [threading.Thread(target=wrap, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errs, errs
    return out


def test_hub_weighted_average_matches_manual():
    n = 3
    g = torch.Generator().manual_seed(5)
    xs = [torch.randn(5000, generator=g).to(torch.bfloat16) for _ in range(n)]
    ws = [32.0, 16.0, 8.0]
    hub, t, err = _start_hub(n)

    def worker(r):
        c = HubClient("127.0.0.1", hub.port, r, n, timeout=30)
        res = [c.allreduce(xs[r], ws[r]) for _ in range(2)]
        c.close()
        return res

    out = _run_threads([lambda r=r: worker(r) for r in range(n)])
    t.join(10)
    assert not err
    manual = sum(x.float() * w for x, w in zip(xs, ws)) / sum(ws)
    for res in out:
        for avg, wait in res:
            assert torch.allclose(avg, manual, rtol=1e-2, atol=1e-2) and wait >= 0
        assert torch.equal(res[0][0], out[0][0][0])
    print("ok  hub weighted average == manual weighted mean (bf16 tolerance), identical on all 3 workers")


def test_hub_rejects_bad_rank_and_times_out():
    hub, t, err = _start_hub(2, timeout=2)
    try:
        HubClient("127.0.0.1", hub.port, 7, 2, timeout=5)
        raise AssertionError("bad rank accepted")
    except RuntimeError as e:
        assert "outside" in str(e)
    t.join(10)
    assert err and "never connected" in str(err[0]) and "[0, 1]" in str(err[0])
    print("ok  hub rejects an invalid rank and reports which ranks never connected")


def _hub_run(rank, n, hub, mode, A, steps=60, H=20):
    torch.manual_seed(100 + rank)
    m = torch.nn.Linear(8, 1)
    w = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))
    X = torch.randn(600, 8, generator=torch.Generator().manual_seed(2))
    Y = X @ w
    X, Y = X[rank::n], Y[rank::n]
    o = torch.optim.AdamW(m.parameters(), 1e-2)
    net = dict(transport="hub", hub=f"127.0.0.1:{hub.port}", rank=rank, world_size=n, hub_timeout=30)
    s = DiLoCo(m.parameters(), H, 0.7, 0.9, weight=16 + 8 * rank, **net) if mode == "diloco"         else AllReduceEveryStep(m.parameters(), weight=16 + 8 * rank, **net)
    snaps = []
    for i in range(steps):
        j = (i * 10) % len(X)
        o.zero_grad()
        torch.nn.functional.mse_loss(m(X[j:j + 10]), Y[j:j + 10]).backward()
        if mode != "diloco":
            s.before_step()
        o.step()
        if s.step():
            snaps.append(torch.cat([p.detach().flatten() for p in m.parameters()]).clone())
    s.close()
    return snaps, s


def test_hub_diloco_identical_params():
    n = 3
    for mode, syncs in (("diloco", 3), ("allreduce", 60)):
        hub, t, err = _start_hub(n)
        out = _run_threads([lambda r=r: _hub_run(r, n, hub, mode, None) for r in range(n)])
        t.join(10)
        assert not err
        for k in range(syncs):
            assert all(torch.equal(out[0][0][k], o[0][k]) for o in out)
        s = out[0][1]
        assert s.syncs == syncs and s.wire_bytes() == 2 * s.bytes and s.bytes == syncs * 9 * 2
        assert s.wait_seconds >= 0 and s.comm_seconds > 0
        print(f"ok  hub, 3 workers, {mode}: identical parameters after each of {syncs} syncs, "
              f"{s.bytes} B payload, {s.wire_bytes():.0f} B on the wire per worker")


def test_hub_matches_gloo_run():
    n = 3
    with tempfile.TemporaryDirectory() as d:
        q = mp.get_context("spawn").SimpleQueue()
        mp.spawn(_worker, args=(3, (d + "/init").replace("\\", "/"), q, "diloco"), nprocs=3)
        gl = sorted(q.get() for _ in range(3))
    hub, t, err = _start_hub(n)

    out = _run_threads([lambda r=r: _hub_run(r, n, hub, "diloco", None, 200, 25) for r in range(n)])
    t.join(10)
    assert not err and out[0][1].syncs == gl[0][3] == 8
    print("ok  hub DiLoCo run does the same number of syncs as the gloo run (8)")


def test_compress_round_trip_bounds():
    torch.manual_seed(0)
    x = torch.randn(10007) * torch.logspace(-3, 1, 10007)
    for name in ("int8", "int4"):
        c = Codec(name)
        payload, dec = c.encode(x)
        assert len(payload) == c.payload_nbytes(x.numel())
        assert torch.equal(c.decode(payload, x.numel()), dec)
        g, q = GROUPS[name], QMAX[name]
        pad = (-x.numel()) % g
        step = (torch.nn.functional.pad(x, (0, pad)).view(-1, g).abs().amax(1) / q).repeat_interleave(g)[:x.numel()]
        err = (x - dec).abs()
        assert (err <= 0.52 * step + 1e-12).all(), name
        s = Codec(name, stochastic=True)
        _, ds = s.encode(x)
        assert ((x - ds).abs() <= 1.02 * step + 1e-12).all()
        mean = torch.stack([s.encode(x)[1] for _ in range(300)]).mean(0)
        assert (mean - x).abs().mean() < 0.35 * err.mean() + 1e-9, name
    for frac in (0.1, 0.01):
        c = Codec("topk", frac)
        payload, dec = c.encode(x)
        k = c.count(x.numel())
        assert len(payload) == 6 * k and int((dec != 0).sum()) == k
        assert torch.equal(c.decode(payload, x.numel()), dec)
        kept = x.abs()[dec != 0]
        assert kept.min() >= x.abs()[dec == 0].max()
        assert torch.allclose(dec[dec != 0], x[dec != 0], rtol=2 ** -7)
    print("ok  int8 / int4 / topk round trips stay inside their error bounds; payload sizes are exact")


def test_error_feedback_accumulates_residual():
    torch.manual_seed(1)
    for name, ef in (("int4", True), ("topk", True), ("int4", False)):
        m = torch.nn.Linear(40, 25)
        s = DiLoCo(m.parameters(), 5, compress=name, topk_frac=0.05, error_feedback=ef)
        n = s.payload_bytes()
        total = sum(p.numel() for p in s.params)
        true_sum = torch.zeros(total)
        sent_sum = torch.zeros(total)
        for _ in range(12):
            g = torch.randn(total) * 0.1
            true_sum += g
            sent_sum += s._average_codec(g)
        if ef:
            assert torch.allclose(sent_sum + s.residual, true_sum, atol=1e-5)
            assert s.residual.abs().max() < true_sum.abs().max()
        else:
            assert s.residual is None and not torch.allclose(sent_sum, true_sum, atol=1e-3)
        assert s.bytes == 12 * n
    print("ok  error feedback: sent + residual == true sum after every round (telescoping); without it the error stays")


def test_hub_compressed_average_matches_manual():
    n = 3
    g = torch.Generator().manual_seed(7)
    xs = [torch.randn(4097, generator=g) * 0.05 for _ in range(n)]
    ws = [32.0, 16.0, 8.0]
    for name, frac in (("int8", 0.1), ("int4", 0.1), ("topk", 0.2)):
        codecs = [Codec(name, frac) for _ in range(n)]
        sent = [c.encode(x) for c, x in zip(codecs, xs)]
        manual = sum(d * w for (_, d), w in zip(sent, ws)) / sum(ws)
        hub, t, err = _start_hub(n)

        def worker(r):
            c = HubClient("127.0.0.1", hub.port, r, n, timeout=30)
            out = [c.exchange(codecs[r].code, xs[r].numel(), sent[r][0], ws[r]) for _ in range(2)]
            c.close()
            return out

        out = _run_threads([lambda r=r: worker(r) for r in range(n)])
        t.join(10)
        assert not err
        assert all(len(o[0][0]) == len(sent[0][0]) for o in out)
        dec = [codecs[0].decode(o[0][0], xs[0].numel()) for o in out]
        assert all(torch.equal(dec[0], d) for d in dec)
        tol = {"int8": 0.01, "int4": 0.15, "topk": None}[name]
        if tol:
            assert (dec[0] - manual).abs().max() <= tol * manual.abs().max() + 1e-6, name
        else:
            assert torch.allclose(dec[0][dec[0] != 0], manual[dec[0] != 0], rtol=1e-2, atol=1e-5)
        if tol:
            second = codecs[0].decode(out[0][1][0], xs[0].numel())
            assert torch.allclose(dec[0] + second, 2 * manual, atol=2 * tol * manual.abs().max() + 1e-5)
        else:
            assert hub.down_residual is not None and hub.down_residual.abs().max() > 0
    print("ok  hub averages decoded int8 / int4 / topk payloads like a manual weighted mean; same bytes on every worker")


def _overlap_run(rank, n, hub, tau, alpha, compress, steps=80, H=20):
    torch.manual_seed(100 + rank)
    m = torch.nn.Linear(8, 1)
    w = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))
    X = torch.randn(600, 8, generator=torch.Generator().manual_seed(2))
    Y = X @ w
    X, Y = X[rank::n], Y[rank::n]
    o = torch.optim.AdamW(m.parameters(), 1e-2)
    net = dict(transport="hub", hub=f"127.0.0.1:{hub.port}", rank=rank, world_size=n, hub_timeout=30, compress=compress)
    s = DiLoCo(m.parameters(), H, 0.7, 0.9, weight=16 + 8 * rank, overlap=True, overlap_delay=tau, overlap_alpha=alpha, **net)
    masters, locals_ = [], []
    for i in range(steps):
        j = (i * 10) % len(X)
        o.zero_grad()
        torch.nn.functional.mse_loss(m(X[j:j + 10]), Y[j:j + 10]).backward()
        o.step()
        if s.step():
            masters.append(torch.cat([t.flatten() for t in s.master]).clone())
            locals_.append(torch.cat([p.detach().flatten() for p in m.parameters()]).clone())
    if s.pending():
        s.sync()
    final = torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
    s.close()
    return masters, locals_, final, s, torch.nn.functional.mse_loss(m(X), Y).item()


def test_overlap_keeps_parameters_identical_after_applied_round():
    n = 3
    for tau, alpha, compress in ((20, 0.5, "none"), (20, 0.5, "int8"), (7, 0.0, "none"), (20, 0.0, "int4")):
        hub, t, err = _start_hub(n)
        out = _run_threads([lambda r=r: _overlap_run(r, n, hub, tau, alpha, compress) for r in range(n)])
        t.join(10)
        assert not err
        k = len(out[0][0])
        assert k == 3 and out[0][3].syncs == 4
        for i in range(k):
            assert all(torch.equal(out[0][0][i], o[0][i]) for o in out)
            if alpha == 0.0:
                assert all(torch.equal(out[0][1][i], o[1][i]) for o in out)
        assert all(torch.equal(out[0][2], o[2]) for o in out)
        s = out[0][3]
        assert s.hidden_seconds >= 0 and s.blocked_seconds >= 0 and s.bytes > 0
    print("ok  overlap: master identical on all workers after each applied round; local params identical when alpha=0 and after the final drain")


def test_make_shards():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        torch.save({"feats": torch.randn(11, 2, 4).half(), "labels": torch.arange(11)}, d + "/train.pt")
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "make_shards.py"),
                               "--src", d + "/train.pt", "--world", "3"], stdout=subprocess.DEVNULL)
        for r in range(3):
            a = FeatureCache(d + f"/train_shard{r}of3.pt")
            b = FeatureCache(d + "/train.pt", r, 3)
            assert torch.equal(a.labels, b.labels) and torch.equal(a.feats, b.feats)
        assert len(FeatureCache(d + "/train.pt", 0, 3, limit=2)) == 2
    print("ok  make_shards writes exactly each rank's rows; --train-limit caps them")


if __name__ == "__main__":
    test_scan_equals_unrolled_sum()
    test_causal()
    test_triton_matches_reference()
    test_gradients_flow()
    test_auc()
    test_sharding()
    test_diloco_h1_equals_plain()
    test_multi_worker_sync()
    test_hub_weighted_average_matches_manual()
    test_hub_rejects_bad_rank_and_times_out()
    test_hub_diloco_identical_params()
    test_hub_matches_gloo_run()
    test_compress_round_trip_bounds()
    test_error_feedback_accumulates_residual()
    test_hub_compressed_average_matches_manual()
    test_overlap_keeps_parameters_identical_after_applied_round()
    test_make_shards()
