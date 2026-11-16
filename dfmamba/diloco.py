import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from .compress import Codec
from .hub import HubClient, parse_hostport


def world():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class _Sync:
    def __init__(self, params, comm_dtype, weight, transport="gloo", hub=None, rank=0, world_size=1,
                 hub_timeout=1800.0, hub_token="", compress="none", topk_frac=0.1, error_feedback=True,
                 stochastic=False):
        if transport not in ("gloo", "hub"):
            raise ValueError(f"transport must be 'gloo' or 'hub', got {transport!r}")
        self.codec = None if compress == "none" else Codec(compress, topk_frac, stochastic)
        self.error_feedback = bool(error_feedback)
        self.residual = None
        self.params = [p for p in params if p.requires_grad]
        self.comm_dtype = comm_dtype
        self.transport = transport
        self.hub = None
        self.rank = int(rank)
        self.n = int(world_size) if transport == "hub" else world()
        if self.codec is not None and transport != "hub" and self.n > 1:
            raise ValueError("compression is only available with transport 'hub'")
        self.weight = float(weight)
        with torch.no_grad():
            if transport == "hub" and self.n > 1:
                if isinstance(hub, str):
                    host, port = parse_hostport(hub)
                    hub = HubClient(host, port, self.rank, self.n, timeout=hub_timeout, token=hub_token)
                self.hub = hub
                start = torch.cat([p.detach().flatten().float().cpu() for p in self.params])
                got, _ = self.hub.allreduce(start, 1.0 if self.rank == 0 else 0.0)
                off = 0
                for p in self.params:
                    k = p.numel()
                    p.copy_(got[off:off + k].view_as(p))
                    off += k
            elif transport == "gloo" and self.n > 1:
                for p in self.params:
                    dist.broadcast(p.data, 0)
        self.steps = 0
        self.syncs = 0
        self.bytes = 0
        self.bytes_down = 0
        self.wait_seconds = 0.0
        self.comm_seconds = 0.0

    def payload_bytes(self):
        n = sum(p.numel() for p in self.params)
        if self.codec is not None:
            return self.codec.payload_nbytes(n)
        return n * torch.finfo(self.comm_dtype).bits // 8

    def wire_bytes(self):
        if self.codec is not None:
            return self.bytes + self.bytes_down if self.n > 1 else 0
        if self.transport == "hub":
            return 2 * self.bytes if self.n > 1 else 0
        return self.bytes * (2 * (self.n - 1) / self.n if self.n > 1 else 0)

    def _encode(self, flat32):
        if self.error_feedback:
            if self.residual is None:
                self.residual = torch.zeros_like(flat32)
            flat32 = flat32 + self.residual
        payload, sent = self.codec.encode(flat32)
        if self.error_feedback:
            self.residual = flat32 - sent
        return payload, sent

    def _exchange(self, payload, sent):
        numel = sent.numel()
        if self.hub is None:
            return sent.cpu(), 0.0, 0.0, len(payload), 0
        t0 = time.time()
        reply, wait = self.hub.exchange(self.codec.code, numel, payload, self.weight)
        avg = self.codec.decode(reply, numel)
        elapsed = time.time() - t0
        return avg, min(wait, elapsed), elapsed, len(payload), len(reply)

    def _account(self, wait, elapsed, up, down):
        self.wait_seconds += wait
        self.comm_seconds += elapsed - wait
        self.bytes += up
        self.bytes_down += down

    def _average_codec(self, flat32):
        payload, sent = self._encode(flat32)
        avg, wait, elapsed, up, down = self._exchange(payload, sent)
        self._account(wait, elapsed, up, down)
        return avg.to(flat32.device)

    def _average(self, flat):
        if self.n == 1:
            self.bytes += flat.numel() * flat.element_size()
            return flat.float()
        if self.transport == "hub":
            t0 = time.time()
            avg, wait = self.hub.allreduce(flat, self.weight)
            wait = min(wait, time.time() - t0)
            self.wait_seconds += wait
            self.comm_seconds += time.time() - t0 - wait
            self.bytes += flat.numel() * flat.element_size()
            return avg.to(flat.device)
        dev_sync = torch.cuda.synchronize if flat.is_cuda else (lambda: None)
        dev_sync()
        t0 = time.time()
        dist.barrier()
        t1 = time.time()
        w = torch.tensor([self.weight], dtype=torch.float32, device=flat.device)
        dist.all_reduce(w)
        flat = (flat.float() * self.weight).to(self.comm_dtype)
        dist.all_reduce(flat)
        dev_sync()
        self.wait_seconds += t1 - t0
        self.comm_seconds += time.time() - t1
        self.bytes += flat.numel() * flat.element_size()
        return flat.float() / w

    def close(self):
        if self.hub is not None:
            self.hub.close()
            self.hub = None

    def _scatter(self, flat, targets):
        off = 0
        for t in targets:
            k = t.numel()
            yield t, flat[off:off + k].view_as(t)
            off += k


class DiLoCo(_Sync):
    def __init__(self, params, inner_steps=200, outer_lr=0.7, outer_momentum=0.9, comm_dtype=torch.bfloat16, weight=1,
                 overlap=False, overlap_delay=None, overlap_alpha=0.5, **net):
        super().__init__(params, comm_dtype, weight, **net)
        self.H = inner_steps
        self.master = [p.detach().clone().float() for p in self.params]
        self.outer = torch.optim.SGD(self.master, lr=outer_lr, momentum=outer_momentum, nesterov=outer_momentum > 0)
        self.overlap = bool(overlap)
        self.tau = int(overlap_delay) if overlap_delay else self.H
        self.alpha = float(overlap_alpha)
        self.future = None
        self.arrive = 0
        self.blocked_seconds = 0.0
        self.hidden_seconds = 0.0
        self.pool = None
        if self.overlap:
            if not 1 <= self.tau <= self.H:
                raise ValueError("overlap_delay must be between 1 and the number of inner steps")
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("overlap_alpha must be between 0 and 1")
            if self.transport != "hub" and self.n > 1:
                raise ValueError("overlap needs transport 'hub'")
            self.pool = ThreadPoolExecutor(max_workers=1)

    def step(self):
        self.steps += 1
        if self.overlap:
            return self._step_overlap()
        if self.steps % self.H:
            return False
        self.sync()
        return True

    def pending(self):
        if self.overlap:
            return self.future is not None or self.steps % self.H != 0
        return self.steps % self.H != 0

    def _delta(self):
        return torch.cat([(m - p.float()).flatten() for m, p in zip(self.master, self.params)])

    def _outer_step(self, flat):
        for m, g in self._scatter(flat, self.master):
            m.grad = g.clone()
        self.outer.step()
        self.outer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def sync(self):
        if self.overlap:
            return self._drain()
        flat = self._delta()
        if self.codec is not None:
            flat = self._average_codec(flat)
        else:
            flat = self._average(flat.to(self.comm_dtype))
        self._outer_step(flat)
        for m, p in zip(self.master, self.params):
            p.copy_(m)
        self.syncs += 1

    @torch.no_grad()
    def _launch(self):
        flat = self._delta()
        if self.codec is not None:
            payload, sent = self._encode(flat)
            job = lambda: self._exchange(payload, sent)
        else:
            cpu = flat.to(self.comm_dtype).cpu()
            weight = self.weight

            def job():
                size = cpu.numel() * cpu.element_size()
                if self.hub is None:
                    return cpu.float(), 0.0, 0.0, size, 0
                t0 = time.time()
                avg, wait = self.hub.allreduce(cpu, weight)
                elapsed = time.time() - t0
                return avg, min(wait, elapsed), elapsed, size, size

        self.future = self.pool.submit(job)
        self.arrive = self.steps + self.tau

    @torch.no_grad()
    def _apply(self, alpha):
        t0 = time.time()
        avg, wait, elapsed, up, down = self.future.result()
        blocked = time.time() - t0
        self.future = None
        self.blocked_seconds += blocked
        self.hidden_seconds += max(elapsed - blocked, 0.0)
        self._account(wait, elapsed, up, down)
        self._outer_step(avg.to(self.master[0].device))
        for m, p in zip(self.master, self.params):
            p.copy_(alpha * p + (1.0 - alpha) * m)
        self.syncs += 1

    @torch.no_grad()
    def _step_overlap(self):
        applied = False
        if self.future is not None and self.steps >= self.arrive:
            self._apply(self.alpha)
            applied = True
        if self.steps % self.H == 0:
            self._launch()
        return applied

    @torch.no_grad()
    def _drain(self):
        if self.future is not None:
            self._apply(self.alpha)
        if self.steps % self.H:
            self._launch()
            self._apply(0.0)
        else:
            for m, p in zip(self.master, self.params):
                p.copy_(m)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        super().close()


class AllReduceEveryStep(_Sync):
    def __init__(self, params, comm_dtype=torch.bfloat16, weight=1, **net):
        super().__init__(params, comm_dtype, weight, **net)

    @torch.no_grad()
    def before_step(self):
        flat = torch.cat([p.grad.flatten() for p in self.params])
        if self.codec is not None:
            flat = self._average_codec(flat.float())
        else:
            flat = self._average(flat.to(self.comm_dtype))
        for p, g in self._scatter(flat, self.params):
            p.grad.copy_(g)
        self.syncs += 1

    def step(self):
        self.steps += 1
        return True

    def pending(self):
        return False
