import argparse
import hashlib
import hmac
import math
import os
import socket
import struct
import sys
import threading
import time

import torch

from .compress import Codec, is_wire_code

MAGIC = b"DFH1"
HEADER = struct.Struct("!4sBBIQddQQ")
HELLO, ROUND, RESULT, ERROR, BYE = 1, 2, 3, 4, 5
DTYPES = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16}
CODES = {v: k for k, v in DTYPES.items()}
MAX_PAYLOAD = 1 << 31
MAX_ERROR_TEXT = 512


def parse_hostport(text):
    host, sep, port = str(text).rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"hub address must look like HOST:PORT, got {text!r}")
    return host, int(port)


def _digest(token):
    return hashlib.sha256(("dfmamba-hub:" + token).encode()).digest()


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed by peer")
        got += k
    return buf


def _send(sock, kind, dtype=0, rank=0, rnd=0, weight=0.0, wait=0.0, numel=0, payload=b""):
    sock.sendall(HEADER.pack(MAGIC, kind, dtype, rank, rnd, weight, wait, numel, len(payload)))
    if len(payload):
        sock.sendall(payload)


def _recv_header(sock):
    magic, kind, dtype, rank, rnd, weight, wait, numel, nbytes = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ValueError("not a dfmamba hub message (bad magic)")
    if nbytes > MAX_PAYLOAD:
        raise ValueError(f"payload of {nbytes} bytes exceeds the {MAX_PAYLOAD} byte limit")
    return kind, dtype, rank, rnd, weight, wait, numel, nbytes


def _to_bytes(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _from_bytes(buf, dtype):
    return torch.frombuffer(buf, dtype=dtype)


class Hub:
    def __init__(self, port, world, host="0.0.0.0", timeout=1800.0, token="", sim_latency_ms=0.0, sim_mbps=0.0):
        self.world = int(world)
        self.sim_latency = float(sim_latency_ms) / 1000.0
        self.sim_mbps = float(sim_mbps)
        self.down_residual = None
        self.timeout = float(timeout)
        self.digest = _digest(token)
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(self.world + 8)
        self.port = self.srv.getsockname()[1]
        self.cond = threading.Condition()
        self.conns = {}
        self.error = None
        self.slots = {}
        self.generation = 0
        self.meta = None
        self.reply = None
        self.waits = {}
        self.finished = set()
        self.rounds = 0

    def _abort(self, msg):
        with self.cond:
            if self.error is None:
                self.error = msg
            self.cond.notify_all()
            socks = list(self.conns.values())
        text = self.error.encode()[:MAX_ERROR_TEXT]
        for s in socks:
            try:
                _send(s, ERROR, payload=text)
            except OSError:
                pass
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _handshake(self, sock):
        sock.settimeout(min(self.timeout, 30.0))
        kind, dtype, rank, world, weight, wait, numel, nbytes = _recv_header(sock)
        try:
            if kind != HELLO or nbytes != 32:
                raise ValueError("expected a hello message")
            digest = bytes(_recv_exact(sock, 32))
            if not hmac.compare_digest(digest, self.digest):
                raise ValueError("wrong hub token")
            if world != self.world:
                raise ValueError(f"worker believes world size is {world}, hub was started with {self.world}")
            if not 0 <= rank < self.world:
                raise ValueError(f"rank {rank} is outside 0..{self.world - 1}")
            if rank in self.conns:
                raise ValueError(f"rank {rank} is already connected")
        except ValueError as e:
            try:
                _send(sock, ERROR, payload=str(e).encode()[:MAX_ERROR_TEXT])
            except OSError:
                pass
            raise
        return rank

    def _rendezvous(self):
        deadline = time.monotonic() + self.timeout
        while len(self.conns) < self.world:
            left = deadline - time.monotonic()
            if left <= 0:
                missing = sorted(set(range(self.world)) - set(self.conns))
                raise TimeoutError(f"rendezvous timed out after {self.timeout:.0f}s: ranks {missing} never connected "
                                   f"({len(self.conns)} of {self.world} arrived)")
            self.srv.settimeout(left)
            try:
                sock, addr = self.srv.accept()
            except socket.timeout:
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                rank = self._handshake(sock)
            except (ValueError, OSError) as e:
                print(f"[hub] rejected {addr[0]}:{addr[1]}: {e}", flush=True)
                sock.close()
                continue
            self.conns[rank] = sock
            print(f"[hub] rank {rank} connected from {addr[0]} ({len(self.conns)}/{self.world})", flush=True)
        for s in self.conns.values():
            _send(s, HELLO, numel=self.world)

    def _throttle(self, nbytes):
        delay = self.sim_latency
        if self.sim_mbps > 0:
            delay += nbytes * 8 / (self.sim_mbps * 1e6)
        if delay > 0:
            time.sleep(delay)

    def _combine(self):
        ws = [self.slots[r][1] for r in range(self.world)]
        total = sum(ws)
        if not total > 0:
            raise ValueError("all worker weights are zero")
        dtype = self.meta[2]
        acc = torch.zeros(self.meta[1], dtype=torch.float32)
        for r in range(self.world):
            acc += self.slots[r][0].float() * self.slots[r][1]
        acc /= total
        last = max(v[2] for v in self.slots.values())
        self.waits = {r: last - self.slots[r][2] for r in range(self.world)}
        if is_wire_code(dtype):
            codec = Codec.from_wire(dtype, self.meta[1], self.first_nbytes)
            if self.down_residual is None or self.down_residual.numel() != acc.numel():
                self.down_residual = torch.zeros_like(acc)
            want = acc + self.down_residual
            payload, dec = codec.encode(want)
            self.down_residual = want - dec
            self.reply = payload
        else:
            self.reply = _to_bytes(acc.to(dtype))

    def _round(self, rank, sock, rnd, dtype_code, weight, numel, nbytes):
        codec = None
        if dtype_code in DTYPES:
            dtype = DTYPES[dtype_code]
            if nbytes != numel * torch.empty(0, dtype=dtype).element_size():
                raise ValueError("payload size does not match numel and dtype")
        elif is_wire_code(dtype_code):
            dtype = dtype_code
            codec = Codec.from_wire(dtype_code, numel, nbytes)
        else:
            raise ValueError(f"unknown dtype code {dtype_code}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"bad weight {weight}")
        raw = _recv_exact(sock, nbytes)
        tensor = codec.decode(raw, numel) if codec else _from_bytes(raw, dtype)
        self._throttle(nbytes)
        arrived = time.monotonic()
        with self.cond:
            if self.error:
                raise RuntimeError(self.error)
            if not self.slots:
                self.meta = (rnd, numel, dtype)
                self.first_nbytes = nbytes
            elif self.meta != (rnd, numel, dtype) or self.first_nbytes != nbytes:
                raise ValueError(f"rank {rank} sent round {rnd}, {numel} values, {dtype}; others sent {self.meta}")
            if rank in self.slots:
                raise ValueError(f"rank {rank} sent round {rnd} twice")
            self.slots[rank] = (tensor, weight, arrived)
            gen = self.generation
            if len(self.slots) == self.world:
                self._combine()
                self.slots = {}
                self.generation += 1
                self.rounds += 1
                self.cond.notify_all()
            else:
                deadline = time.monotonic() + self.timeout
                while self.generation == gen and not self.error:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        missing = sorted(set(range(self.world)) - set(self.slots))
                        raise TimeoutError(f"round {rnd} timed out after {self.timeout:.0f}s waiting for ranks {missing}")
                    self.cond.wait(left)
            if self.error:
                raise RuntimeError(self.error)
            reply, wait = self.reply, self.waits[rank]
        self._throttle(len(reply))
        _send(sock, RESULT, dtype_code, rank, rnd, 0.0, wait, numel, reply)

    def _serve_conn(self, rank, sock):
        sock.settimeout(self.timeout)
        try:
            while True:
                try:
                    kind, dtype, r, rnd, weight, wait, numel, nbytes = _recv_header(sock)
                except socket.timeout:
                    raise TimeoutError(f"rank {rank} sent nothing for {self.timeout:.0f}s")
                if kind == BYE:
                    with self.cond:
                        self.finished.add(rank)
                    return
                if kind != ROUND or r != rank:
                    raise ValueError(f"unexpected message kind {kind} from rank {rank}")
                self._round(rank, sock, rnd, dtype, weight, numel, nbytes)
        except Exception as e:
            if isinstance(e, (ConnectionError, socket.timeout, OSError)) and self.error is None:
                self._abort(f"lost rank {rank}: {e}")
            elif self.error is None:
                self._abort(f"rank {rank}: {e}")

    def serve(self):
        try:
            self._rendezvous()
        except Exception as e:
            self._abort(f"rendezvous failed: {e}")
            self.close()
            raise
        print(f"[hub] all {self.world} workers connected, training can start", flush=True)
        threads = [threading.Thread(target=self._serve_conn, args=(r, s), daemon=True) for r, s in self.conns.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        err = self.error
        self.close()
        if err:
            raise RuntimeError(err)
        print(f"[hub] done after {self.rounds} rounds", flush=True)

    def close(self):
        for s in list(self.conns.values()) + [self.srv]:
            try:
                s.close()
            except OSError:
                pass


class HubClient:
    def __init__(self, host, port, rank, world, timeout=1800.0, connect_timeout=120.0, token=""):
        self.rank = int(rank)
        self.world = int(world)
        self.timeout = float(timeout)
        self.address = f"{host}:{port}"
        self.round = 0
        deadline = time.monotonic() + connect_timeout
        last = None
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=10)
                break
            except OSError as e:
                last = e
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"could not reach the hub at {self.address} within {connect_timeout:.0f}s "
                                          f"({last}). Is the hub running, and is this the externally mapped port?")
                time.sleep(1.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout)
        try:
            _send(self.sock, HELLO, rank=self.rank, rnd=self.world, payload=_digest(token))
            kind, _, _, _, _, _, _, nbytes = _recv_header(self.sock)
            self._check(kind, nbytes)
        except socket.timeout:
            raise TimeoutError(f"the other workers did not all connect to the hub within {self.timeout:.0f}s")

    def _check(self, kind, nbytes):
        if kind == ERROR:
            text = bytes(_recv_exact(self.sock, min(nbytes, MAX_ERROR_TEXT))).decode(errors="replace")
            raise RuntimeError(f"hub {self.address} reported an error: {text}")
        if kind not in (HELLO, RESULT):
            raise ValueError(f"unexpected message kind {kind} from the hub")

    def exchange(self, code, numel, payload, weight):
        self.round += 1
        try:
            _send(self.sock, ROUND, code, self.rank, self.round, float(weight), 0.0, numel, payload)
            kind, dtype, _, rnd, _, wait, rnumel, nbytes = _recv_header(self.sock)
            self._check(kind, nbytes)
            if rnd != self.round or rnumel != numel or dtype != code:
                raise ValueError("hub reply does not match the request")
            out = _recv_exact(self.sock, nbytes)
        except socket.timeout:
            raise TimeoutError(f"no reply from the hub {self.address} for {self.timeout:.0f}s in round {self.round}; "
                               f"another worker is probably stuck or gone")
        except (ConnectionError, OSError) as e:
            raise ConnectionError(f"lost the connection to the hub {self.address} in round {self.round}: {e}")
        return out, wait

    def allreduce(self, flat, weight):
        flat = flat.detach().cpu().contiguous()
        self.round += 1
        code = CODES[flat.dtype]
        try:
            _send(self.sock, ROUND, code, self.rank, self.round, float(weight), 0.0, flat.numel(), _to_bytes(flat))
            kind, dtype, _, rnd, _, wait, numel, nbytes = _recv_header(self.sock)
            self._check(kind, nbytes)
            if rnd != self.round or numel != flat.numel() or dtype != code:
                raise ValueError("hub reply does not match the request")
            out = _from_bytes(_recv_exact(self.sock, nbytes), flat.dtype)
        except socket.timeout:
            raise TimeoutError(f"no reply from the hub {self.address} for {self.timeout:.0f}s in round {self.round}; "
                               f"another worker is probably stuck or gone")
        except (ConnectionError, OSError) as e:
            raise ConnectionError(f"lost the connection to the hub {self.address} in round {self.round}: {e}")
        return out.float(), wait

    def close(self):
        try:
            _send(self.sock, BYE, rank=self.rank)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Star-topology all-reduce hub: workers connect out to ONE TCP port")
    ap.add_argument("--port", type=int, default=29600)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--token", default=os.environ.get("DFM_HUB_TOKEN", ""))
    ap.add_argument("--sim-latency-ms", type=float, default=0.0,
                    help="test only: delay every message by this many ms in each direction (RTT = 2x)")
    ap.add_argument("--sim-mbps", type=float, default=0.0,
                    help="test only: cap each worker link at this many megabits per second, each direction (0 = off)")
    a = ap.parse_args()
    hub = Hub(a.port, a.world, a.host, a.timeout, a.token, a.sim_latency_ms, a.sim_mbps)
    print(f"[hub] listening on {a.host}:{hub.port}, waiting for {a.world} workers", flush=True)
    try:
        hub.serve()
    except Exception as e:
        print(f"[hub] stopped: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
