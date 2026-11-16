# Running DiLoCo across several PCs on one network

This guide is for 2-4 Windows PCs with NVIDIA GPUs (for example an RTX 4060 Ti desktop and RTX 3050 / 4050 laptops), each training with WSL2, connected over home Wi-Fi or Ethernet.

Training runs on cached EfficientNet features (see the README), so each worker needs well under 1 GB of GPU memory. A 4 GB laptop GPU is enough.

## 1. On every PC: WSL2, Python, this repo

In PowerShell:

```powershell
wsl --install -d Ubuntu
```

Inside Ubuntu:

```bash
sudo apt update && sudo apt install -y python3-venv git iperf3
python3 -m venv ~/dfm && source ~/dfm/bin/activate
pip install torch torchvision pyyaml numpy opencv-python-headless
```

Use the **same PyTorch version on every machine** (`python -c "import torch; print(torch.__version__)"`). Different versions can fail to talk to each other.

Copy this repository to the same path on every machine. Copy the feature folder too (`train.pt`, `val.pt`, `test.pt`, written once by `dfmamba.extract` on the machine with the strongest GPU). Each worker loads `train.pt` and keeps only its own share: rank r takes clips r, r+N, r+2N, and so on.

## 2. On every PC: let the WSL2 workers reach each other

By default, WSL2 sits behind a private NAT that other machines cannot reach. Mirrored networking gives WSL2 the PC's own LAN address.

1. Create or edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

2. Restart WSL in PowerShell:

```powershell
wsl --shutdown
```

3. Allow inbound connections to WSL from the LAN. These are firewall changes, so make them yourself, only on a trusted home network, and undo them after training. In PowerShell as administrator:

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
```

To undo it later:

```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Block
```

4. Find each machine's LAN address inside WSL with `ip -br -4 addr`. Pick the one on your home network, for example `192.168.1.23`.

## 3. Check the network before training

On rank 0:

```bash
iperf3 -s
```

On each other machine (replace the address):

```bash
iperf3 -c 192.168.1.23 -t 10
```

Note the measured bandwidth for each machine. One DiLoCo synchronization sends each worker's pseudo-gradient: 2 bytes x the trainable parameters (11,503,874 parameters x 2 bytes = 23.0 MB for the default model). An all-reduce moves more than that over the network: about 2 x (N-1)/N times the payload per worker in a ring, so about 34.5 MB each with 4 workers. At 50 Mbit/s that is roughly 6 seconds per synchronization, once every H inner steps, and longer if the workers share one Wi-Fi access point.

## 4. Start the workers

Rank 0 first, then the others, within a couple of minutes of each other. Same command on each machine, changing only the rank and, if you like, the batch size (smaller GPUs can use a smaller batch):

```bash
bash scripts/launch_lan.sh 0 4 192.168.1.23 /path/to/features 32
```

```bash
bash scripts/launch_lan.sh 1 4 192.168.1.23 /path/to/features 32
```

```bash
bash scripts/launch_lan.sh 2 4 192.168.1.23 /path/to/features 16
```

```bash
bash scripts/launch_lan.sh 3 4 192.168.1.23 /path/to/features 16
```

Arguments: rank, number of machines, rank 0's LAN address, feature folder, batch size. The script picks the network interface that routes to rank 0 and sets `GLOO_SOCKET_IFNAME` to it. Logs go to `runs/lan_rank<r>.log`. Rank 0 also writes `model.pt`, validation metrics after every synchronization, and `test_metrics.json`.

To compare against synchronizing every step on the same machines, add `--mode allreduce` by editing the last line of `launch_lan.sh`, or run `dfmamba.train` directly.

## 5. If it hangs

- **Stuck at start:** a worker cannot reach rank 0 on port 29500. Check the firewall step, the address, and `MASTER_PORT`.
- **Hangs at the first synchronization:** gloo also opens connections between every pair of workers, on random ports. Every machine needs the firewall step, not only rank 0.
- **"Connection reset" midway:** a laptop slept or switched Wi-Fi networks. Disable sleep while training.
- **Wrong interface:** set it by hand, for example `GLOO_SOCKET_IFNAME=eth1 bash scripts/launch_lan.sh ...`.

## 6. Over the internet (machines in different places)

Sections 1-5 need every PC on one network. This section is for workers in different places: GPU machines in containers that expose only a few ports, plus your home PC behind a router.

**How it works.** The normal setup (gloo) makes every worker open random ports to every other worker. Containers often expose only a few ports, and your home PC cannot accept connections from outside. So here every worker connects **out** to one small program, the **hub**, on a single TCP port. Each round the hub collects every worker's update, averages it (weighted by batch size, as before) and sends the result back. Nobody ever connects to the home PC.

The cost of this design: the hub receives N updates and sends N results every round, so its internet link carries N x 23 MB each way per round (default model, bfloat16). The numbers in the training log use these rules:

- `payload_MB_sent`: what this worker uploaded (23.0 MB per round).
- `wire_MB_est`: **2 x payload per worker** (upload plus download). Gloo ring estimate is 2(N-1)/N x payload, so the two are not directly comparable.
- `wait_s`: time this worker sat waiting for the slowest worker (measured by the hub, so clock differences between machines do not matter).
- `comm_s`: everything else in the round (sending, receiving, averaging at the hub).

### 6.1 Set up the machines

1. Use a container image with PyTorch on each machine. One GPU with at least 8 GB is plenty.
2. For the machine that will run the hub, publish the port in the docker options with `-p 29600:29600`. The host may map it to a different external port (for example `203.0.113.7:41234 -> 29600`). Workers must use the **external** port (41234), not 29600.
3. The other containers need no open ports, because they only connect out.
4. Note each container SSH address and port from the host's connection details. Below, `SSHPORT` and `SSHHOST` stand for those.

The hub can run on a worker box (the cheapest option) or on a small separate machine. Choose a box with good upload speed, since it carries N x payload per round.

### 6.2 Put the code and data on each machine

The training features file is about 364 MB. Cut it into one file per worker first, so each box downloads only about 1/N of it. On the machine that has `train.pt` (3 workers in this example):

```bash
python scripts/make_shards.py --src /path/to/features/train.pt --world 3
```

This writes `train_shard0of3.pt`, `train_shard1of3.pt`, `train_shard2of3.pt` next to `train.pt`. Each holds exactly the rows that rank would have kept itself. Copy the repository and that worker shard to its box (plus `val.pt` and `test.pt` for rank 0 only, which does the evaluation and saves the model):

```bash
scp -P SSHPORT -r deepfake-detector-mamba root@SSHHOST:/workspace/
scp -P SSHPORT /path/to/features/train_shard1of3.pt root@SSHHOST:/workspace/features/
scp -P SSHPORT /path/to/features/val.pt /path/to/features/test.pt root@SSHHOST:/workspace/features/
```

If the folder passed as `--data` contains `train_shard{rank}of{world}.pt`, the trainer uses it. Otherwise it falls back to `train.pt` and takes its own rows. If a machine is short on disk or time, add `--train-limit 2000` to use only the first 2000 clips of its share.

On each machine, install what is missing: `pip install pyyaml numpy`.

Set a shared password so strangers cannot join the hub. Use the same value on the hub and on every worker, before starting them:

```bash
export DFM_HUB_TOKEN=pick-a-long-random-string
```

### 6.3 Measure the round-trip time first

From each worker (and from the home PC) to the hub machine, using the hub public IP:

```bash
bash scripts/launch_wan.sh ping HUBIP
```

Note the average RTT and the packet loss next to your iperf results. RTT matters little for DiLoCo (one round per H steps) and a lot for synchronizing every step (one round per step: a 100 ms RTT adds at least 100 ms to every single step).

### 6.4 Start the hub

On the hub box, from the repository folder (the **internal** port is 29600):

```bash
bash scripts/launch_wan.sh hub 29600 3
```

It prints a line when each rank connects and says "all 3 workers connected" when training can start. It exits by itself when the workers finish. If somebody does not show up it stops after 30 minutes (`--timeout`) and names the ranks that never connected.

### 6.5 Start the workers

Use the hub public IP and the **external** port. Start them within a few minutes of each other. Rank 0 can be on the hub box.

```bash
bash scripts/launch_wan.sh worker 0 3 HUBIP:41234 /workspace/features 32 diloco 100
bash scripts/launch_wan.sh worker 1 3 HUBIP:41234 /workspace/features 32 diloco 100
```

The last two arguments are the mode and H.

**Your home PC** (rank 2, WSL2 or Linux). It only makes an outbound connection, so no firewall or router changes are needed:

```bash
export DFM_HUB_TOKEN=pick-a-long-random-string
bash scripts/launch_wan.sh worker 2 3 HUBIP:41234 ~/features 16 diloco 100
```

A smaller batch on a smaller GPU is fine, because the average is weighted by batch size. Without the script, the same worker is `python -m dfmamba.train --config configs/default.yaml --data DIR --out runs/x --transport hub --hub HUBIP:41234 --rank 2 --world 3`. No `torchrun` and no `WORLD_SIZE` are needed in this mode.

### 6.6 The comparison: every step versus DiLoCo H=100

Run the whole thing twice with the same machines and the same step count. The hub exits by itself when the workers finish, so start a fresh one for the second run.

1. **DiLoCo, H=100:** the commands above, with mode `diloco` and H `100`. Logs: `runs/wan_diloco_rank<r>.log`.
2. **Every step:** start a fresh hub, then run the workers with mode `allreduce`:

```bash
bash scripts/launch_wan.sh worker 0 3 HUBIP:41234 /workspace/features 32 allreduce
```

Logs: `runs/wan_allreduce_rank<r>.log`. Compare, per worker: `elapsed_s` (total time), `wait_s` and `comm_s`, `payload_MB_sent`, and rank 0 `val` and `test` metrics. Expect the every-step run to send 100 times more data and to spend most of its time in `comm_s`; the exact slowdown depends on your RTT and bandwidth. Record the RTT from step 6.3 next to the results.

### 6.7 If it hangs or fails

- **"could not reach the hub":** wrong port (use the external one the host shows, not 29600), the hub is not running yet, or the port mapping was not added in the docker options. Workers keep retrying for two minutes.
- **"wrong hub token":** `DFM_HUB_TOKEN` differs between machines, or is unset on one.
- **"already connected" or "world size" mismatch:** two workers use the same rank, or `--world` differs.
- **"timed out ... waiting for ranks [...]":** that worker is dead or very slow. All workers stop with the same message. Restart everything; there is no resume.
- **Limits.** The hub is one machine: its bandwidth is N x payload per round each way, and if it dies the run dies. The connection is not encrypted, so the token only keeps strangers from joining; it does not hide the traffic. If that matters, tunnel through SSH (`ssh -p SSHPORT -L 29600:localhost:29600 root@SSHHOST`) and point workers at `localhost:29600`.

## 7. Cutting the traffic: compression and overlap (hub transport only)

Both options are flags of `dfmamba.train`, so with `scripts/launch_wan.sh` you put them after the config argument (the 9th argument; pass `configs/images.yaml` or your own config explicitly). Every worker must use the same compression flags. The hub needs no flag: it recognises the message type, decodes, averages in fp32, and sends the result back in the same format.

### 7.1 Compression

```bash
bash scripts/launch_wan.sh worker 2 3 HUBIP:41234 ~/features 16 diloco 100 configs/images.yaml --compress int8
```

| Flag | Meaning |
|---|---|
| `--compress none` | default: bfloat16, 2 bytes per parameter, same behaviour as before |
| `--compress int8` | 1 byte per parameter plus one 2-byte scale per 256 values (about 1.008 bytes per parameter) |
| `--compress int4` | 4 bits per parameter, two per byte, plus one 2-byte scale per 64 values (about 0.53 bytes per parameter) |
| `--compress topk --topk-frac 0.1` | send only the largest 10% of entries, 6 bytes each (4-byte index and 2-byte value); `--topk-frac 0.01` keeps 1% |
| `--no-error-feedback` | switch off the residual memory (for comparison only; leave it on) |
| `--stochastic` | random rounding for int8 / int4 instead of nearest |

Error feedback is on by default whenever compression is on. Compression does not work with the gloo transport; the trainer stops with an error if you ask for it.

The log keeps the same fields: `payload_MB_sent` is what this worker uploaded, and for compressed runs `wire_MB_est` is the measured upload plus the measured download.

### 7.2 Overlap

```bash
bash scripts/launch_wan.sh worker 2 3 HUBIP:41234 ~/features 16 diloco 100 configs/images.yaml --overlap
```

With `--overlap` the worker starts the exchange in a background thread and keeps training. The averaged update is applied `--overlap-delay` inner steps later (default: H, which hides a whole segment of compute behind the network). `--overlap-alpha` (default 0.5) is how much of the worker's own parameters survive the merge. New log fields: `blocked_s` (time the training loop really waited for the network) and `hidden_s` (exchange time that ran behind training). Overlap works with DiLoCo only. It can be combined with `--compress`.

### 7.3 Rehearse a slow link on one machine

`dfmamba.hub` has two test-only flags that slow its connections down, so you can see the effect of compression and overlap without a second machine:

```bash
python -m dfmamba.hub --port 29600 --world 4 --host 127.0.0.1 --sim-latency-ms 125 --sim-mbps 20
```

`--sim-latency-ms 125` delays every message by 125 ms in each direction (a 250 ms round trip). `--sim-mbps 20` caps each worker's link at 20 megabits per second (2.5 MB/s) in each direction, independently per worker. Do not use these flags for real training.
