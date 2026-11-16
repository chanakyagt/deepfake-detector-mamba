#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <rank> <world_size> <rank0_lan_ip> <feature_dir> [batch_size] [config]"
  exit 1
fi

RANK=$1
WORLD=$2
MASTER=$3
DATA=$4
BATCH=${5:-32}
CFG=${6:-configs/default.yaml}
PORT=${MASTER_PORT:-29500}
OWN=$(ip -o -4 addr | awk -v ip="$MASTER" '{split($4, a, "/"); if (a[1] == ip) print $2}' | head -n1)
ROUTE=$(ip -o -4 route get "$MASTER" | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
IFACE=${GLOO_SOCKET_IFNAME:-${OWN:-$ROUTE}}
export GLOO_SOCKET_IFNAME=$IFACE

cd "$(dirname "$0")/.."
mkdir -p runs
echo "rank $RANK/$WORLD, master $MASTER:$PORT, interface $IFACE, batch $BATCH"
torchrun --nnodes="$WORLD" --nproc_per_node=1 --node_rank="$RANK" \
  --master_addr="$MASTER" --master_port="$PORT" \
  -m dfmamba.train --config "$CFG" --data "$DATA" --out "runs/lan_rank$RANK" --batch-size "$BATCH" \
  2>&1 | tee "runs/lan_rank$RANK.log"
