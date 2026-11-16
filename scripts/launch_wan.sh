#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "hub:    $0 hub <internal_port> <world_size>"
  echo "worker: $0 worker <rank> <world_size> <hub_host:external_port> <feature_dir> [batch_size] [diloco|allreduce] [inner_steps] [config]"
  echo "ping:   $0 ping <hub_host>"
  exit 1
}

[ "$#" -ge 2 ] || usage
cd "$(dirname "$0")/.."
mkdir -p runs
CMD=$1

case "$CMD" in
  hub)
    [ "$#" -ge 3 ] || usage
    python -m dfmamba.hub --port "$2" --world "$3" 2>&1 | tee runs/wan_hub.log
    ;;
  worker)
    [ "$#" -ge 5 ] || usage
    RANK=$2
    WORLD=$3
    HUB=$4
    DATA=$5
    BATCH=${6:-32}
    MODE=${7:-diloco}
    H=${8:-100}
    CFG=${9:-configs/default.yaml}
    EXTRA=("${@:10}")
    echo "rank $RANK/$WORLD, hub $HUB, mode $MODE, H $H, batch $BATCH"
    python -m dfmamba.train --config "$CFG" --data "$DATA" --out "runs/wan_${MODE}_rank$RANK" \
      --transport hub --hub "$HUB" --rank "$RANK" --world "$WORLD" \
      --mode "$MODE" --inner-steps "$H" --batch-size "$BATCH" "${EXTRA[@]}" \
      2>&1 | tee "runs/wan_${MODE}_rank$RANK.log"
    ;;
  ping)
    ping -c 20 "$2" | tee runs/wan_ping.log
    ;;
  *)
    usage
    ;;
esac
