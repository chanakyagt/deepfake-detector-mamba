import json
import sys
from pathlib import Path

ORDER = ["bf16", "int8_ef", "int4_ef", "topk10_ef", "topk1_ef", "int8_noef", "int4_noef", "topk1_noef", "bf16_overlap", "int8_ef_overlap"]


def load(run):
    logs = [json.loads((run / f"log_rank{r}.json").read_text()) for r in range(4) if (run / f"log_rank{r}.json").exists()]
    test = json.loads((run / "test_metrics.json").read_text())
    val = json.loads((run / "val_metrics.json").read_text()) if (run / "val_metrics.json").exists() else {}
    last = [[x for x in log if "payload_MB_sent" in x][-1] for log in logs]
    info = logs[0][0]
    return {
        "test_acc": test["accuracy"], "test_auc": test["auc"], "val_acc": val.get("accuracy"), "val_auc": val.get("auc"),
        "payload_MB": last[0]["payload_MB_sent"], "wire_MB": last[0]["wire_MB_est"],
        "per_sync_MB": info["payload_MB"],
        "comm_s": sum(x["comm_s"] for x in last) / len(last), "wait_s": sum(x["wait_s"] for x in last) / len(last),
        "blocked_s": sum(x.get("blocked_s", 0) for x in last) / len(last),
        "hidden_s": sum(x.get("hidden_s", 0) for x in last) / len(last),
        "elapsed_s": max(x["elapsed_s"] for x in last), "syncs": last[0]["syncs"],
    }


def main():
    root = Path(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = {}
    for name in ORDER:
        run = root / (name + suffix)
        if (run / "test_metrics.json").exists():
            rows[name] = load(run)
    base = rows.get("bf16")
    print("| run | test acc | test AUC | val acc | val AUC | payload MB/worker | reduction vs bf16 | wire MB/worker | exchange s (mean/worker) | exposed s | hidden s | wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        red = f"{base['payload_MB'] / r['payload_MB']:.1f}x" if base else "-"
        exch = r["comm_s"] + r["wait_s"]
        exposed = r["blocked_s"] if "overlap" in name else exch
        print(f"| {name} | {100 * r['test_acc']:.1f}% | {r['test_auc']:.3f} | {100 * r['val_acc']:.1f}% | {r['val_auc']:.3f} | "
              f"{r['payload_MB']:.1f} | {red} | {r['wire_MB']:.1f} | {exch:.0f} | {exposed:.0f} | {r['hidden_s']:.0f} | {r['elapsed_s']:.0f} |")
    (root / f"results_raw{suffix}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
