import json
import sys
from pathlib import Path

ORDER = ["single_no_temporal", "single_H1", "allreduce_H1", "diloco_H50", "diloco_H100", "diloco_H500"]
LABEL = {"single_no_temporal": "1 worker, no Mamba layers",
         "single_H1": "1 worker, Mamba",
         "allreduce_H1": "4 workers, sync gradients every step",
         "diloco_H50": "4 workers, DiLoCo, H = 50",
         "diloco_H100": "4 workers, DiLoCo, H = 100",
         "diloco_H500": "4 workers, DiLoCo, H = 500"}


def main():
    root = Path(sys.argv[1])
    rows = []
    for name in ORDER:
        d = root / name
        if not (d / "log_rank0.json").exists():
            continue
        log = json.loads((d / "log_rank0.json").read_text())
        info = log[0]
        last = [r for r in log if "step" in r][-1]
        test = json.loads((d / "test_metrics.json").read_text()) if (d / "test_metrics.json").exists() else {}
        multi = info["world"] > 1
        rows.append(f"| {LABEL[name]} | {info['inner_steps'] if info['mode'] == 'diloco' else ('every step' if multi else '-')} | "
                    f"{last['syncs'] if multi else 0} | {last['payload_MB_sent'] if multi else 0:.1f} | "
                    f"{last['wire_MB_est']:.1f} | {test.get('accuracy', float('nan')) * 100:.1f}% | {test.get('auc', float('nan')):.3f} |")
    steps = info0 = None
    for name in ORDER:
        f = root / name / "log_rank0.json"
        if f.exists():
            L = json.loads(f.read_text())
            info0, steps = L[0], [r for r in L if "step" in r][-1]["step"]
            break
    head = [f"| Run ({steps:,} steps per worker, batch {info0['batch_size']}) | Inner steps H | Syncs | Payload sent per worker (MB) | Est. ring all-reduce traffic per worker (MB) | Test accuracy | Test AUC |",
            "|---|---|---|---|---|---|---|"]
    print("\n".join(head + rows))


if __name__ == "__main__":
    main()
