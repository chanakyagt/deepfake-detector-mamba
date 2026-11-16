import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent / "results" / "speed"
raw = sorted((D / "raw").glob("*.json"))
recs = []
for p in raw:
    recs += json.loads(p.read_text())
(D / "speed.json").write_text(json.dumps(recs, indent=1))
notes = {p.stem for p in raw}


def f(v, nd=1):
    return "n/a" if v is None else f"{v:,.{nd}f}"


rows = []
for r in recs:
    rows.append(r)
rows.sort(key=lambda r: r.get("gflops_per_unit") or 0)
head = ("| Model | Input | Params total (M) | Trainable (M) | Feature dim (backbone out) | GFLOPs / image | bf16 b=1 ms | fp32 b=1 ms "
        "| bf16 img/s b=64 | fp32 img/s b=64 | Peak MB b=1 (bf16) | Peak MB b=64 (bf16) | SSM state (KB) |\n"
        "|" + "---|" * 13)
lines = [head]
for r in rows:
    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        r["model"], r["input_resolution"], f(r["params_total"] / 1e6, 2), f(r["params_trainable"] / 1e6, 2),
        r.get("feature_dim_from_backbone") or r.get("classifier_input_dim"), f(r.get("gflops_per_unit"), 2), f(r.get("latency_ms_bf16_b1"), 2),
        f(r.get("latency_ms_fp32_b1"), 2), f(r.get("throughput_units_per_s_bf16_b%d" % r["big_batch"]), 0),
        f(r.get("throughput_units_per_s_fp32_b%d" % r["big_batch"]), 0), f(r.get("peak_mem_MB_bf16_b1"), 0),
        f(r.get("peak_mem_MB_bf16_b%d" % r["big_batch"]), 0), f(r.get("ssm_state_KB_fp32") or None, 0)))
table = "\n".join(lines) + "\n"
(D / "table.md").write_text(table)
tpl = Path(__file__).resolve().parent / "speed_readme_template.md"
(D / "README.md").write_text(tpl.read_text().replace("{{TABLE}}", table))
print(table)
