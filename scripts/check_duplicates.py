import argparse
import collections
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser(description="MD5-hash every image under <root>/{Train,Validation,Test}/{Real,Fake} and count exact duplicates across splits.")
ap.add_argument("root")
ap.add_argument("--out", default="results/dataset_duplicates/dup.json")
args = ap.parse_args()

files = []
for sp in ("Train", "Validation", "Test"):
    for c in ("Real", "Fake"):
        d = os.path.join(args.root, sp, c)
        for f in os.listdir(d):
            files.append((sp, c, os.path.join(d, f)))


def md5(t):
    try:
        with open(t[2], "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


with ThreadPoolExecutor(16) as ex:
    hs = list(ex.map(md5, files))

by = collections.defaultdict(list)
for (sp, c, p), x in zip(files, hs):
    if x:
        by[x].append((sp, c, os.path.basename(p)))

res = {"files": len(files), "unreadable": sum(x is None for x in hs), "unique": len(by)}
pairs = collections.Counter()
cross_label = 0
examples = []
for x, group in by.items():
    if len(group) > 1:
        splits = tuple(sorted(set(s for s, _, _ in group)))
        pairs[splits] += 1
        if len(set(c for _, c, _ in group)) > 1:
            cross_label += 1
        if len(splits) > 1 and len(examples) < 10:
            examples.append(group)


def images_in_split_with_copy_in(split, other):
    n = 0
    for group in by.values():
        splits = {s for s, _, _ in group}
        if split in splits and other in splits:
            n += sum(1 for s, _, _ in group if s == split)
    return n


res.update(
    dup_groups_by_splits={str(k): v for k, v in pairs.items()},
    cross_label_groups=cross_label,
    test_imgs_with_train_copy=images_in_split_with_copy_in("Test", "Train"),
    test_imgs_with_val_copy=images_in_split_with_copy_in("Test", "Validation"),
    val_imgs_with_train_copy=images_in_split_with_copy_in("Validation", "Train"),
    examples=examples,
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
