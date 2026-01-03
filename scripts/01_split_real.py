import argparse
from collections import defaultdict
import random
from swellsight.utils import read_jsonl, write_jsonl, ensure_dir


def stratified_split(items, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for r in items:
        key = (r["wave_type"], r["direction"])
        buckets[key].append(r)

    train, val, test = [], [], []

    for _, bucket in buckets.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_index", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = read_jsonl(args.real_index)
    train, val, test = stratified_split(items, args.train_ratio, args.val_ratio, args.seed)

    ensure_dir(args.out_dir)
    write_jsonl(train, f"{args.out_dir}/train.jsonl")
    write_jsonl(val, f"{args.out_dir}/val.jsonl")
    write_jsonl(test, f"{args.out_dir}/test.jsonl")

    print(f"Train {len(train)}, Val {len(val)}, Test {len(test)}")


if __name__ == "__main__":
    main()
