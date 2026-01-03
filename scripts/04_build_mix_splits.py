import argparse
import random
from swellsight.utils import read_jsonl, write_jsonl, ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_splits_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--real_aug_index", default="")
    p.add_argument("--param_synth_index", default="")

    p.add_argument("--real_keep_train", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    real_train = read_jsonl(f"{args.real_splits_dir}/train.jsonl")
    real_val = read_jsonl(f"{args.real_splits_dir}/val.jsonl")
    real_test = read_jsonl(f"{args.real_splits_dir}/test.jsonl")

    # Train mix: real (optionally downsampled) + synthetic
    rng.shuffle(real_train)
    keep_n = int(round(len(real_train) * args.real_keep_train))
    real_train_kept = real_train[:keep_n]

    mix_train = list(real_train_kept)

    if args.real_aug_index:
        mix_train.extend(read_jsonl(args.real_aug_index))
    if args.param_synth_index:
        mix_train.extend(read_jsonl(args.param_synth_index))

    rng.shuffle(mix_train)

    # Val and test: real only
    mix_val = real_val
    mix_test = real_test

    ensure_dir(args.out_dir)
    write_jsonl(mix_train, f"{args.out_dir}/train.jsonl")
    write_jsonl(mix_val, f"{args.out_dir}/val.jsonl")
    write_jsonl(mix_test, f"{args.out_dir}/test.jsonl")

    print(f"Mix train {len(mix_train)}, val {len(mix_val)}, test {len(mix_test)}")


if __name__ == "__main__":
    main()
