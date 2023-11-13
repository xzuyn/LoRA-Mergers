import os
import torch
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import gc
import time


def twopiece(afilestr, bfilestr, device="cuda"):
    a = {}
    b = {}
    twopiece = {}
    with safe_open(afilestr, framework="pt", device=device) as f:
        for k in f.keys():
            with safe_open(afilestr, framework="pt", device=device) as fa, safe_open(
                bfilestr, framework="pt", device=device
            ) as fb:
                if k in fb.keys():
                    a[k], b[k] = (
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                    )
                    print(
                        f"Merging A & B tensors for key: {k}, Size: {a[k].size()}"
                    )
                    twopiece[k] = ((a[k] + b[k]) / 2).to("cpu")
                    a[k], b[k] = None, None
                else:
                    a[k] = fa.get_tensor(k)
                    print(f"Keeping A tensor for key: {k}, Size: {a[k].size()}")
                    twopiece[k] = (a[k]).to("cpu")
                    a[k] = None

            a[k], b[k] = None, None
            gc.collect()

    a, b = None, None
    gc.collect()

    return twopiece


def main():
    parser = argparse.ArgumentParser(description="Merge two safetensors models.")
    parser.add_argument(
        "--a", type=str, help="Path to the first model file (ie. 'a.safetensors')"
    )
    parser.add_argument(
        "--b", type=str, help="Path to the second model file (ie. 'b.safetensors')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file (ie. 'merge.safetensors')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for processing (default: cuda)",
    )
    args = parser.parse_args()

    start_time = time.time()
    result = twopiece(args.a, args.b, args.device)
    save_file(result, args.output)
    end_time = time.time()

    print(f"Time to merge 2 models: {end_time - start_time}s")


if __name__ == "__main__":
    main()
