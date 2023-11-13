import os
import torch
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import gc
import time


def threepiece(afilestr, bfilestr, cfilestr, device="cuda"):
    a = {}
    b = {}
    c = {}
    threepiece = {}
    with safe_open(afilestr, framework="pt", device=device) as f:
        for k in f.keys():
            with safe_open(afilestr, framework="pt", device=device) as fa, safe_open(
                bfilestr, framework="pt", device=device
            ) as fb, safe_open(cfilestr, framework="pt", device=device) as fc:
                if k in fb.keys() and k in fc.keys():
                    a[k], b[k], c[k] = (
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                        fc.get_tensor(k),
                    )
                    print(
                        f"Merging A, B, & C tensors for key: {k}, Size: {a[k].size()}"
                    )
                    threepiece[k] = ((a[k] + b[k] + c[k]) / 3).to("cpu")
                    a[k], b[k], c[k] = None, None, None
                elif k in fb.keys() and k not in fc.keys():
                    a[k], b[k] = (
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                    )
                    print(f"Merging A & B tensors for key: {k}, Size: {a[k].size()}")
                    threepiece[k] = ((a[k] + b[k]) / 2).to("cpu")
                    a[k], b[k] = None, None
                elif k in fc.keys() and k not in fb.keys():
                    a[k], c[k] = (
                        fa.get_tensor(k),
                        fc.get_tensor(k),
                    )
                    print(f"Merging A & C tensors for key: {k}, Size: {a[k].size()}")
                    threepiece[k] = ((a[k] + c[k]) / 2).to("cpu")
                    a[k], c[k] = None, None
                else:
                    a[k] = fa.get_tensor(k)
                    print(f"Keeping A tensor for key: {k}, Size: {a[k].size()}")
                    threepiece[k] = (a[k]).to("cpu")
                    a[k] = None

            a[k], b[k], c[k] = None, None, None
            gc.collect()

    a, b, c = None, None, None
    gc.collect()

    return threepiece


def main():
    parser = argparse.ArgumentParser(description="Merge three safetensors models.")
    parser.add_argument(
        "--a", type=str, help="Path to the first model file (ie. 'a.safetensors')"
    )
    parser.add_argument(
        "--b", type=str, help="Path to the second model file (ie. 'b.safetensors')"
    )
    parser.add_argument(
        "--c", type=str, help="Path to the third model file (ie. 'c.safetensors')"
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
    result = threepiece(args.a, args.b, args.c, args.device)
    save_file(result, args.output)
    end_time = time.time()

    print(f"Time to merge 3 models: {end_time - start_time}s")


if __name__ == "__main__":
    main()
