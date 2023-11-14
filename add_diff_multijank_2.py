import os
import torch
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import gc
import time


def diff(basefilestr, afilestr, bfilestr, cfilestr, dfilestr, device="cuda"):
    a = {}
    b = {}
    c = {}
    d = {}
    base = {}
    diff = {}
    with safe_open(afilestr, framework="pt", device=device) as f:
        for k in f.keys():
            with safe_open(
                basefilestr, framework="pt", device=device
            ) as fbase, safe_open(
                afilestr, framework="pt", device=device
            ) as fa, safe_open(
                bfilestr, framework="pt", device=device
            ) as fb, safe_open(
                cfilestr, framework="pt", device=device
            ) as fc, safe_open(
                dfilestr, framework="pt", device=device
            ) as fd:
                if (
                    k in fbase.keys()
                    and k in fb.keys()
                    and k in fc.keys()
                    and k in fd.keys()
                ):
                    base[k], a[k], b[k], c[k], d[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                        fc.get_tensor(k),
                        fd.get_tensor(k),
                    )
                    print(
                        f"Adding difference from B, C, & D to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (
                        ((b[k] - base[k]) * (1 / 3))
                        + ((c[k] - base[k]) * (1 / 3))
                        + ((d[k] - base[k]) * (1 / 3))
                    )
                    base[k], a[k], b[k], c[k], d[k] = None, None, None, None, None
                elif (
                    k in fbase.keys()
                    and k not in fb.keys()
                    and k in fc.keys()
                    and k in fd.keys()
                ):
                    base[k], a[k], c[k], d[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fc.get_tensor(k),
                        fd.get_tensor(k),
                    )
                    print(
                        f"Adding difference from C & D to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (
                        ((c[k] - base[k]) * (1 / 2)) + ((d[k] - base[k]) * (1 / 2))
                    )
                    base[k], a[k], c[k], d[k] = None, None, None, None
                elif (
                    k in fbase.keys()
                    and k in fb.keys()
                    and k not in fc.keys()
                    and k in fd.keys()
                ):
                    base[k], a[k], b[k], d[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                        fd.get_tensor(k),
                    )
                    print(
                        f"Adding difference from B & D to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (
                        ((b[k] - base[k]) * (1 / 2)) + ((d[k] - base[k]) * (1 / 2))
                    )
                    base[k], a[k], b[k], d[k] = None, None, None, None
                elif (
                    k in fbase.keys()
                    and k in fb.keys()
                    and k in fc.keys()
                    and k not in fd.keys()
                ):
                    base[k], a[k], b[k], c[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                        fc.get_tensor(k),
                    )
                    print(
                        f"Adding difference from B & C to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (
                        ((b[k] - base[k]) * (1 / 2)) + ((c[k] - base[k]) * (1 / 2))
                    )
                    base[k], a[k], b[k], c[k] = None, None, None, None
                elif (
                    k in fbase.keys()
                    and k in fb.keys()
                    and k not in fc.keys()
                    and k not in fd.keys()
                ):
                    base[k], a[k], b[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fb.get_tensor(k),
                    )
                    print(
                        f"Adding difference from B to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (b[k] - base[k])
                    base[k], a[k], b[k] = None, None, None
                elif (
                    k in fbase.keys()
                    and k not in fb.keys()
                    and k in fc.keys()
                    and k not in fd.keys()
                ):
                    base[k], a[k], c[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fc.get_tensor(k),
                    )
                    print(
                        f"Adding difference from C to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (c[k] - base[k])
                    base[k], a[k], c[k] = None, None, None
                elif (
                    k in fbase.keys()
                    and k not in fb.keys()
                    and k not in fc.keys()
                    and k in fd.keys()
                ):
                    base[k], a[k], d[k] = (
                        fbase.get_tensor(k),
                        fa.get_tensor(k),
                        fd.get_tensor(k),
                    )
                    print(
                        f"Adding difference from D to A for key: {k}, Size: {a[k].size()}"
                    )
                    diff[k] = a[k] + (d[k] - base[k])
                    base[k], a[k], d[k] = None, None, None
                else:
                    a[k] = fa.get_tensor(k)
                    print(f"Keeping A tensor for key: {k}, Size: {a[k].size()}")
                    diff[k] = a[k]
                    a[k] = None

            base[k], a[k], b[k], c[k] = None, None, None, None
            gc.collect()

    base, a, b, c = None, None, None, None
    gc.collect()

    return diff


def main():
    parser = argparse.ArgumentParser(
        description="Add the differences other models to another model."
    )
    parser.add_argument(
        "--base",
        type=str,
        help="Path to the base model file (ie. 'sd_xl_base_1.0.safetensors')",
    )
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
        "--d", type=str, help="Path to the forth model file (ie. 'd.safetensors')"
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
    result = diff(args.base, args.a, args.b, args.c, args.d, args.device)
    save_file(result, args.output)
    end_time = time.time()

    print(f"Time to add difference: {end_time - start_time}s")


if __name__ == "__main__":
    main()
