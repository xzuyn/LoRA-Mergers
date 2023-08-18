import os, torch
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

def Blend(aPath, bPath, cPath, outName):
    a = {}
    b = {}
    c = {}
    only_copy = True
    with safe_open(aPath, framework="pt", device="cpu") as fa:
        for ka in fa.keys():
            a[ka] = fa.get_tensor(ka)
            
    with safe_open(bPath, framework="pt", device="cpu") as fb:
        for kb in fb.keys():
            b[kb] = fb.get_tensor(kb)
            
    with safe_open(cPath, framework="pt", device="cpu") as fc:
        for kc in fc.keys():
            c[kc] = fc.get_tensor(kc)
    
    all_keys = list(set(a.keys()) | set(b.keys()) | set(c.keys()))  # Get all unique keys
    
    for key in tqdm(all_keys):
        if key in a and key in b and key in c:
            print(f"Merging tensors for key: {key}, Size: {a[key].size()}")
            a[key] = (a[key] + b[key] + c[key]) / 3
            only_copy = False
        elif key in b and key in c:
            print(f"Merging tensors for key: {key}, Size: {a[key].size()}")
            a[key] = (b[key] + c[key]) / 2
            only_copy = False
        elif key in a and key in c:
            print(f"Merging tensors for key: {key}, Size: {a[key].size()}")
            a[key] = (a[key] + c[key]) / 2
            only_copy = False
        elif key in b:
            print(f"Copying tensor for key: {key}, Size: {b[key].size()}")
            a[key] = b[key]
        elif key in c:
            print(f"Copying tensor for key: {key}, Size: {b[key].size()}")
            a[key] = c[key]
        # No need to handle the cases where key is only in 'a', as it is already in 'a'

    if only_copy is True:
       print("No merge happened, only copies. Not saving.")
    elif only_copy is False:
        save_file(a, outName)

Blend(
    ".safetensors",
    ".safetensors",
    ".safetensors",
    ".safetensors"
)
