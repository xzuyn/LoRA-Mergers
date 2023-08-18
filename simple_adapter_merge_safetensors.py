import os, torch
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

def Blend(aPath, bPath, outName):
    a = {}
    b = {}
    with safe_open(aPath, framework="pt", device="cpu") as fa:
        for ka in fa.keys():
            a[ka] = fa.get_tensor(ka)
    with safe_open(bPath, framework="pt", device="cpu") as fb:
        for kb in fb.keys():
            b[kb] = fb.get_tensor(kb)
    
    all_keys = list(set(a.keys()) | set(b.keys()))  # Get all unique keys
    
    for key in tqdm(all_keys):
        if key in a and key in b:
            a[key] = (a[key] + b[key]) / 2
        elif key in b:
            a[key] = b[key]
        # No need to handle the case where key is only in 'a', as it is already in 'a'

    save_file(a, outName)

Blend(
    "Gloomifier_V2_TheGlow.safetensors",
    "Gloomifier_TheDread_V1_LECO.safetensors",
    "GlowingDread.safetensors"
)
