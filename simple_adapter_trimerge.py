import os, torch
from tqdm import tqdm

def Blend(aPath, bPath, cPath, outName):
    a = torch.load(aPath, map_location="cpu")
    b = torch.load(bPath, map_location="cpu")
    c = torch.load(cPath, map_location="cpu")
    for key in tqdm(a.keys()):
        a[key] = (a[key] + b[key] + c[key]) / 3
    torch.save(a, outName)

Blend("adapter_model_1.bin", "adapter_model_2.bin", "adapter_model_3", "adapter_merge.bin")
