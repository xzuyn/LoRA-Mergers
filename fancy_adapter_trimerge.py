import os, torch
from tqdm import tqdm

def Blend(aPath, bPath, outName):
    a = torch.load(aPath, map_location="cpu")
    b = torch.load(bPath, map_location="cpu")
    c = torch.load(cPath, map_location="cpu")
    for key in tqdm(a.keys()):
        a[key] = b[key] * (abs(a[key] - b[key]) > abs(a[key] - c[key])) + c[key] * (abs(a[key] - b[key]) <= abs(a[key] - c[key]))
    torch.save(a, outName)

Blend("adapter_model_1.bin", "adapter_model_2.bin", "adapter_model_3", "adapter_merge.bin")
