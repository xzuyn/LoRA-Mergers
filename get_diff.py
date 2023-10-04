# Version: 0.18
# Created by: xzuyn
# Description: Script to subtract one model from another. Also gives the option
#              to apply that element-wise difference onto another model.
#              Currently, with argparse you have to apply the difference to a
#              model. If you use the functions you can just get the dictionary
#              of the difference, which you can then save, or play around with.
#
#              Idea came from the way we can make Inpainting SD models out of
#              Non-Inpainting SD models by removing the Base SD model from the
#              Inpainting SD model to then be left with only the Inpainting
#              elements. That then can be mixed with any Non-Inpainting model
#              to create a properly working Inpainting model.

import torch
from transformers import AutoModel
from safetensors.torch import save_file, safe_open
from tqdm import tqdm
import argparse
import gc


def get_applied_diff_pytorch(
    base: str,
    a: str,
    b: str,
    sub_alpha: float = 1,
    apl_alpha: float = 1,
    device1: str = "cpu",
    device2: str = "cuda",
    is_safetensors: bool = False,
):
    """
    applied_diff_pytorch = a + ((b - (base * sub_alpha)) * apl_alpha)
    applied_diff_pytorch = 'airo' + (('hermes' - ('L2' * sub_alpha)) * apl_alpha)

        >>> applied_diff_pytorch = get_applied_diff_pytorch(
        >>>     'base_model.bin',
        >>>     'model_a.bin',
        >>>     'model_b.bin',
        >>>     sub_alpha=1,
        >>>     apl_alpha=1,
        >>>     device1='cpu',
        >>>     device2='cuda'
        >>>)
    """
    diff_model_pytorch = {}
    applied_diff_pytorch = {}
    if is_safetensors:
        basePyTorch = {}
        bPyTorch = {}
        with safe_open(base, framework="pt", device=device1) as f:
            for k in f.keys():
                basePyTorch[k] = f.get_tensor(k)
        with safe_open(b, framework="pt", device=device1) as f:
            for k in f.keys():
                bPyTorch[k] = f.get_tensor(k)
    else:
        basePyTorch = torch.load(base, map_location=device1)
        bPyTorch = torch.load(b, map_location=device1)

    print("Getting Difference.")
    for k in tqdm(basePyTorch.keys()):
        if k in bPyTorch.keys():
            if device2 != device1:
                basePyTorch[k], bPyTorch[k] = (
                    basePyTorch[k].to(device2),
                    bPyTorch[k].to(device2),
                )
            diff_model_pytorch[k] = torch.add(
                input=bPyTorch[k],
                other=basePyTorch[k],
                alpha=sub_alpha,
            )
        else:
            if device2 != device1:
                basePyTorch[k] = basePyTorch[k].to(device2)
            diff_model_pytorch[k] = torch.mul(input=basePyTorch[k], other=0)
        basePyTorch[k], bPyTorch[k] = None, None
        gc.collect()

    basePyTorch, bPyTorch = None, None
    gc.collect()

    if is_safetensors:
        aPyTorch = {}
        with safe_open(b, framework="pt", device=device1) as f:
            for k in f.keys():
                aPyTorch[k] = f.get_tensor(k)
    else:
        aPyTorch = torch.load(a, map_location=device1)

    print("Applying Difference.")
    for k in tqdm(diff_model_pytorch.keys()):
        if k in aPyTorch.keys():
            if device2 != device1:
                aPyTorch[k] = aPyTorch[k].to(device2)
            applied_diff_pytorch[k] = torch.add(
                input=aPyTorch[k], other=diff_model_pytorch[k], alpha=apl_alpha
            )
        aPyTorch[k], diff_model_pytorch[k] = None, None
        gc.collect()

    aPyTorch, diff_model_pytorch = None, None
    gc.collect()

    return applied_diff_pytorch


def get_applied_diff_model(
    base: str,
    a: str,
    b: str,
    sub_alpha: float = 1,
    apl_alpha: float = 1,
    device1: str = "cpu",
    device2: str = "cuda",
):
    """
    applied_diff_model = a + ((b - (base * sub_alpha)) * apl_alpha)
    applied_diff_model = 'airo' + (('hermes' - ('L2' * sub_alpha)) * apl_alpha)

        >>> applied_diff = get_applied_diff_model(
        >>>     'meta-llama/Llama-2-7b-hf',
        >>>     'jondurbin/airoboros-l2-7b-2.2.1',
        >>>     'NousResearch/Nous-Hermes-llama-2-7b',
        >>>     sub_alpha=1,
        >>>     apl_alpha=1,
        >>>     device1='cuda',
        >>>     device2='cpu'
        >>>)
    """
    diff_model = {}
    applied_diff_model = {}

    baseModel = AutoModel.from_pretrained(
        base, load_in_8bit=False, torch_dtype=torch.float16, device_map=device1
    )

    bModel = AutoModel.from_pretrained(
        b, load_in_8bit=False, torch_dtype=torch.float16, device_map=device1
    )

    print("Getting Difference.")
    for k in tqdm(baseModel.state_dict().keys()):
        if k in bModel.state_dict().keys():
            if device2 != device1:
                (
                    baseModel.state_dict()[k],
                    bModel.state_dict()[k],
                ) = baseModel.state_dict()[k].to(device2), bModel.state_dict()[
                    k
                ].to(
                    device2
                )
            diff_model[k] = torch.sub(
                input=bModel.state_dict()[k],
                other=baseModel.state_dict()[k],
                alpha=sub_alpha,
            )
        else:
            if device2 != device1:
                baseModel.state_dict()[k] = baseModel.state_dict()[k].to(
                    device2
                )
            diff_model[k] = torch.mul(input=baseModel.state_dict()[k], other=0)
        baseModel.state_dict()[k], bModel.state_dict()[k] = None, None
        gc.collect()

    baseModel, bModel = None, None
    gc.collect()

    aModel = AutoModel.from_pretrained(
        a, load_in_8bit=False, torch_dtype=torch.float16, device_map=device1
    )

    print("Applying Difference.")
    for k in tqdm(diff_model.keys()):
        if k in aModel.state_dict().keys():
            applied_diff_model[k] = torch.add(
                input=aModel.state_dict()[k],
                other=diff_model[k],
                alpha=apl_alpha,
            )
        aModel.state_dict()[k], diff_model[k] = None, None
        gc.collect()

    aModel, diff_model = None, None
    gc.collect()

    return applied_diff_model


def main(args):
    if args.mode == "pytorch":
        result = get_applied_diff_pytorch(
            base=args.pytorch_base,
            a=args.pytorch_a,
            b=args.pytorch_b,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device1=args.device1,
            device2=args.device2,
            is_safetensors=args.is_safetensors,
        )
    elif args.mode == "model":
        result = get_applied_diff_model(
            base=args.model_base,
            a=args.model_a,
            b=args.model_b,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device1=args.device1,
            device2=args.device2,
        )
    else:
        raise ValueError("Invalid mode. Please choose 'adapter' or 'model'.")

    if args.save_format == "safetensors":
        save_file(tensors=result, filename=args.output_path)
    elif args.save_format == "pytorch":
        torch.save(obj=result, f=args.output_path)
    else:
        raise ValueError(
            "Invalid format. Please choose 'safetensors' or 'pytorch'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save applied differential updates for"
        "adapters or models."
    )
    parser.add_argument(
        "--mode",
        choices=["pytorch", "model"],
        required=True,
        help="Choose 'pytorch', 'model' mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Choose 'safetensors', 'pytorch' mode.",
    )
    parser.add_argument(
        "--sub_alpha",
        type=float,
        default=1,
        help="Scaling factor for difference (default: 1).",
    )
    parser.add_argument("--apl_alpha", type=float, default=1)
    parser.add_argument("--device1", type=str, default="cpu"),
    parser.add_argument("--device2", type=str, default="cuda"),
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the result.",
    )
    parser.add_argument(
        "--pytorch_base",
        type=str,
        help="Path to any PyTorch model which you'd like to treat as the 'base' model. This will be subtracted from adapter_b so that we hopefully don't duplicate that information during merging.",
    )
    parser.add_argument(
        "--pytorch_a",
        type=str,
        help="Path to any PyTorch model which will be your main model. The difference from base and adapter_b will be applied to this model.",
    )
    parser.add_argument(
        "--pytorch_b",
        type=str,
        help="Path to any PyTorch model which will be the model you are trying to get the difference of to apply onto adapter_a.",
    )
    parser.add_argument("--model_base", type=str)
    parser.add_argument("--model_a", type=str)
    parser.add_argument("--model_b", type=str)
    parser.add_argument("--is_safetensors", type=bool, default=False)

    args = parser.parse_args()
    main(args)
