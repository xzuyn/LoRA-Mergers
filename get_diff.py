# Version: 0.10
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


def get_diff_adapter(
    a: str, b: str, sub_alpha: float = 1, device: str = "cpu"
):
    """
    Compute the difference between two PyTorch dictionaries.

    This function takes two file paths `a` and `b`, loads PyTorch dictionaries
    from these files, and computes the element-wise difference between
    corresponding keys in the dictionaries. diff_adapter = (a - (b * sub_alpha)).

    Args:
        a (str): File path to the first PyTorch dictionary.
        b (str): File path to the second PyTorch dictionary.
        sub_alpha (Number): The multiplier for b.
        device (str): Specifies the device (e.g., 'cpu' or 'cuda') on which
            to load the models.

    Returns:
        dict: A dictionary containing the element-wise difference between
              corresponding keys in the input PyTorch dictionaries.

    Example:
        Suppose you have two saved PyTorch dictionaries in 'a.bin' and 'b.bin',
        and you want to compute the difference between them:

        >>> result = get_diff_adapter('a.bin', 'b.bin', 1, 'cuda')
    """
    diff_adapter = {}

    # Load adapter using torch.load to avoid needing the base model
    aLoRA = torch.load(a, map_location=device)
    bLoRA = torch.load(b, map_location=device)

    for k in tqdm(aLoRA.keys()):
        if k in bLoRA.keys():
            diff_adapter[k] = torch.sub(
                input=aLoRA[k], other=bLoRA[k], alpha=sub_alpha
            )
        elif k not in bLoRA.keys():
            diff_adapter[k] = aLoRA[k]

    aLoRA = None
    bLoRA = None

    return diff_adapter


def get_applied_diff_adapter(
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
    applied_diff_model = 'airo' + (('chronos' - ('L2' * sub_alpha)) * apl_alpha)

        >>> applied_diff = get_applied_diff_adapter(
        >>>     'base.bin',
        >>>     'a.bin',
        >>>     'b.bin',
        >>>     sub_alpha=1,
        >>>     apl_alpha=1,
        >>>     device1='cpu',
        >>>     device2='cuda'
        >>>)
    """
    diff_adapter = {}

    # Load adapter using torch.load to avoid needing the base model
    baseLoRA = torch.load(base, map_location=device1)
    aLoRA = torch.load(a, map_location=device1)
    bLoRA = torch.load(b, map_location=device1)

    for k in tqdm(aLoRA.keys()):
        baseLoRA[k], aLoRA[k], bLoRA[k] = (
            baseLoRA[k].to(device2),
            aLoRA[k].to(device2),
            bLoRA[k].to(device2),
        )
        diff_adapter[k] = torch.add(
            input=aLoRA[k],
            other=torch.sub(
                input=bLoRA[k], other=baseLoRA[k], alpha=sub_alpha
            ),
            alpha=apl_alpha,
        )
        baseLoRA[k], aLoRA[k], bLoRA[k] = None, None, None

    baseLoRA, aLoRA, bLoRA = None, None, None

    gc.collect()

    return diff_adapter


def get_diff_model(a: str, b: str, sub_alpha: float = 1, device: str = "cpu"):
    """
    Compute the difference between two LlamaForCausalLM models.

    This function takes two pre-trained LlamaForCausalLM models loaded from
    file paths `a` and `b` and computes the element-wise difference between
    corresponding layers in the models.

    Args:
        a (str): File path or identifier of the first LlamaForCausalLM model.
        b (str): File path or identifier of the second LlamaForCausalLM model.
        sub_alpha (Number): The multiplier for b.
        device (str): Specifies the device (e.g., 'cpu' or 'cuda') on which
            to load the models.

    Returns:
        dict: A dictionary containing the element-wise difference between
            corresponding layers in the input LlamaForCausalLM models.

    Example:
        Suppose you have two pre-trained LlamaForCausalLM models identified
        as 'model_a' and 'model_b', and you want to compute the difference
        between them:

        >>> result = get_diff_model(
        >>>        'jondurbin/airoboros-l2-7b-2.2.1',
        >>>        'meta-llama/Llama-2-7b-hf',
        >>>        1,
        >>>        'cuda'
        >>>    )
    """
    diff_model = {}

    aModel = AutoModel.from_pretrained(
        a, load_in_8bit=False, torch_dtype=torch.float16, device_map=device
    )

    bModel = AutoModel.from_pretrained(
        b, load_in_8bit=False, torch_dtype=torch.float16, device_map=device
    )

    for k in tqdm(aModel.state_dict().keys()):
        diff_model[k] = torch.sub(
            input=aModel.state_dict()[k],
            other=bModel.state_dict()[k],
            alpha=sub_alpha,
        )

    aModel = None
    bModel = None

    return diff_model


def get_applied_diff_model(
    a: str,
    b: str,
    c: str,
    sub_alpha: float = 1,
    apl_alpha: float = 1,
    device: str = "cpu",
):
    """
    Generate a model with applied differential updates based on three
    pretrained models.

    This function takes three pretrained models (a, b, and c) and computes the
    difference between the first two models (a - b) with an optional scaling
    factor `sub_alpha`. Then, it applies the computed differential updates
    to the third model (c) with an optional scaling factor `apl_alpha`.

    Args:
        a (str): File path or identifier of the first pretrained model.
        b (str): File path or identifier of the second pretrained model.
        c (str): File path or identifier of the third pretrained model.
        sub_alpha (float, optional): Scaling factor for the difference
            between a and b. Default is 1.
        apl_alpha (float, optional): Scaling factor for applying the
            differential updates to c. Default is 1.
        device (str, optional): Specifies the device (e.g., 'cpu' or 'cuda')
            on which to load the models. Default is 'cpu'.

    Returns:
        dict: A dictionary containing the state_dict of the model with
            applied differential updates.

    Example:
        Suppose you have three pretrained models identified as 'model_a',
        'model_b', and 'model_c', and you want to generate a model with
        applied differential updates:

        >>> applied_diff = get_applied_diff_model(
        >>>     'jondurbin/airoboros-l2-7b-2.2.1',
        >>>     'meta-llama/Llama-2-7b-hf',
        >>>     'NousResearch/Nous-Hermes-llama-2-7b',
        >>>     sub_alpha=1,
        >>>     apl_alpha=1,
        >>>     device='cuda'
        >>>)
    """
    diff_model = {}
    applied_diff_model = {}

    aModel = AutoModel.from_pretrained(
        a, load_in_8bit=False, torch_dtype=torch.float16, device_map=device
    )

    bModel = AutoModel.from_pretrained(
        b, load_in_8bit=False, torch_dtype=torch.float16, device_map=device
    )

    for k in tqdm(aModel.state_dict().keys()):
        diff_model[k] = torch.sub(
            input=aModel.state_dict()[k],
            other=bModel.state_dict()[k],
            alpha=sub_alpha,
        )
        aModel.state_dict()[k], bModel.state_dict()[k] = None, None

    aModel = None
    bModel = None

    cModel = AutoModel.from_pretrained(
        c, load_in_8bit=False, torch_dtype=torch.float16, device_map=device
    )

    for k in tqdm(cModel.state_dict().keys()):
        applied_diff_model[k] = torch.mul(
            torch.add(input=cModel, other=diff_model[k]), apl_alpha
        )

    cModel = None
    diff_model = None

    return applied_diff_model


# TODO: allow setting where base, a, & b load individually
# TODO: allow setting to either do math on cpu or gpu
def diff_with_base(base, a, b, x, is_safetensors):
    cLoRA = {}
    if is_safetensors is True:
        baseLoRA = {}
        aLoRA = {}
        bLoRA = {}
        with safe_open(base, framework="pt", device="cpu") as f:
            for k in f.keys():
                baseLoRA[k] = f.get_tensor(k)
        with safe_open(a, framework="pt", device="cpu") as f:
            for k in f.keys():
                aLoRA[k] = f.get_tensor(k)
        with safe_open(b, framework="pt", device="cpu") as f:
            for k in f.keys():
                bLoRA[k] = f.get_tensor(k)
    else:
        baseLoRA = torch.load(base, map_location="cpu")
        aLoRA = torch.load(a, map_location="cpu")
        bLoRA = torch.load(b, map_location="cpu")

    for k in tqdm(baseLoRA.keys()):
        if k in aLoRA.keys() and k in bLoRA.keys():
            baseLoRA[k], aLoRA[k], bLoRA[k] = (
                baseLoRA[k].to("cuda"),
                aLoRA[k].to("cuda"),
                bLoRA[k].to("cuda"),
            )
            cLoRA[k] = torch.div(
                torch.add(
                    torch.sub(aLoRA[k], baseLoRA[k]),
                    torch.sub(bLoRA[k], baseLoRA[k]),
                ),
                x,
            )
            cLoRA[k].to("cpu")
            baseLoRA[k] = None
            aLoRA[k] = None
            bLoRA[k] = None
        elif k in aLoRA.keys():
            baseLoRA[k], aLoRA[k] = baseLoRA[k].to("cuda"), aLoRA[k].to("cuda")
            cLoRA[k] = torch.sub(aLoRA[k], baseLoRA[k])
            cLoRA[k].to("cpu")
            baseLoRA[k] = None
            aLoRA[k] = None
        elif k in bLoRA.keys():
            baseLoRA[k], bLoRA[k] = baseLoRA[k].to("cuda"), bLoRA[k].to("cuda")
            cLoRA[k] = torch.sub(bLoRA[k], baseLoRA[k])
            cLoRA[k].to("cpu")
            baseLoRA[k] = None
            bLoRA[k] = None
        else:
            baseLoRA[k] = baseLoRA[k].to("cuda")
            cLoRA[k] = torch.sub(baseLoRA[k], baseLoRA[k])
            cLoRA[k].to("cpu")
            baseLoRA[k] = None

    baseLoRA = None
    aLoRA = None
    bLoRA = None

    gc.collect()

    return cLoRA


def main(args):
    if args.mode == "adapter":
        result = get_applied_diff_adapter(
            base=args.adapter_base,
            a=args.adapter_a,
            b=args.adapter_b,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device1=args.device1,
            device2=args.device2,
        )
    elif args.mode == "model":
        result = get_applied_diff_model(
            a=args.model_a,
            b=args.model_b,
            c=args.model_c,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device=args.device,
        )
    elif args.mode == "diff_with_base":
        result = diff_with_base(
            base=args.dwb_base,
            a=args.dwb_a,
            b=args.dwb_b,
            x=args.dwb_x,
            is_safetensors=args.dwb_is_safetensors,
        )
    else:
        raise ValueError("Invalid mode. Please choose 'adapter' or 'model'.")

    save_file(result, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save applied differential updates for"
        "adapters or models."
    )
    parser.add_argument(
        "--mode",
        choices=["adapter", "model", "diff_with_base"],
        required=True,
        help="Choose 'adapter', 'model', or 'diff_with_base' mode.",
    )
    parser.add_argument(
        "--sub_alpha",
        type=float,
        default=1,
        help="Scaling factor for difference (default: 1).",
    )
    parser.add_argument(
        "--apl_alpha",
        type=float,
        default=1,
        help="Scaling factor for applying updates. 0.5 would be an average merge. 1 would add the difference ontop. 0 would 0 out the weights. (default: 0.5.).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models (e.g., 'cpu' or 'cuda'). Default is 'cpu'.",
    ),
    parser.add_argument("--device1", type=str, default="cpu"),
    parser.add_argument("--device2", type=str, default="cuda"),
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the result.",
    )
    parser.add_argument(
        "--adapter_base",
        type=str,
        help="Path to any PyTorch model which you'd like to treat as the 'base' model. This will be subtracted from adapter_b so that we hopefully don't duplicate that information during merging.",
    )
    parser.add_argument(
        "--adapter_a",
        type=str,
        help="Path to any PyTorch model which will be your main model. The difference from base and adapter_b will be applied to this model.",
    )
    parser.add_argument(
        "--adapter_b",
        type=str,
        help="Path to any PyTorch model which will be the model you are trying to get the difference of to apply onto adapter_a.",
    )
    parser.add_argument(
        "--model_a", type=str, help="Identifier or path of the first model."
    )
    parser.add_argument(
        "--model_b", type=str, help="Identifier or path of the second model."
    )
    parser.add_argument(
        "--model_c", type=str, help="Identifier or path of the third model."
    )
    parser.add_argument(
        "--dwb_base",
        type=str,
    )
    parser.add_argument(
        "--dwb_a",
        type=str,
    )
    parser.add_argument(
        "--dwb_b",
        type=str,
    )
    parser.add_argument("--dwb_x", type=float, default=2)
    parser.add_argument("--dwb_is_safetensors", type=bool, default=False)

    args = parser.parse_args()
    main(args)
