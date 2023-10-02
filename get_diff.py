# Version: 0.04
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
        a: str,
        b: str,
        sub_alpha: float = 1,
        device: str = "cpu"
):
    """
    Compute the difference between two PyTorch dictionaries.

    This function takes two file paths `a` and `b`, loads PyTorch dictionaries
    from these files, and computes the element-wise difference between
    corresponding keys in the dictionaries.

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
                input=aLoRA[k],
                other=bLoRA[k],
                alpha=sub_alpha
            )
        elif k not in bLoRA.keys():
            diff_adapter[k] = aLoRA[k]

    aLoRA = None
    bLoRA = None

    return diff_adapter


def get_applied_diff_adapter(
        a: str,
        b: str,
        c: str,
        sub_alpha: float = 1,
        apl_alpha: float = 1,
        device: str = "cpu"
):
    """
    Generate a model with applied adapter differential updates based on three
    adapter files.

    This function takes three adapter files (a, b, and c) and computes the
    difference between the first two adapter files (a - b) with an optional
    scaling factor `sub_alpha`. Then, it applies the computed adapter
    differential updates to the third adapter file (c) with an optional
    scaling factor `apl_alpha`.

    Args:
        a (str): File path or identifier of the first adapter file.
        b (str): File path or identifier of the second adapter file.
        c (str): File path or identifier of the third adapter file.
        sub_alpha (float, optional): Scaling factor for the difference
            between a and b. Default is 1.
        apl_alpha (float, optional): Scaling factor for applying the adapter
            differential updates to c. Default is 1.
        device (str, optional): Specifies the device (e.g., 'cpu' or 'cuda')
            on which to load the adapters. Default is 'cpu'.

    Returns:
        dict: A dictionary containing the adapter state_dict of the model
            with applied adapter differential updates.

    Example:
        Suppose you have three adapter files identified as 'adapter_a',
        'adapter_b', and 'adapter_c', and you want to generate a model
        with applied adapter differential updates:

        >>> applied_diff = get_applied_diff_adapter(
        >>>     'a.bin',
        >>>     'b.bin',
        >>>     'c.bin',
        >>>     sub_alpha=1,
        >>>     apl_alpha=1,
        >>>     device='cuda'
        >>>)
    """
    diff_adapter = {}
    applied_diff_model = {}

    # Load adapter using torch.load to avoid needing the base model
    aLoRA = torch.load(a, map_location=device)
    bLoRA = torch.load(b, map_location=device)

    for k in tqdm(aLoRA.keys()):
        diff_adapter[k] = torch.sub(
            input=aLoRA[k],
            other=bLoRA[k],
            alpha=sub_alpha
        )

    aLoRA = None
    bLoRA = None
    cLoRA = torch.load(c, map_location=device)

    for k in tqdm(cLoRA.keys()):
        applied_diff_model[k] = torch.add(
            input=cLoRA[k],
            other=torch.mul(
                diff_adapter[k],
                apl_alpha
            ),
            alpha=sub_alpha
        )

    cLoRA = None
    diff_adapter = None

    return applied_diff_model


def get_diff_model(
        a: str,
        b: str,
        sub_alpha: float = 1,
        device: str = "cpu"
):
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
        a,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device
    )

    bModel = AutoModel.from_pretrained(
        b,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device
    )

    for k in tqdm(aModel.state_dict().keys()):
        diff_model[k] = torch.sub(
            input=aModel.state_dict()[k],
            other=bModel.state_dict()[k],
            alpha=sub_alpha
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
        device: str = "cpu"
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
        a,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device
    )

    bModel = AutoModel.from_pretrained(
        b,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device
    )

    for k in tqdm(aModel.state_dict().keys()):
        diff_model[k] = torch.sub(
            input=aModel.state_dict()[k],
            other=bModel.state_dict()[k],
            alpha=sub_alpha
        )

    aModel = None
    bModel = None

    cModel = AutoModel.from_pretrained(
        c,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device
    )

    for k in tqdm(cModel.state_dict().keys()):
        applied_diff_model[k] = torch.add(
            input=cModel,
            other=torch.mul(
                diff_model[k],
                apl_alpha
            )
        )

    cModel = None
    diff_model = None

    return applied_diff_model


def diff_with_base(base, a, b, x=1, is_safetensors=True):
    cLoRA = {}
    if is_safetensors is True:
        base = {}
        aLoRA = {}
        bLoRA = {}
        with safe_open(base, framework="pt", device="cpu") as fbase:
            for kbase in fbase.keys():
                base[kbase] = fbase.get_tensor(kbase)
        with safe_open(a, framework="pt", device="cpu") as faLoRA:
            for kaLoRA in faLoRA.keys():
                aLoRA[kaLoRA] = faLoRA.get_tensor(kaLoRA)
        with safe_open(b, framework="pt", device="cpu") as fbLoRA:
            for kbLoRA in fbLoRA.keys():
                bLoRA[kbLoRA] = fbLoRA.get_tensor(kbLoRA)
    else:
        base = torch.load(base, map_location="cpu")
        aLoRA = torch.load(a, map_location="cpu")
        bLoRA = torch.load(b, map_location="cpu")

    for k in tqdm(base.keys()):
        if k in aLoRA.keys() and k in bLoRA.keys():
            cLoRA[k] = (
                    torch.sub(aLoRA[k], base[k])
                    +
                    torch.sub(bLoRA[k], base[k])
            ) / x
            base[k] = None
            aLoRA[k] = None
            bLoRA[k] = None
        elif k in aLoRA.keys():
            cLoRA[k] = torch.sub(aLoRA[k], base[k])
            base[k] = None
            aLoRA[k] = None
        elif k in bLoRA.keys():
            cLoRA[k] = torch.sub(bLoRA[k], base[k])
            base[k] = None
            bLoRA[k] = None
        else:
            cLoRA[k] = torch.sub(base[k], base[k])
            base[k] = None

    base = None
    aLoRA = None
    bLoRA = None

    gc.collect()

    return cLoRA


def main(args):
    if args.mode == "adapter":
        result = get_applied_diff_adapter(
            a=args.adapter_a,
            b=args.adapter_b,
            c=args.adapter_c,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device=args.device
        )
    elif args.mode == "model":
        result = get_applied_diff_model(
            a=args.model_a,
            b=args.model_b,
            c=args.model_c,
            sub_alpha=args.sub_alpha,
            apl_alpha=args.apl_alpha,
            device=args.device
        )
    elif args.mode == "diff_with_base":
        result = diff_with_base(
            base=args.dwb_base,
            a=args.dwb_a,
            b=args.dwb_b,
            x=args.dwb_x
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
        help="Choose 'adapter', 'model', or 'diff_with_base' mode.")
    parser.add_argument(
        "--sub_alpha",
        type=float,
        default=1,
        help="Scaling factor for difference (default: 1)."
    )
    parser.add_argument(
        "--apl_alpha",
        type=float,
        default=1,
        help="Scaling factor for applying updates (default: 1)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models (e.g., 'cpu' or 'cuda'). Default is 'cpu'."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the result."
    )
    parser.add_argument(
        "--adapter_a",
        type=str,
        help="Path to the first adapter file."
    )
    parser.add_argument(
        "--adapter_b",
        type=str,
        help="Path to the second adapter file."
    )
    parser.add_argument(
        "--adapter_c",
        type=str,
        help="Path to the third adapter file."
    )
    parser.add_argument(
        "--model_a",
        type=str,
        help="Identifier or path of the first model."
    )
    parser.add_argument(
        "--model_b",
        type=str,
        help="Identifier or path of the second model."
    )
    parser.add_argument(
        "--model_c",
        type=str,
        help="Identifier or path of the third model."
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
    parser.add_argument(
        "--dwb_c",
        type=str,
    )

    args = parser.parse_args()
    main(args)
