import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
# from torchvision.transforms import ToTensor
# from romatch.utils.utils import tensor_to_pil

# from model_tiny1 import  TinyRoMaExportH1 as TinyRoMaExport

from model_tiny2 import   TinyRomaV2_2  as TinyRoma

from thop import profile,clever_format

import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import sys 

weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}

def tiny_roma_v1_model_export(weights = None, freeze_xfeat=False, exact_softmax=False, xfeat = None):
    model = TinyRoMaExport(
        xfeat = xfeat,
        freeze_xfeat=freeze_xfeat, 
        exact_softmax=exact_softmax)
    if weights is not None:
        model.load_state_dict(weights, strict=False)
    return model

def tiny_roma_v1_outdoor_export(device, weights = None, xfeat = None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"],
            map_location=device)
    if xfeat is None:
        xfeat = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat', 
            pretrained = True, 
            top_k = 4096).net

    return tiny_roma_v1_model_export(weights = weights, xfeat = xfeat).to(device) 

def proc(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            print(f"name:{name} BN", module.affine)
            module.affine=True

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    args, _ = parser.parse_known_args()

    os.makedirs("onnx",exist_ok=True, mode=0o777)

    # Create model
    # roma_model = tiny_roma_v1_outdoor_export(device=device)
    roma_model = TinyRoma(freeze_xfeat=False, exact_softmax=False).to(device)
    # ckpt = "workspace/checkpoints-122016/train_ddp_tiny_roma_v1_outdoor2024352.pth"
    # roma_model.load_state_dict(torch.load(ckpt, map_location=device)['model'],strict=False)
    roma_model.eval()

    roma_model.forward = roma_model.forward_export

    height = 640
    width = 320

    x1 = torch.rand((1,3,height,width)).to(device)
    x2 = torch.rand((1,3,height,width)).to(device)

    input = (x1,x2)
    input_names=["x1","x2"]

    flops, params = profile(roma_model, input)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Flops:{flops}, Params:{params}")

    # jit_path=f"onnx/roma_tiny_{height}x{width}.pt"
    # traced_script_module = torch.jit.trace(roma_model, input)
    # traced_script_module.save(jit_path)

    onnx_path = f"onnx/roma_tiny2_{height}x{width}.onnx"
    torch.onnx.export(roma_model, input, onnx_path, input_names=input_names, output_names=["fine_matches"], opset_version=16)
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_path))



