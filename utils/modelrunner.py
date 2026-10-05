import os
import sys
import torch
import importlib.util
import importlib

def modelrunner(ckpt_path):
    spec = importlib.util.spec_from_file_location(
        "MCF", os.path.join("models", "mcf.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    opt = checkpoint["opt"]

    opt.device = device
    
    safe_viz_dir = os.path.join("artifacts", "viz")
    
    model = module.MCF.load_from_checkpoint(
        ckpt_path, 
        map_location=device,
        viz_dir=safe_viz_dir
    )

    model.eval()

    return model, opt