from pathlib import Path
import torch

def load_state_dict_maybe_wrapped(obj, path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        obj.load_state_dict(state["model_state_dict"])
    else:
        obj.load_state_dict(state)

def save_ckpt(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
