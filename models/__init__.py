# models/__init__.py
import os
import torch
from .stead_model import STEAD, build_stead_from_cfg
from .autoencoder_model import FeatureAutoencoder


def build_model(cfg, device, DEBUG=False):
    """
    Build STEAD or Autoencoder model from config, without loading checkpoint.
    """
    model_type = cfg["model"]["type"].lower()

    if model_type == "stead":
        stead_cfg = cfg["model"]["stead"]
        model = STEAD(
            feature_dim=stead_cfg["feature_dim"],
            hidden_dim=stead_cfg["hidden_dim"],
            seq_len=stead_cfg["seq_len"],
            num_heads=stead_cfg["num_heads"]
        ).to(device)

    elif model_type == "autoencoder":
        ae_cfg = cfg["model"]["autoencoder"]
        model = FeatureAutoencoder(
            input_dim=ae_cfg.get("input_dim", 2048),
            hidden_dim=ae_cfg.get("hidden_dim", 512)
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if DEBUG:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[DEBUG] Built {model_type.upper()} model with {total_params} parameters on {device}")

    return model


def load_model(cfg, checkpoint, device, DEBUG=False):
    """
    Build model and safely load checkpoint.
    """
    model = build_model(cfg, device, DEBUG)

    # ------------------------------
    # Safe checkpoint loading
    # ------------------------------
    if isinstance(checkpoint, str) and checkpoint.strip() != "":
        if os.path.exists(checkpoint):
            if DEBUG:
                print(f"[DEBUG] Loading checkpoint from: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            if DEBUG:
                print("[DEBUG] Checkpoint loaded successfully")
        else:
            if DEBUG:
                print(f"[WARNING] Checkpoint file does not exist: {checkpoint}")
    else:
        if checkpoint is not None and DEBUG:
            print(f"[WARNING] Invalid checkpoint provided (ignored): {checkpoint}")

    model.eval()
    return model
