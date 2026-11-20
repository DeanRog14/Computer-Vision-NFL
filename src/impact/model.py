"""Binary r3d_18; forward returns logits, sigmoid is applied for inference only."""
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


def build_model(pretrained=False):
    model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(False)
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), saved
