"""
Modified based on https://github.com/microsoft/unilm/blob/master/beit/modeling_discrete_vae.py
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteVAE(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.image_size = image_size

    def load_model(self):
        try:
            import dall_e
        except ModuleNotFoundError:
            raise ModuleNotFoundError("DALL-E is not found. Please install it with `pip install DALL-E`")

        from dall_e import load_model
        print("Loading DVAE ... (This may be slow for downloading models)")
        self.encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", torch.device("cuda"))
        print("DVAE loaded!")

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return F.softmax(z_logits, dim=1)
