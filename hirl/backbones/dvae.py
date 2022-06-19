"""
Modified based on https://github.com/microsoft/unilm/blob/master/beit/modeling_discrete_vae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteVAE(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir, device):
        try:
            import dall_e
        except ModuleNotFoundError:
            raise ModuleNotFoundError("DALL-E is not found. Please install it with `pip install DALL-E`")

        from dall_e import load_model
        self.encoder = load_model(os.path.join(model_dir, "encoder.pkl"), device)
        self.decoder = load_model(os.path.join(model_dir, "decoder.pkl"), device)

    def decode(self, img_seq):
        batch_size = img_seq.size()[0]
        img_seq = img_seq.view(batch_size, self.image_size // 8, self.image_size // 8)
        z = F.one_hot(img_seq, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return F.softmax(z_logits, dim=1)

    def forward(self, img_seq_prob, no_process=False):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            batch_size, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(batch_size, self.image_size // 8, self.image_size // 8, self.encoder.vocab_size)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()
