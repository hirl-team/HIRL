import torch
import torch.nn as nn
from torchvision import models


class SimSiam(nn.Module):
    """
    Re-implementation of Simple Siamese (SimSiam)
    based on https://github.com/facebookresearch/simsiam.

    Args:
        base_arch (str): the base architecture of encoder (default: "resnet50")
        dim (int): feature dimension (default: 2048)
        pred_dim (int): hidden dimension of predictor (default: 512)
    """
    def __init__(self, base_arch="resnet50", dim=2048, pred_dim=512, **kwargs):
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = models.__dict__[base_arch](num_classes=dim, zero_init_residual=True)

        self.backbone = self.encoder

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        # hack: not use bias as it is followed by BN
        self.encoder.fc[6].bias.requires_grad = False 

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.criterion = nn.CosineSimilarity(dim=1)

        # create hooks for the embeddings before and after projection
        self.entangled_dim = prev_dim
        self.inst_dim = dim
        self._embeddings = {}
        def hook_function(module, input_emb, output_emb):
            # always unpack the input and output first
            if isinstance(input_emb, tuple) and len(input_emb) == 1:
                input_emb = input_emb[0]
            if isinstance(output_emb, tuple) and len(output_emb) == 1:
                output_emb = output_emb[0]
            self._embeddings["entangled_emb"] = input_emb
            self._embeddings["inst_emb"] = output_emb

        project_layer = dict([*self.encoder.named_modules()])["fc"]
        project_layer.register_forward_hook(hook_function)

    def get_loss(self, predictions, targets):
        losses = list()
        for pred_id, pred in enumerate(predictions):
            for target_id, target in enumerate(targets):
                if pred_id != target_id: # cross prediction
                    loss = -self.criterion(pred, target).mean()
                    losses.append(loss)
        
        # average over view
        loss = torch.stack(losses, 0).mean()

        return loss

    def forward_feature(self, images):
        if self.hook_on_predictor:
            return self.predictor(self.encoder(images))
        elif self.hook_on_predictor_hidden:
            return self.predictor[0](self.encoder(images))
        else:
            return self.encoder(images)

    def forward(self, images, return_hooks=False):
        """
        Given a list of images, compute cross-view predictive loss.

        Args:
            images (list of torch.Tensor [N,C,H,W]): image tensors. The first two 
                elements are global views.
            return_hooks (bool): whether return the hidden output by hook function. (default: False)

        Returns:
            output_dict: dict with the following fields:
                ``loss`` (torch.Tensor [1,]): loss for backward.
                ``embeddings``: dict used for further loss computation in hirl.
        """
        predictions = list()
        targets = list()
        embedding_dict = dict()
        for vid, im in enumerate(images):
            feat = self.encoder(im)
            pred = self.predictor(feat)
            if vid==0:
                for k,v in self._embeddings.items():
                    embedding_dict[k] = v
            target = self.encoder(im).detach()
            predictions.append(pred)
            targets.append(target)

        loss = self.get_loss(predictions, targets)

        output_dict = dict(loss=loss)
        if return_hooks:
            output_dict["embeddings"] = embedding_dict
        
        return output_dict

def simsiam_resnet50(**kwargs):
    return SimSiam(base_arch="resnet50", **kwargs)
