import torch
import torch.nn as nn
import torch.nn.functional as F

from hirl import layers, models

class HIRL(nn.Module):
    """
    The framework of Hierarchical Image Representation Learning (HIRL), 
    contributed by ChrisAllenMing.

    Args:
        model (nn.Module): the image representation model.
        num_cluster (list[int]): the number of clusters in each semantic hierarchy.
        num_neg_path (int, optional): negative path sampling size (default: 1000).
        train_emb (str, optional): the image embedding to train on
            (choices: ["cnn", "vit_cls", "vit_mean"], default: "cnn").
        num_mlp_layer (int, optional): the number of hidden layers for each projection head (default: 2).
        activation (str or function, optional): activation function for projection heads (default: "relu").
        batch_norm (bool, optional): whether to use batch normalization in projection heads (default: True).
    """

    emb_keys = {"cnn": "inst_emb", "vit_cls": "inst_cls_emb", "vit_mean": "inst_mean_emb"}
    eps = 1e-6

    def __init__(self, model, num_cluster, num_neg_path=1000, train_emb="cnn", 
                    num_mlp_layer=2, activation="relu", batch_norm=True, **kwargs):
        super().__init__()

        self.model = model
        self.num_cluster = num_cluster
        self.num_hierarchy = len(num_cluster)
        self.num_neg_path = num_neg_path
        self.emb_key = self.emb_keys[train_emb]

        self.input_dim = model.inst_dim
        hidden_dims = [self.input_dim] * self.num_hierarchy
        self.projections = nn.ModuleList()
        for i in range(self.num_hierarchy):
            projection = layers.MLP(self.input_dim, [hidden_dims[i]] * num_mlp_layer, activation=activation, batch_norm=batch_norm)
            self.projections.append(projection)

    def sample_neg_path(self, index, cluster_result):
        im2cluster = cluster_result["im2cluster"]
        centroids = cluster_result["centroids"]
        cluster2cluster = cluster_result["cluster2cluster"]

        # sample negative paths
        neg_im2cluster = []
        for hierarchy_id in range(self.num_hierarchy):
            if hierarchy_id == 0:
                im2cluster_ = torch.multinomial(torch.ones(centroids[0].shape[0], device=centroids[0].device), 
                    len(index) * self.num_neg_path, replacement=True).view(len(index), -1)
                neg_im2cluster.append(im2cluster_)
            else:
                im2cluster_ = cluster2cluster[hierarchy_id - 1][im2cluster_.view(-1)].view(len(index), -1)
                neg_im2cluster.append(im2cluster_)
        neg_im2cluster = torch.stack(neg_im2cluster, dim=-1)  # [N, N_neg, L]

        # get true negatives
        pos_im2cluster = torch.stack([im2cluster[hid][index] for hid in range(self.num_hierarchy)], dim=-1)  # [N, L]
        true_negative = torch.any(neg_im2cluster != pos_im2cluster.unsqueeze(1), dim=-1).to(torch.long)  # [N, N_neg]
        
        return neg_im2cluster, true_negative

    def semantic_path_discrimination_loss(self, semantic_embs, index, cluster_result, neg_im2cluster, true_negative):
        centroids = cluster_result["centroids"]
        im2cluster = cluster_result["im2cluster"]

        # BCE loss for positive paths
        batch_size = semantic_embs.shape[0]
        semantic_embs = semantic_embs.view(batch_size, self.num_hierarchy, -1)  # [N, L, D]
        semantic_embs = F.normalize(semantic_embs, dim=-1)
        pos_path = torch.stack([F.normalize(centroids[h_id][cluster_label[index]], dim=-1) \
                            for (h_id, cluster_label) in enumerate(im2cluster)], dim=1)  # [N, L, D]
        pos_path_sim = torch.prod((((semantic_embs * pos_path).sum(-1) + 1) / 2), dim=1)  # [N,]
        semantic_loss = -torch.log(pos_path_sim + self.eps) 

        # BCE loss for negative paths
        neg_paths = [F.normalize(centroids[h_id][neg_im2cluster[:, :, h_id].view(-1)].view(batch_size, self.num_neg_path, -1), dim=-1) \
                            for h_id in range(self.num_hierarchy)]
        neg_paths = torch.stack(neg_paths, -2)  # [N, N_neg, L, D]
        neg_prototype_sim = (semantic_embs.unsqueeze(1).repeat(1, self.num_neg_path, 1, 1) * neg_paths).sum(-1)  # [N, N_neg, L]
        neg_path_sim = torch.prod((neg_prototype_sim + 1) / 2, dim=-1)  # [N, N_neg]
        neg_path_sim_loss = -torch.log(1 - neg_path_sim + self.eps)
        neg_path_sim_loss = (neg_path_sim_loss * true_negative).sum(-1) / (true_negative.sum(-1) + self.eps)  # [N,]
        
        semantic_loss = torch.stack([semantic_loss, neg_path_sim_loss], dim=-1).mean()
        return semantic_loss

    def forward_feature(self, images):
        return self.model.forward_feature(images)

    def forward(self, images, index, cluster_result, **kwargs):
        """
        Args:
            images (torch.tensor or list[torch.tensor]): a batch of input images.
            index (torch.tensor): the indices of training samples.
            cluster_result (dict): cluster assignments, centroids and density. 

        Returns:
            output_dict (dict): all outputs (e.g., losses and embeddings).
        """        
        output_dict = self.model(images, return_hooks=True, **kwargs)
        image_loss, embedding_dict = output_dict["loss"], output_dict["embeddings"]
        loss = image_loss
        output_dict["image_loss"] = image_loss.item()

        # Apply semantic modeling loss in the second training stage
        if cluster_result is not None:
            all_img_emb = embedding_dict[self.emb_key]
            img_emb = all_img_emb[0] if isinstance(all_img_emb, list) else all_img_emb
            semantic_embs = []
            for hierarchy_id in range(self.num_hierarchy):
                semantic_emb = self.projections[hierarchy_id](img_emb)
                semantic_emb = F.normalize(semantic_emb, dim=-1)
                semantic_embs.append(semantic_emb)

            semantic_embs = torch.cat(semantic_embs, dim=-1)
            neg_im2cluster, true_negative = self.sample_neg_path(index, cluster_result)
            semantic_loss = self.semantic_path_discrimination_loss(semantic_embs, index, cluster_result, 
                                                                    neg_im2cluster, true_negative)
            loss += semantic_loss
            output_dict["semantic_loss"] = semantic_loss.item()

        output_dict["loss"] = loss
        return output_dict
