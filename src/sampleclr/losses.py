import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCECosine(nn.Module):
    def __init__(self, temperature=0.5, reg_coef=0, reg_radius=200):
        super(InfoNCECosine, self).__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features, *args, **kwargs):
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)
        a = F.normalize(a)
        b = F.normalize(b)
        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature
        tempered_alignment = cos_ab.trace() / batch_size
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2
        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature=1, exaggeration=1):
        super(InfoNCECauchy, self).__init__()
        self.temperature = temperature
        self.exaggeration = exaggeration

    def forward(self, features, *args, **kwargs):
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]
        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)
        tempered_alignment = torch.diagonal(sim_ab).log().mean()
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()
        raw_uniformity = logsumexp_1 + logsumexp_2
        loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features, *args, **kwargs):
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]
        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()
        tempered_alignment = sim_ab.trace() / batch_size
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2
        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss

def ordinal_regression_loss(probas, target):
    """
    Computes the ordinal regression loss.

    Args:
        probas (Tensor): Probabilities from the ordinal regression head (batch_size x num_classes - 1).
        target (Tensor): True ordinal labels (batch_size), values from 0 to num_classes - 1.

    Returns:
        Tensor: Loss value.
    """
    device = probas.device
    batch_size = target.size(0)
    num_classes = probas.size(1) + 1

    y = torch.zeros((batch_size, num_classes - 1), device=device)
    for k in range(num_classes - 1):
        y[:, k] = (target > k).float()

    loss = F.binary_cross_entropy(probas, y, reduction='mean')
    return loss


class InfoNCECauchyBatchAware(nn.Module):
    def __init__(self, temperature=1, exaggeration=1, batch_weight=0.5):
        """
        Args:
            temperature: scaling factor for the distances.
            exaggeration: scaling factor for the positive alignment.
            batch_weight: weight for negative pairs from the same batch.
                          Should be between 0 and 1 (e.g. 0.5 means same-batch negatives count half as much).
        """
        super(InfoNCECauchyBatchAware, self).__init__()
        self.temperature = temperature
        self.exaggeration = exaggeration
        self.batch_weight = batch_weight

    def forward(self, features, batch_labels=None, *args, **kwargs):
        """
        Args:
            features: Tensor of shape (2*B, F) where the first B rows and the last B rows 
                      are two different augmentations/views of B samples.
            batch_labels: (optional) Tensor of shape (B,) holding the batch labels for each sample.
                          If provided, negative similarities between samples sharing the same batch
                          will be downweighted by self.batch_weight.
        """
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]

        # Compute pairwise similarities (using a Cauchy kernel)
        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        if batch_labels is not None:
            # Assume batch_labels is a tensor of shape (B,)
            # Create a weight matrix for same-batch negatives
            # For sim_aa and sim_bb (within-view negatives)
            W = torch.where(batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0),
                            self.batch_weight, 1.0)
            sim_aa = sim_aa * W
            sim_bb = sim_bb * W
            # For cross-view negatives in sim_ab
            W_ab = torch.where(batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0),
                               self.batch_weight, 1.0)
            sim_ab = sim_ab * W_ab

        # Positive alignment: take diagonal of sim_ab (each sample with its augmentation)
        tempered_alignment = torch.diagonal(sim_ab).log().mean()

        # For uniformity terms, mask out the self-similarities in sim_aa and sim_bb.
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return loss


class XSampleCLR(nn.Module):
    """
    Expert-style InfoNCE with:
      - AB: targets = softmax(G / T)  (diagonal kept; self allowed)
      - AA/BB: targets = softmax((G with diag=-inf) / T)  (self forced to 0)

    Only the target construction differs from the original expert loss.
    """
    def __init__(self, similarity_graph=None, temperature=0.5, graph_temperature=1.0, diag_mask_value=-1e9):
        super().__init__()
        self.temperature = float(temperature)
        self.graph_temperature = float(graph_temperature)
        self.similarity_graph = similarity_graph
        self.diag_mask_value = float(diag_mask_value)  

    def forward(self, features, sample_ids=None, *args, **kwargs):
        batch_size = features.shape[0] // 2

        a = F.normalize(features[:batch_size], dim=1)
        b = F.normalize(features[batch_size:], dim=1)

        if self.similarity_graph is not None and sample_ids is not None:
            if hasattr(self.similarity_graph, "loc"): 
                g_np = self.similarity_graph.loc[sample_ids, sample_ids].values
            else:
                g_np = self.similarity_graph
            G = torch.as_tensor(g_np, device=features.device, dtype=features.dtype)
            G = F.softmax(G / self.graph_temperature, dim=1)
        else:
            G = torch.eye(batch_size, device=features.device, dtype=features.dtype)

        cos_ab = ((a @ b.t()) / self.temperature).exp()
        cos_aa = ((a @ a.t()) / self.temperature).exp()
        cos_bb = ((b @ b.t()) / self.temperature).exp()

        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa = cos_aa.masked_fill(self_mask, 0)
        cos_bb = cos_bb.masked_fill(self_mask, 0)
        sims_stacked_1 = torch.vstack((cos_ab.T, cos_bb))
        sims_stacked_2 = torch.vstack((cos_aa, cos_ab))
        
        sims_stacked_1 = sims_stacked_1 / sims_stacked_1.sum(dim=0)
        sims_stacked_2 = sims_stacked_2 / sims_stacked_2.sum(dim=0)

        # Model has to put the highest weight to the positive pair. Self-similarities are masked out.
        logsumexp_1 = F.binary_cross_entropy(sims_stacked_1, torch.vstack((G, G.masked_fill(self_mask, 0))))
        logsumexp_2 = F.binary_cross_entropy(sims_stacked_2, torch.vstack((G.masked_fill(self_mask, 0), G)))

        loss = (logsumexp_1 + logsumexp_2) / 2

        return loss

