import torch
import torch.nn as nn
import torch.nn.functional as F

# Regular
# using multiclass classification for the SEED-IV (4 classes) or SEED (3 classes i think) dataset
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)

## L = 1/2N sum_1_N{y_i dot d_i^2 + (1-y_i) dot max(0, m - d_i)^2}
## L = I[y_i = y_j]||M(x_i)-M(x_j)||_2^2 + I[y_i =/= y_j]max(0, epsilon-||M(x_i)-M(x_j)||_2)^2
#### epsilon = max margin loss --> hyperparam

# Contrastive
## Supervised contrastic learning:
## L = sum_1toI{-1 / |P(i)| sum_pInP(i) * {log(e^(z_i dot z_p / tau)) / sum_aInA(i){e^(z_i dot z_a / tau)} }}
#### i = index anchor sample
#### P(i) all positives in batch
#### A(i) all samples in batch other than i
#### z normalized projection embeddings
#### tau = temperature param to scale embeddings
#### z_i dot z_p = calculating sim betw anchor and positive

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super.__init__()
        self.tau = temperature
    
    def forward(self, features, labels):
        N = features.shape[0]
        self_mask = torch.eye(N, dtype=torch.bool, device=features.device)
        Z = F.normalize(features, dim=1)

        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        pos_mask.masked_fill_(self_mask, 0)

        zi_DOT_za = torch.mm(Z, Z.T) / self.tau
        zi_DOT_za_stable = zi_DOT_za - zi_DOT_za.max(dim=1, keepdim=True).values.detach()
        denominator = torch.exp(zi_DOT_za_stable).masked_fill_(self_mask, 0).sum(dim=1, keepdim=True)
        numerator = torch.exp(zi_DOT_za_stable).masked_fill(~pos_mask.bool(), 0)
        log_prob = torch.log(numerator + 1e-8) - torch.log(denominator + 1e-8)

        Pi_cardinality = pos_mask.sum(dim=1)
        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / Pi_cardinality
        loss = loss_per_anchor[Pi_cardinality > 0].mean()

        return loss

# Contrastive-Prototype

# Leave-out Contrastive Prototype
