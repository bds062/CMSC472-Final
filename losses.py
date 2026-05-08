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

class ContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
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
        denominator = torch.exp(zi_DOT_za_stable).masked_fill(self_mask, 0).sum(dim=1, keepdim=True)
        numerator = torch.exp(zi_DOT_za_stable).masked_fill(~pos_mask.bool(), 0)
        log_prob = torch.log(numerator + 1e-8) - torch.log(denominator + 1e-8)

        Pi_cardinality = pos_mask.sum(dim=1)
        valid = Pi_cardinality > 0
        loss_per_anchor = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / Pi_cardinality[valid]
        loss = loss_per_anchor.mean()

        return loss

# Contrastive-Prototype

## For each class c, compute a prototype p_c by averaging the normalized embeddings of all samples in the batch with label c
## then compare each sample embedding z_i to every class prototype-- pos, neg, and neutral
## correct class prototype should have highest simmilarity

## equation:
## logits[i, c] = z_i dot p_c / tau
## loss = CrossEntropyLoss(logits, labels)

##features: tensor of shape [N, D]
## N = batch size
## D = embedding dimension
## labels: tensor of shape [N]
## class labels, should be integers from 0 to num_classes - 1

class ConstrativePrototypeLearning(nn.Module):
    def _init_(self, num_classes, temperature = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        device = features.device
        Z = F.normalize(features, dim=1)
        prototypes = []

        for c in range(self.num_classes):
            class_mask = labels == c
            if class_mask.sum() == 0:
                prototype = torch.zeros(Z.shape[1], device=device) ## 0 prototype of no features c found
            else:
                prototype = Z[class_mask].mean(dim=0)
                prototype = F.normalize(prototype, dim=0)
                
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=0)
        logits = torch.matmul(Z, prototypes.T) / self.tau
        loss = self.loss_fn(logits, labels)
        return loss
    
# Leave-out Contrastive Prototype
class LeaveOneOutContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
    
    def forward(self, features, labels):
        N = features.shape[0]
        