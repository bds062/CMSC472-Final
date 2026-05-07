import torch
import torch.nn as nn
import torch.nn.functional as F

# regular
# using multiclass classification for the SEED-IV (4 classes) or SEED (3 classes i think) dataset
class ClassificationLoss(nn.Module):
   def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


# contrastive

# Contrastive-Prototype


#Leave-out Contrastive Prototype
