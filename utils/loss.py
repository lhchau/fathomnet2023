import torch.nn.functional as F 

class BCEL2Loss(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(CustomLoss, self).__init__()
        self.weight_decay = weight_decay
        
    def forward(self, inputs, targets):
        l2_regularization = 0.0
        for param in self.parameters():
            l2_regularization += torch.norm(param, 2)
        loss = F.BCEWithLogitsLoss(inputs, targets)
        loss += self.weight_decay * l2_regularization
        return loss