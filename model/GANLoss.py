import torch
import torch.nn as nn
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss1, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, pred, real):
        '''
        pred: the prediction output from a discriminator
        real: whether the ground truth label is for real images or fake images
        '''
        if real:
            real_label = torch.ones(pred.shape).cuda()
            loss = self.loss(pred, real_label)
        else:
            feak_label = torch.zeros(pred.shape).cuda()
            loss = self.loss(pred, feak_label)
        return loss

