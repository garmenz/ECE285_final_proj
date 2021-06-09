import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemeted')
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
    
    def get_target_tensor(self, pred, real):
        '''
        form the target in the shape of the pred

        pred: the prediction output from a discriminator
        real: whether the ground truth label is for real images or fake images
        '''
        if real == True:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)
    
    def __call__(self, pred, real):
        '''
        pred: the prediction output from a discriminator
        real: whether the ground truth label is for real images or fake images
        '''
        if self.gan_mode in ['lsgan', 'vanilla']:
            target = self.get_target_tensor(pred, real)
            loss = self.loss(pred, target)
        elif self.gan_mode == 'wgangp':
            if real:
                loss = -pred.mean()
            else:
                loss = pred.mean()
        return loss

# class GAN(nn.Module):
#     def __init__(self, G, D, real_A, real_B):
#         super(GAN, self).__init__()
#         self.G = G
#         self.D = D
#         self.real_A = real_A
#         self.real_B = real_B
#         self.fake_B = None
#         self.GANLoss = GANLoss()
#         self.L1Loss = nn.L1Loss()

#     def forward():
#         self.fake_B = self.G(self.real_A)

#     def G_loss():

