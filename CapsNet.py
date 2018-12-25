# *_*coding:utf-8 *_*
import torch
import numpy as np
from torch.autograd import Variable

# Conv1
class Conv1(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9):
        super(Conv1, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

# Primary Capsules
class PrimaryCapsules(torch.nn.Module):
    def __init__(self, input_shape=(256, 20, 20), capsule_dim=8,
                 out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.input_shape = input_shape
        self.capsule_dim = capsule_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = self.input_shape[0]

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels * self.capsule_dim,
            self.kernel_size,
            self.stride
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, x.size()[1], x.size()[2], self.out_channels, self.capsule_dim)
        return x

# Routing
class Routing(torch.nn.Module):
    def __init__(self, caps_dim_before=8, caps_dim_after=16,
                 n_capsules_before=(6 * 6 * 32), n_capsules_after=10):
        super(Routing, self).__init__()
        self.n_capsules_before = n_capsules_before
        self.n_capsules_after = n_capsules_after
        self.caps_dim_before = caps_dim_before
        self.caps_dim_after = caps_dim_after

        # Parameter initialization not specified in the paper
        n_in = self.n_capsules_before * self.caps_dim_before
        variance = 2 / (n_in)
        std = np.sqrt(variance)
        self.W = torch.nn.Parameter(
            torch.randn(
                self.n_capsules_before,
                self.n_capsules_after,
                self.caps_dim_after,
                self.caps_dim_before) * std,
            requires_grad=True)

    # Equation (1)
    @staticmethod
    def squash(s):
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm2 = torch.pow(s_norm, 2)
        v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

    # Equation (2)
    def affine(self, x):
        x = self.W @ x.unsqueeze(2).expand(-1, -1, 10, -1).unsqueeze(-1)
        return x.squeeze()

    # Equation (3)
    @staticmethod
    def softmax(x, dim=-1):
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim, keepdim=True)

    # Procedure 1 - Routing algorithm.
    def routing(self, u, r, l):
        b = Variable(torch.zeros(u.size()[0], l[0], l[1]), requires_grad=False).cuda()  # torch.Size([?, 1152, 10])

        for iteration in range(r):
            c = Routing.softmax(b)  # torch.Size([?, 1152, 10])
            s = (c.unsqueeze(-1).expand(-1, -1, -1, u.size()[-1]) * u).sum(1)  # torch.Size([?, 1152, 16])
            v = Routing.squash(s)  # torch.Size([?, 10, 16])
            b += (u * v.unsqueeze(1).expand(-1, l[0], -1, -1)).sum(-1)
        return v

    def forward(self, x, n_routing_iter):
        x = x.view((-1, self.n_capsules_before, self.caps_dim_before))
        x = self.affine(x)  # torch.Size([?, 1152, 10, 16])
        x = self.routing(x, n_routing_iter, (self.n_capsules_before, self.n_capsules_after))
        return x

# Norm
class Norm(torch.nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        x = torch.norm(x, p=2, dim=-1)
        return x

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, in_features, out_features, output_size):
        super(Decoder, self).__init__()
        self.decoder = self.assemble_decoder(in_features, out_features)
        self.output_size = output_size

    def assemble_decoder(self, in_features, out_features):
        HIDDEN_LAYER_FEATURES = [512, 1024]
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, HIDDEN_LAYER_FEATURES[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[0], HIDDEN_LAYER_FEATURES[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[1], out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = x[np.arange(0, x.size()[0]), y.cpu().data.numpy(), :].cuda()
        x = self.decoder(x)
        x = x.view(*((-1,) + self.output_size))
        return x

# CapsNet
class CapsNet(torch.nn.Module):
    def __init__(self, input_shape, n_routing_iter=3, use_reconstruction=True):
        super(CapsNet, self).__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.n_routing_iter = n_routing_iter
        self.use_reconstruction = use_reconstruction

        self.conv1 = Conv1(input_shape[0], 256, 9)
        self.primary_capsules = PrimaryCapsules(
            input_shape=(256, 20, 20),
            capsule_dim=8,
            out_channels=32,
            kernel_size=9,
            stride=2
        )
        self.routing = Routing(
            caps_dim_before=8,
            caps_dim_after=16,
            n_capsules_before=6 * 6 * 32,
            n_capsules_after=10
        )
        self.norm = Norm()

        if (self.use_reconstruction):
            self.decoder = Decoder(16, int(np.prod(input_shape)),input_shape)

    def n_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def forward(self, x, y=None):
        conv1 = self.conv1(x)
        primary_capsules = self.primary_capsules(conv1)
        digit_caps = self.routing(primary_capsules, self.n_routing_iter)
        scores = self.norm(digit_caps)

        if (self.use_reconstruction and y is not None):
            reconstruction = self.decoder(digit_caps, y).view((-1,) + self.input_shape)
            return scores, reconstruction

        return scores

# Define Loss Functions
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

# Margin Loss
class MarginLoss(torch.nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lamb=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lamb = lamb

    # Equation (4)
    def forward(self, scores, y):
        y = Variable(to_categorical(y, 10))

        Tc = y.float()
        loss_pos = torch.pow(torch.clamp(self.m_pos - scores, min=0), 2)
        loss_neg = torch.pow(torch.clamp(scores - self.m_neg, min=0), 2)
        loss = Tc * loss_pos + self.lamb * (1 - Tc) * loss_neg
        loss = loss.sum(-1)
        return loss.mean()

# Reconstruction Loss
class SumSquaredDifferencesLoss(torch.nn.Module):
    def __init__(self):
        super(SumSquaredDifferencesLoss, self).__init__()

    def forward(self, x_reconstruction, x):
        loss = torch.pow(x - x_reconstruction, 2).sum(-1).sum(-1)
        return loss.mean()

# Total Loss
class CapsNetLoss(torch.nn.Module):
    def __init__(self, reconstruction_loss_scale=0.0005):
        super(CapsNetLoss, self).__init__()
        self.digit_existance_criterion = MarginLoss()
        self.digit_reconstruction_criterion = SumSquaredDifferencesLoss()
        self.reconstruction_loss_scale = reconstruction_loss_scale

    def forward(self, x, y, x_reconstruction, y_pred):
        margin_loss = self.digit_existance_criterion(y_pred.cuda(), y)
        reconstruction_loss = self.reconstruction_loss_scale * \
                              self.digit_reconstruction_criterion(x_reconstruction, x)
        loss = margin_loss + reconstruction_loss
        return loss, margin_loss, reconstruction_loss

# Optimizer
def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer