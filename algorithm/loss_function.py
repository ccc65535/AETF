import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import fft
import scipy.signal as signal


def fft_analyse(sig):
    trans = fft(sig)
    yf = abs(trans)  # 取绝对值
    # yf1=np.real(trans1)
    yf = yf / len(sig)  # 归一化处理

    return yf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
        
class diff_avg_y(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        # return  torch.mean(torch.pow((out - y), 2))

        batch = y.size(0)
        mean_y = y.mean(1).reshape(batch, 1, -1)
        std_y = y.std(1).reshape(batch, 1, -1)

        loss = torch.mean(torch.abs(out - y) / (mean_y.abs() + 1))

        return loss


class diff_norm_time(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        # return  torch.mean(torch.pow((out - y), 2))

        batch = y.size(0)
        mean_y = y.mean(1).reshape(batch, 1, -1)
        std_y = y.std(1).reshape(batch, 1, -1)

        y = (y - mean_y) / std_y
        loss_m = torch.mean(torch.abs(out - y))

        return loss_m, mean_y, std_y


class diff_abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        return torch.mean(torch.abs(out - y))


class diff_pow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        return torch.mean(torch.pow(out - y, 2))


class diff_mean_fft(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, out, y):
        device = out.device
        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        diff1 = torch.mean(torch.pow(out - y, 2))
        diff2 = (fft_analyse(out_) - fft_analyse(y_))

        diff2 = torch.Tensor(diff2)
        diff2 = torch.mean(torch.pow(diff2, 2))

        # a=0.5
        # b=0.1

        return self.a * diff1 + self.b * diff2


class diff_stft(nn.Module):
    def __init__(self, srate, win_size):
        super().__init__()
        self.srate = srate
        self.win = win_size

    def forward(self, out, y):
        device = out.device
        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        f, t, s1_ = signal.stft(out_, self.srate, nperseg=self.win)
        f, t, s2_ = signal.stft(y_, self.srate, nperseg=self.win)

        s1 = torch.Tensor(s1_.real)
        s2 = torch.Tensor(s2_.real)

        diff1 = torch.mean(torch.pow(out - y, 2))
        diff2 = torch.sum(torch.pow(s1 - s2, 2))
        # print('stft diff:',diff2)

        return diff1 + diff2


class diff_stft2(nn.Module):
    def __init__(self, srate, win_size):
        super().__init__()
        self.srate = srate
        self.win = win_size

    def forward(self, out, y):
        device = out.device
        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        f, t, s1_ = signal.stft(out_, self.srate, nperseg=self.win)
        f, t, s2_ = signal.stft(y_, self.srate, nperseg=self.win)

        # s1=torch.Tensor(s1_.real)
        # s2=torch.Tensor(s2_.real)
        s1 = torch.Tensor(s1_.__abs__())
        s2 = torch.Tensor(s2_.__abs__())

        diff1 = torch.mean(torch.pow(out - y, 2))
        diff2 = torch.mean(torch.pow(s1 - s2, 2))
        # print('stft diff:',diff2)

        return diff1 + diff2


class diff_stft3(nn.Module):
    def __init__(self, srate, win_size, a=1, b=10):
        super().__init__()
        self.srate = srate
        self.win = win_size
        self.a = a
        self.b = b

    def forward(self, out, y):
        device = out.device
        ch_num = out.shape[2]

        out_ = out.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        stft1 = []
        stft2 = []

        for ch in range(ch_num):
            f, t, s1_ = signal.stft(out_[:, :, ch], self.srate, nperseg=self.win)
            f, t, s2_ = signal.stft(y_[:, :, ch], self.srate, nperseg=self.win)
            stft1.append(s1_.__abs__())
            stft2.append(s2_.__abs__())

        stft1 = torch.Tensor(np.array(stft1))
        stft2 = torch.Tensor(np.array(stft2))

        diff1 = torch.pow(out - y, 2)
        diff2 = torch.pow(stft1 - stft2, 2)

        # diff1 = torch.mean(diff1,axis=(1))
        # diff2 = torch.mean(diff2,axis=(1,2))

        # diff1 = torch.mean(diff1)
        # diff2 = torch.sum(diff2)
        # print(f'diff1:{diff1},diff2:{diff2}')

        diff1 = torch.mean(diff1)
        diff2 = torch.mean(diff2)

        return self.a * diff1 + self.b * diff2


class pearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        # output=output.detach().cpu()
        # y=y.detach().cpu()

        n_batch, n_sample, n_block = y.size()

        output_mean = torch.mean(output, dim=1)
        y_mean = torch.mean(y, dim=1)
        sumTop = 0.0
        sumBottom = 0.0
        output_pow = 0.0
        y_pow = 0.0
        for i in range(n_sample):
            sumTop += (output[:, i, :] - output_mean) * (y[:, i, :] - y_mean)
        for i in range(n_sample):
            output_pow += torch.pow(output[:, i, :] - output_mean, 2)
        for i in range(n_sample):
            y_pow += torch.pow(y[:, i, :] - y_mean, 2)
        sumBottom = torch.sqrt(output_pow * y_pow)
        loss = -(sumTop / sumBottom).sum()
        return loss


class corrLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_, y_):
        # output=output.detach().cpu()
        # y=y.detach().cpu()

        y = y_.reshape(-1, y_.size(-1))
        output = output_.reshape(-1, output_.size(-1))

        n_batch, n_sample = y.size()

        output_mean = torch.mean(output, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)

        a = output - output_mean
        b = y - y_mean

        top = torch.sum(a * b, dim=1)

        b1 = torch.sum(torch.pow(a, 2), dim=1)
        b2 = torch.sum(torch.pow(b, 2), dim=1)
        bottom = torch.sqrt(b1 * b2)

        loss = top / bottom
        return torch.mean(torch.abs(loss))

    def corr(self, x, y):
        x_ = x - torch.mean(x)
        y_ = y - torch.mean(y)
        top = torch.sum(x_ * y_)

        b1 = torch.sum(torch.pow(x_, 2))
        b2 = torch.sum(torch.pow(y_, 2))
        bottom = torch.sqrt(b1 * b2)

        return top / bottom
