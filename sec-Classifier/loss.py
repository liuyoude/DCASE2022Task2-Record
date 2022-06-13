import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASDLoss(nn.Module):
    def __init__(self, num_classes, alpha, samples_per_cls):
        super(ASDLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=2, num_classes=num_classes,
                                    samples_per_cls=samples_per_cls)
        self.multi_focal_loss = Multi_Focal_Loss(weight=1, gamma=2)

    def forward(self, out, target):
        out_classes, out_atts = out
        labels, one_hots = target
        ce_loss = self.ce_loss(out_classes, labels)
        # ce_loss = self.focal_loss(out_classes, labels)
        bce_loss = self.bce_loss(out_atts, one_hots)
        # bce_loss = self.multi_focal_loss(out_atts, one_hots)
        loss = ce_loss + 0 * bce_loss
        return loss, ce_loss, bce_loss


# TODO focal loss
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

    def __init__(self, num_classes=3, alpha=None, gamma=2, beta=0.999, samples_per_cls=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_num = num_classes
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.Tensor(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)
        if samples_per_cls:
            self.alpha = self.class_balanced_alpha(beta, samples_per_cls)
            print('CB Focal loss', self.alpha)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        self.alpha = self.alpha.to(inputs.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    def class_balanced_alpha(self, beta, samples_per_cls):
        # https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.class_num
        return torch.Tensor(weights)


# def focal_loss(labels, logits, alpha, gamma):
#     """Compute the focal loss between `logits` and the ground truth `labels`.
#     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
#     Args:
#       labels: A float tensor of size [batch, num_classes].
#       logits: A float tensor of size [batch, num_classes].
#       alpha: A float tensor of size [batch_size]
#         specifying per-example weight for balanced cross entropy.
#       gamma: A float scalar modulating loss from hard and easy examples.
#     Returns:
#       focal_loss: A float32 scalar representing normalized total loss.
#     """
#     BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
#
#     if gamma == 0.0:
#         modulator = 1.0
#     else:
#         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
#                                                                            torch.exp(-1.0 * logits)))
#
#     loss = modulator * BCLoss
#
#     weighted_loss = alpha * loss
#     focal_loss = torch.sum(weighted_loss)
#
#     focal_loss /= torch.sum(labels)
#     return focal_loss

class Focal_Loss():
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

class Multi_Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Multi_Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = torch.softmax(preds, dim=1)
        target = labels

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        # floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weights=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss
