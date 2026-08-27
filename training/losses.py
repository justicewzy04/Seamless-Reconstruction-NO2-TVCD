import math
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import torch


def CE_Loss(inputs, target, cls_weights, num_classes=1):
    _, c, h, w = inputs.size()
    _, _, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss
def MSEloss1(inputs, target):
    MSEloss = nn.MSELoss()(inputs, target)
    return MSEloss
def MSEloss2(inputs, target,mask):

    #过滤掉两个变量中任一个变量中为0的区域
    filtered_target = target[mask]
    filtered_inputs = inputs[mask]
    # mask = ~torch.isnan(target)
    # masked_target = target[mask]
    # masked_inputs = inputs[mask]


    # mask2 = ~torch.isnan(masked_inputs)
    # masked_target2 = masked_target[mask2]
    # masked_inputs2 = masked_inputs[mask2]    

    MSEloss = nn.MSELoss()(filtered_inputs, filtered_target)
    # MSEloss = nn.MSELoss()(inputs, target)
    return MSEloss
def Compositeloss(pred, target,mask,geos):
    def geos_loss(pred, target, geos):

        # 1. 物理约束（预测值应与GEOS保持趋势一致）
        trend_loss = F.l1_loss(pred - pred.mean(), geos - geos.mean())
        
        # 2. 残差学习强化
        residual = target - geos
        res_loss = F.mse_loss(pred - geos, residual)
        
        return 0.5 * res_loss + 0.5 * trend_loss
        #抗低估分位数损失
    def Quantile_Loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
        losses = []
        for q in quantiles:
            errors = target - pred
            losses.append(torch.max((q-1) * errors, q * errors).mean())
        
        # 对高浓度区（90分位）施加3倍权重
        return 0.3*losses[0] + losses[1] + 3.0*losses[2]

    filtered_target = target[mask]
    filtered_pred = pred[mask]
    filtered_geos = geos[mask]

    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    geos_loss=geos_loss(filtered_pred, filtered_target, filtered_geos)
    Quantile_Loss=Quantile_Loss(filtered_pred, filtered_target)


    totalloss=(0.4*MSEloss+0.3*geos_loss+0.3*Quantile_Loss)
    return totalloss

def Compositeloss2(pred, target,mask,geos):
    def geos_loss(pred, geos):

        # 1. 物理约束（预测值应与GEOS保持趋势一致）
        trend_loss = F.l1_loss(pred - pred.mean(), geos - geos.mean())
        
        return trend_loss
    filtered_target = target[mask]
    filtered_pred = pred[mask]
    filtered_geos = geos[mask]

    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    geos_loss=geos_loss(filtered_pred, filtered_geos)
    totalloss=(0.7*MSEloss+0.3*geos_loss)
    return totalloss

def Compositeloss3(pred, target,mask,geos):
    def geos_loss(pred, geos):

        # 1. 物理约束（预测值应与GEOS保持趋势一致）
        trend_loss = F.l1_loss(pred - pred.mean(), geos - geos.mean())

        
        return trend_loss

    filtered_target = target[mask]
    filtered_pred = pred[mask]
    filtered_geos = geos[mask]

    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    geos_loss=geos_loss(filtered_pred, filtered_geos)
    totalloss=(0.5*MSEloss+0.5*geos_loss)

    return totalloss

def Compositeloss4(pred, target,mask,geos):
    def geos_loss(pred, geos):

        # 1. 物理约束（预测值应与GEOS保持趋势一致）
        trend_loss = F.l1_loss(pred - pred.mean(), geos - geos.mean())

        
        return trend_loss

    filtered_target = target[mask]
    filtered_pred = pred[mask]
    filtered_geos = geos[mask]

    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    geos_loss=geos_loss(filtered_pred, filtered_geos)
    totalloss=(0.3*MSEloss+0.7*geos_loss)

    return totalloss








def Compositeloss20(pred, target,mask,geos):
    def geos_loss(pred, target, geos):
        # 1. 物理约束（预测值应与GEOS保持趋势一致）
        trend_loss = F.l1_loss(pred - pred.mean(), geos - geos.mean())
        # 2. 残差学习强化
        residual = target - geos
        res_loss = F.mse_loss(pred - geos, residual)
        return 0.5 * res_loss + 0.5 * trend_loss
    filtered_target = target[mask]
    filtered_pred = pred[mask]
    filtered_geos = geos[mask]

    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    geos_loss=geos_loss(filtered_pred, filtered_target, filtered_geos)
    totalloss=(0.2*MSEloss+0.8*geos_loss)
    return totalloss

def Compositeloss21(pred, target, mask):
    # 加强对30%-70%的约束，期望改善主体NMB指标
    def Quantile_Loss(pred, target, quantiles=[0.3, 0.5, 0.7]):
        losses = []
        for q in quantiles:
            errors = target - pred
            losses.append(torch.max((q-1) * errors, q * errors).mean())
        return 0.3*losses[0] + 0.4*losses[1] + 0.3*losses[2]
    filtered_target = target[mask]
    filtered_pred = pred[mask]
    MSEloss = nn.MSELoss()(filtered_pred, filtered_target)
    Quantile_Loss=Quantile_Loss(filtered_pred, filtered_target)
    totalloss=(0.4*MSEloss+0.6*Quantile_Loss)
    return totalloss



def weighted_mean_loss(inputs, target,mask):
    filtered_target = target[mask]
    filtered_inputs = inputs[mask]


    weights = torch.Tensor([0.9789,0.0154,0.0039,0.0012,0.0004,0.0001,3.1731e-05,3.1731e-05,4.4706e-06,1.8729e-06])

    # 生成区间边界
    intervals = torch.linspace(0, 100, 11)

    # 初始化权重和加权平均值
    filtered_target_means = torch.zeros(len(intervals) - 1)
    filtered_target_weighted_avg = torch.zeros(1)

    # 计算每个区间的加权平均值
    for i in range(len(intervals) - 1):
        filtered_target_mask = (filtered_target >= intervals[i]) & (filtered_target < intervals[i+1])
        if filtered_target_mask.any():
            filtered_target_means[i] = torch.mean(filtered_target[filtered_target_mask])


    # 计算加权平均值
    filtered_target_weighted_avg =  torch.sum(torch.matmul(filtered_target_means, weights))


    ## 初始化权重和加权平均值
    filtered_inputs_means = torch.zeros(len(intervals) - 1)
    filtered_inputs_weighted_avg = torch.zeros(1)

    # 计算每个区间的加权平均值
    for i in range(len(intervals) - 1):
        filtered_inputs_mask = (filtered_inputs >= intervals[i]) & (filtered_inputs < intervals[i+1])
        if filtered_inputs_mask.any():
            filtered_inputs_means[i] = torch.mean(filtered_inputs[filtered_inputs_mask])
            
            
    # 计算加权平均值
    filtered_inputs_weighted_avg =  torch.sum(torch.matmul(filtered_inputs_means, weights))


    weighted_mean_loss= torch.abs(filtered_target_weighted_avg-filtered_inputs_weighted_avg)
    
    return weighted_mean_loss


def Ratioloss(inputs, target,mask):

    #过滤掉两个变量中任一个变量中为0的区域
    filtered_target = target[mask]
    filtered_inputs = inputs[mask]
    Ratioloss = torch.mean(torch.abs((filtered_inputs - filtered_target)/ filtered_target))
    return Ratioloss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    _, c, h, w = inputs.size()
    _, _, ht, wt = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    _, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
