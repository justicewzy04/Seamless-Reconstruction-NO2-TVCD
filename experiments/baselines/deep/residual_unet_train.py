import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ufill.data.baseline_dataset import BaselineDataset as APDataset
from ufill.models.residual_unet import Unet
from ufill.training.callbacks import LossHistory
from ufill.training.engine import fit_one_epoch
from ufill.training.losses import get_lr_scheduler, set_optimizer_lr, weights_init
from ufill.utils.image import download_weights, show_config
from ufill.config import CHECKPOINT_ROOT, DATA_ROOT, OUTPUT_ROOT, SPLIT_ROOT, env_path


if __name__ == "__main__":
    Cuda = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False

    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    #-----------------------------------------------------#
    # 映射回归
    num_classes = 1

    backbone    = "resnet50"

    pretrained  = False
    model_path = env_path("UFILL_MODEL_PATH", CHECKPOINT_ROOT / "unet_vgg_voc.pth")
    input_shape = [1408, 800]
    
 
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2

    Freeze_Train        = False


    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'step'
    save_period         = 10
    save_dir            = OUTPUT_ROOT / "baselines" / "residual_unet" / "training"
    
    #------------------------------
    #   数据集路径
    #------------------------------#
    GEMS_path = env_path("UFILL_GEMS_ROOT", DATA_ROOT / "GEMS" / "hourly")
    TROPOMI_path = env_path("UFILL_TROPOMI_ROOT", DATA_ROOT / "TROPOMI" / "daily")
    #GEOS_path = os.path.join(relative_path,'data/air/GEOS_V2.1')
    GEOS_path = env_path("UFILL_GEOS_ROOT", DATA_ROOT / "GEOS-CF_VM_daily")
    train_list = SPLIT_ROOT / "train.txt"
    val_list = SPLIT_ROOT / "val.txt"
    #------------------------------------------------------------------#

    dice_loss       = False
    focal_loss      = False
    loss_type = 'mse_mask'
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 1

    ngpus_per_node  = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)
        
    #初始化model，并调用Unet
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    
    with open(train_list,"r") as f:
        train_lines = f.readlines()
    with open(val_list,"r") as f:
        val_lines = f.readlines()    
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = APDataset(train_lines, GEMS_path, TROPOMI_path, GEOS_path)
        val_dataset = APDataset(val_lines, GEMS_path, TROPOMI_path, GEOS_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True,  sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True,  sampler=val_sampler)
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True,  sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True,  sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, loss_type, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.close()
