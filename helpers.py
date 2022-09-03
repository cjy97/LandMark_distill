import torch
import torch.nn as nn
import torch.optim as optim
import os

from commondataset import ImageDataset, ImageDataset_MultiList, default_filelist_reader
from model import Model

def prepare_dataloader(args):

    
    if args.dataset=="TLD":
        # train dataset
        roots = [
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv3/',
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/',
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/ctrip/',
        ]
        filelists = [
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv3/TLDv3_filelist_train.txt',
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/TLDv4web_filelist_train_0427_finegrained_right.txt',
          '/apdcephfs/share_1324356/data/videoqa/scenic_spot/ctrip/TLDv4ctrip_filelist_0530addmore.txt',
        ]

        # val dataset
        val_root = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv3/'
        val_filelist = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv3/TLDv3_filelist_val.txt'

        # test dataset
        test_root = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/'
        test_filelist = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/TLDv4web_matched_10perlandmark_filelist_test_0419.txt'


    # for train
    label2Flabel_v2_path = '/apdcephfs/share_1324356/shared_info/paulhliu/so1so/landmark/models/swin_8gpu_cosfc_s32md0_bs200_loadRight_lrd01_TLDv3v4webfgv4ctrip_ftRight_lrd005_noWeightInit_ftmd2_label2flabel_md2_lrd001_md25/checkpoints/label2Flabel_gtd47SameNamed4_SameNamed34_maxScore_fix0530.pt'

    num_device = torch.cuda.device_count()
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("num_device: {}\n".format(num_device) )
    print("num_device: ", num_device)
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("num_workers: {}\n".format(num_workers) )

    
    train_dataset = ImageDataset_MultiList(root_paths=roots, filelist_paths=filelists, train=True, label2Flabel_path=label2Flabel_v2_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=num_workers)
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("len(train_loader): {}\n".format(len(train_loader)) )

    # for val
    val_dataset = ImageDataset(val_root, val_filelist, train=False, filelist_reader=default_filelist_reader)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=num_workers)
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("len(val_loader): {}\n".format(len(val_loader)) )

    # for test
    test_dataset = ImageDataset(test_root, test_filelist, train=False, filelist_reader=default_filelist_reader)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=num_workers)

    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("len(test_loader): {}\n".format(len(test_loader)) )
    
    return train_loader, val_loader, test_loader

def prepare_model(args):

    model = Model(args)

    if args.init_weights is not None:
        # pretrained_dict = torch.load(args.init_weights)['model_state']
        pretrained_dict = torch.load(args.init_weights)['params']
        pretrained_dict = {k.replace("encoder.module.", ""): v for k, v in pretrained_dict.items() if "encoder.module." in k}

        model_dict = model.encoder.state_dict()     

        for (k1, v1), (k2, v2) in zip( model_dict.items(), pretrained_dict.items() ):
            print(k1, v1.size() )
            print(k2, v2.size() )
            print('--')
    
        model_dict.update(pretrained_dict)
        msg = model.encoder.load_state_dict(model_dict)

        if len(msg.missing_keys) != 0:
            print("Missing keys:{}".format(msg.missing_keys), level="warning")
        if len(msg.unexpected_keys) != 0:
            print("Unexpected keys:{}".format(msg.unexpected_keys), level="warning")

    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("torch.cuda.is_available(): {}\n".format(torch.cuda.is_available()) )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("device: {}\n".format(device) )

    student_para_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    teacher_para_trainable = sum(p.numel() for p in model.distill_layer.parameters() if p.requires_grad)
    student_para_sum = sum(p.numel() for p in model.encoder.parameters())
    teacher_para_sum = sum(p.numel() for p in model.distill_layer.parameters())
    print("student para trainable/sum: ", student_para_trainable, student_para_sum)
    print("teacher para trainable/sum: ", teacher_para_trainable, teacher_para_sum)
    with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
        f.write("student para trainable/sum {}/{}; teacher para trainable/sum {}/{}\n".format(student_para_trainable, student_para_sum, teacher_para_trainable, teacher_para_sum) )

    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        model.distill_layer = nn.DataParallel(model.distill_layer, dim=0)
    
    model = model.to(device)

    return model

def prepare_optimizer(model, args):

    # 如果固定head，则优化器只加载学生模型除head之外部分的参数
    if args.head_fixed:
        param = [v for k,v in model.encoder.named_parameters() if 'head' not in k]
    else:
        param = model.encoder.parameters()

    if args.optim == "SGD":
        optimizer = optim.SGD(
            # model.encoder.parameters(),
            param, 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            # model.encoder.parameters(),
            param, 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(
            # model.encoder.parameters(),
            param, 
            lr=args.lr,
            weight_decay=args.weight_decay #, do not use weight_decay here
        )
    else:
        raise ValueError('No Such Optimizer')

    # sum = 0
    # for para in optimizer.param_groups[0]['params']:
    #     print("para : ", para.size())
    #     sum += para.numel()
    # print("sum: ", sum)


    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    
    return optimizer, lr_scheduler


if __name__ == '__main__':
    train_loader, test_loader = prepare_dataloader(args=None)

    model = prepare_model(args=None)

    for i, (input, label) in enumerate(train_loader):
        input = input.cuda()
        label = label.cuda()
        print("label: ", label.size() )

        print("input: ", input.size() )

        x, logits = model.forward(input, label)
        print("x: ", x.size() )
        print("logits: ", logits.size() )

        print(torch.mean(logits))
        

