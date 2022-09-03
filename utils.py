import os
import argparse
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.preprocessing import normalize


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_class", type=str, default="",
    #                     choices=[])
    parser.add_argument("--backbone_class", type=str, default="Resnet20",
                        choices=['Swin_base', 'Swin_tiny', 'Resnet20', 'Resnet12', 'Resnet50'])
    parser.add_argument("--dataset", type=str, default="TLD",
                        choices=['TLD'])

    # optimization parameters
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    # parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)

    # usually untouched parameters
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='/apdcephfs/private_jiayancchen/checkpoints/')

    # parameters for distillation
    parser.add_argument("--is_distill", action="store_true")
    parser.add_argument("--teacher_backbone_class", type=str, default='', choices=['Swin_base'])
    parser.add_argument('--teacher_init_weights', type=str, default="")
    parser.add_argument('--kd_loss', type=str, default='KD', choices=['KD', 'global_KD'])
    parser.add_argument('--kd_weight', type=float, default=1.0)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--head_fixed", action="store_true")
    parser.add_argument('--suffix', type=str, default="")   # 手动指定的输出目录后缀

    return parser

def postprocess_args(args):
    save_path1 = '-'.join([args.dataset, args.backbone_class])

    if args.is_distill:
        save_path1 += '-{}'.format(args.kd_loss) + '-{}'.format(args.kd_weight)
    else:
        save_path1 += '-no_distill'

    if args.init_weights is not None:
        save_path1 += '-Pre'
    
    if not args.augment:
        save_path1 += '-NoAug'

    if args.optim == "step" or args.optim == "multistep":
        optim_info = args.optim + '-step{}'.format(args.step_size) + '-gamma{}'.format(args.gamma)
    else:
        optim_info = args.optim
    
    save_path2 = '_'.join(['lr{:.2g}'.format(args.lr), optim_info, args.lr_scheduler, 'epoch{}'.format(args.max_epoch), "warmup{}".format(args.warmup_epoch), 'bs{}'.format(args.batch_size)  ] ) #,  str(time.strftime('%Y%m%d_%H%M%S'))]) # 需要中断重启训练，则目录末尾不添加时间戳
    
    if args.head_fixed:
        save_path2 += "-head_fixed"
    
    save_path2 += args.suffix

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, save_path1) ):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    if not os.path.exists(os.path.join(args.save_dir, save_path1, save_path2) ):
        os.mkdir(os.path.join(args.save_dir, save_path1, save_path2))
    else:   # 如果输出目录已经存在，resume
        args.resume = True
    
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    
    if not os.path.exists(os.path.join(args.save_path, "record.txt")):
        f = open(os.path.join(args.save_path, "record.txt"), 'w')
        f.close()
    
    return args



def accuracy(output, target, topk=1):
    """
    Calc the acc of tpok.

    output and target have the same dtype and the same shape.

    Args:
        output (torch.Tensor or np.ndarray): The output.
        target (torch.Tensor or np.ndarray): The target.
        topk (int or list or tuple): topk . Defaults to 1.

    Returns:
        float: acc.
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = {
            "Tensor": torch.topk,
            "ndarray": lambda output, maxk, axis: (
                None,
                torch.from_numpy(topk_(output, maxk, axis)[1]).to(target.device),
            ),
        }[output.__class__.__name__](output, topk, 1)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        # res = correct_k.mul_(100.0 / batch_size).item()
        # print(f"cuda:{dist.get_rank()} res before {res}")
        # correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(correct_k, op=dist.ReduceOp.SUM)
            batch_size *= dist.get_world_size()
        res = correct_k.mul_(100.0 / batch_size).item()
        # print(f"cuda:{dist.get_rank()} res after {res}")
        return res, correct_k

def topk_(matrix, K, axis):
    """
    the function to calc topk acc of ndarrary.

    TODO

    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm




@torch.no_grad()
def test_epoch_openset(test_loader, model, save_path, norm=True):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    feat_all = list()
    with torch.no_grad():
        offset = 0
        for cur_iter, (inputs, labels) in enumerate(test_loader):
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            features, _ = model(inputs, labels) 
            if norm:
                features = F.normalize(features)
            feat_data = features.cpu().detach()  
            feat_data_l = feat_data.numpy().tolist()
            feat_all.extend(feat_data_l)
            if cur_iter % 100 == 0:
                print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'),) + str(cur_iter))

        np.save(save_path, feat_all)
        print('{}  saved:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), save_path))



def find_score(far, vr, target=1e-4):
    # far is an ordered array, find the index of far whose element is closest to target, and return vr[index]
    # assert isinstance(far, list)
    l = 0
    u = far.size - 1
    while u - l > 1:
        mid = (l + u) // 2
        # print far[mid]
        if far[mid] == target:
            return vr[mid]
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    # Actually, either array[l] or both[u] is not equal to target, so I do interpolation here.
    # print (vr[l] + vr[u]) / 2.0
    if far[l] / target >= 8:  # cannot find points that's close enough to target.
        return 0.0
    return (vr[l] + vr[u]) / 2.0


def compute_roc(score, label, num_thresholds=1000):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]
    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    data_max = np.max(score)
    data_min = np.min(score)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    #np.savetxt('threshold.txt', threshold)
    return far, vr

def benchmark_TLDv4web_gpu(save_path):

    val_filelist = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/TLDv4web_matched_10perlandmark_filelist_test_finegrained_0419.txt'
    val_filelist_f = open(val_filelist, 'r', encoding='utf-8')

    mask_path = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/TLDv4web_matched_10perlandmark_test_0419_mask.npy'
    mask_finegrained_path = '/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv4web/TLDv4web_matched_10perlandmark_test_finegrained_0419_mask_0602mergeSame_right.npy'
    norm = True  # True False

    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'start')
    feat_all = np.load(save_path).astype('float16')
    if norm:
        feat_all = normalize(feat_all)
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'load feat end')

    lines = val_filelist_f.readlines()
    num_img = len(lines)
    print('num_img:{}'.format(num_img))  # 37649

    mask_triu = np.load(mask_path)
    mask_finegrained_triu = np.load(mask_finegrained_path)
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'load mask_triu end')

    feat_val = feat_all
    feat_val_t = torch.from_numpy(feat_val).cuda()
    score_val_t = torch.matmul(feat_val_t, feat_val_t.t())
    score_val = score_val_t.cpu().numpy()
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'compute score end')  # 6s
    
    score_val_triu = score_val[np.triu_indices(num_img, 1)]  # 7亿
    
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'score_triu end')  # 16s

    far, vr = compute_roc(score_val_triu.ravel(), mask_triu.ravel(), num_thresholds=2000)
    targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    vrs1 = []
    for tg in targets:
        vrs1.append(np.around(find_score(far, vr, tg), 5))
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'coarsegrained end')
    print('val_coarsegrained vrs:{}'.format(vrs1))

    far, vr = compute_roc(score_val_triu.ravel(), mask_finegrained_triu.ravel(), num_thresholds=2000)
    targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    vrs2 = []
    for tg in targets:
        vrs2.append(np.around(find_score(far, vr, tg), 5))
    print('%s  ' % (time.strftime('%Y-%m-%d %H:%M:%S'), ) + 'finegrained end')
    print('val_finegrained vrs:{}'.format(vrs2))

    return vrs1, vrs2