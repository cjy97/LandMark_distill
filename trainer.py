import numpy as np
import torch
import torch.nn.functional as F
import os
import time

import abc
import lr_sched as lr_sched


from helpers import (
    prepare_dataloader, prepare_model, prepare_optimizer,
)

from utils import accuracy, compute_confidence_interval, test_epoch_openset, benchmark_TLDv4web_gpu



def prepare_one_hot_label(label, num_class):
    one_hot_label = torch.zeros([label.size(0), num_class])
    one_hot_label = one_hot_label.cuda()

    print("one_hot_label: ", one_hot_label.size())
    one_hot_label.scatter_(1, label.view(-1, 1), 1.0)
    print("one_hot_label: ", one_hot_label.size())

    return one_hot_label


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args


        # ------------- #

        self.train_loader, self.val_loader, self.test_loader = prepare_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
    
    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            os.path.join(self.args.save_path, name + '.pth')
        )

    def save_checkpoint(self, args, epoch, max_acc, model, optimizer, filename='checkpoint.pth.tar'):
        state = {
                 'args': args,
                 'epoch': epoch + 1,
                 'max_acc' : max_acc,
                 'state_dict': model.state_dict(),
                 'optimizer' : optimizer.state_dict()                 
                }
        
        torch.save(state, os.path.join(args.save_path, filename))
    

    def train(self):

        args = self.args
        self.model.train()

        # 如果resume，加载此前保存的运行状态
        if args.resume == True:
            state = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))

            init_epoch = state['epoch']
            max_acc = state['max_acc']
            msg = self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict( state['optimizer'])

            with open(os.path.join(self.args.save_path, 'record.txt'), 'a') as f:
                f.write("--resume from epoch:{} max_acc:{}, {}\n".format(init_epoch, max_acc, msg) )

        else:
            init_epoch = 1
            max_acc = 0.0

        for epoch in range(init_epoch, args.max_epoch + 1):


            # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            

            self.model.train()

            # correct = 0 # 统计每个输出interval的分类正确样本数
            # total = 0   # 统计每个输出interval的样本总数

            train_len = len(self.train_loader)
            record = np.zeros((train_len, 3)) # ce_loss, kd_loss and acc

            for i, batch in enumerate(self.train_loader):

                # 改为每个batch更新一次学习率
                lr = lr_sched.adjust_learning_rate(self.optimizer, i / train_len + epoch, args)

                input, label = batch
                input = input.cuda()
                label = label.cuda()

                x, logits, teacher_x, teacher_logits = self.model.forward(input, label)
                # print("x: ", x.size() )                             # [bs, 1024]
                # print("logits: ", logits.size() )                   # [bs, 252903]
                # print("teacher_x: ", teacher_x.size() )             # [bs, 1024]
                # print("teacher_logits: ", teacher_logits.size() )   # [bs, 252903]

                if self.args.kd_loss == "KD":
                    if teacher_logits is not None:
                        T = 4.0
                        p_s = F.log_softmax(logits / T, dim=1)
                        p_t = F.softmax(teacher_logits, dim=1)
                        kd_loss = F.kl_div(
                            p_s,
                            p_t,
                            reduction='sum'
                            # size_average=False
                        ) #* (T**2)                        
                    else:
                        kd_loss = torch.Tensor([0.0])
                
                elif self.args.kd_loss == "global_KD":
                    if teacher_x is not None:
                        kd_loss = F.mse_loss(x, teacher_x, reduction='sum')
                    else:
                        kd_loss = torch.Tensor([0.0])
                
                ce_loss = F.cross_entropy(logits, label)

                if self.args.is_distill:
                    loss = ce_loss + kd_loss * args.kd_weight
                else:
                    loss = ce_loss

                acc, correct_k = accuracy(logits, label.reshape(-1), topk=1)

                record[i, 0] = ce_loss.item()
                record[i, 1] = kd_loss.item()
                record[i, 2] = acc

                # pred = torch.topk(logits, k=1, dim=1)[1]
                # correct += correct_k # pred.T.eq(label.view(1, -1).expand_as(pred.T)).sum().item()
                # total   += label.reshape(-1).size(0)
                
                
                # if i % 100 == 0 or i == train_len-1:
                #     print("epoch{}: ({}/{})--{}".format(epoch, i, train_len, str(time.strftime('%Y%m%d_%H:%M:%S'))))
                #     print("ce_loss: ", ce_loss)
                #     print("kd_loss: ", kd_loss)
                #     print('correct/total: {}/{}'.format(correct, total) )

                #     correct = 0
                #     total = 0



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            t_celoss, _ = compute_confidence_interval(record[:,0])
            t_kdloss, _ = compute_confidence_interval(record[:,1])
            t_acc,    _ = compute_confidence_interval(record[:,2])

            # self.lr_scheduler.step()
            vl, va, vap = self.evaluate()
            print("epoch {}  current learning rate: {}".format(epoch, lr))
            self.epoch_record(epoch, lr, vl, va, vap, train_acc = t_acc, avg_ce_loss = t_celoss, avg_kd_loss = t_kdloss)

            self.save_checkpoint(args, epoch, max_acc, self.model, self.optimizer)

            if va >= max_acc:
                max_acc = va
                self.save_model("max_acc")


    def evaluate(self):

        self.model.eval()

        val_len = len(self.val_loader)
        record = np.zeros((val_len, 2)) # loss and acc
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):

                input, label = batch

                input = input.cuda()
                label = label.cuda()

                logits = self.model.forward(input, label)

                loss = F.cross_entropy(logits, label)
                acc, _ = accuracy(logits, label.reshape(-1), topk=1)

                record[i, 0] = loss.item()
                record[i, 1] = acc

                # print("{}/{}".format(i, val_len))
                # print('pred targets are {}'.format(pred.T))
                # print('eq targets are {}/{}'.format(pred.T.eq(label.view(1, -1).expand_as(pred.T)).sum().item(), label.reshape(-1).shape[0] ))
                # print("Acc: ", acc)

        vl, _   = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        self.model.train()

        print("------------vl, va, vap: ", vl, va, vap)
        return vl, va, vap


    def test(self): # 测试集的label编号与训练集不一致，无法直接采用分类方法；需采用特征匹配方案

        # 加载最优模型的权重
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()

        save_path = os.path.join(self.args.save_path, "opensetFeat.npy")
        test_epoch_openset(self.test_loader, self.model.encoder, save_path)

        vrs1, vrs2 = benchmark_TLDv4web_gpu(save_path)        

        with open(os.path.join(self.args.save_path, 'record.txt'), 'a') as f:
            f.write("test acc: coarsegrained:{}--finegrained:{}\n".format(vrs1, vrs2) )

        
        """
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        
        test_len = len(self.test_loader)
        record = np.zeros((test_len, 2)) # loss and acc

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):    # 测试集的标签与训练集不一致，无法直接测试
                
                input, label = batch

                input = input.cuda()
                label = label.cuda()
                print("label: ", label)

                logits = self.model.forward(input, label)

                loss = F.cross_entropy(logits, label)
                acc, _ = accuracy(logits, label.reshape(-1), topk=1)

                record[i, 0] = loss.item()
                record[i, 1] = acc
                # print("{}/{}".format(i, test_len))
                # print('pred targets are {}'.format(pred.T))
                # print('eq targets are {}/{}'.format(pred.T.eq(label.view(1, -1).expand_as(pred.T)).sum().item(), label.reshape(-1).shape[0] ))
                print("Acc: ", acc)

        tl, _ = compute_confidence_interval(record[:,0])
        ta, tap = compute_confidence_interval(record[:,1])

        self.model.train()

        with open(os.path.join(self.args.save_path, 'record.txt'), 'a') as f:
            f.write("test acc: {}\n".format(ta) )
        print("test loss, teat acc, test ap: ", tl, ta, tap)
        """

    def epoch_record(self, epoch, lr, vl, va, vap, train_acc, avg_ce_loss, avg_kd_loss):
        print(self.args.save_path)
        with open(os.path.join(self.args.save_path, 'record.txt'), 'a') as f:
            f.write('epoch {}  lr {}: train_acc={:.4f}, eval_loss={:.4f}, eval_acc={:.4f}+{:.4f}, avg_ce_loss={:.4f}, avg_kd_loss={:.4f}\n'.format(epoch, lr, train_acc, vl, va, vap, avg_ce_loss, avg_kd_loss))
