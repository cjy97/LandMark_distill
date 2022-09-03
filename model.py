from copy import deepcopy
import torch.nn as nn
import torch

from swin_transformer import SwinTransformer

class DistillLayer(nn.Module):
    def __init__(
        self,
        teacher_backbone_class,
        teacher_init_weights,
        is_distill,
        kd_loss,
    ):
        super(DistillLayer, self).__init__()
        self.teacher_backbone_class = teacher_backbone_class
        self.encoder = self._load_state_dict(teacher_backbone_class, teacher_init_weights, is_distill, type = "encoder.")

        # if kd_loss == "KD" or kd_loss == "ALL":     # 只有基于分类logits的传统蒸馏方法（KD）或全部损失方法并用时，教师模型需要加载线性分类头
        #     self.fc = self._load_state_dict(teacher_backbone_class, teacher_init_weights, is_distill, type = "fc.")
        #     self.GAP = nn.AvgPool2d(5, stride=1)
        # else:
        #     self.fc = None

    def _load_state_dict(self, teacher_backbone_class, teacher_init_weights, is_distill, type):
        new_model = None

        if is_distill and teacher_init_weights is not None:
            
            # if type == "encoder.":
            #     if teacher_backbone_class == 'Res12':
            #         new_model = ResNet()
            # elif type == "fc.":
            #     if teacher_backbone_class == 'Res12':
            #         new_model = nn.Linear(640, 64)

            if type == "encoder.":
                if teacher_backbone_class == "Swin_base":
                    new_model = SwinTransformer(img_size=384, patch_size=4, num_classes=252903, embed_dim=128,
                                depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=12,
                                cos_fc=True, cos_scale=32.0, cos_margin=0.1,
                                ml=False, scoremap=False, one_classifier=False)
            
            model_dict = new_model.state_dict()

            pretrained_dict = torch.load(teacher_init_weights)['model_state']
            pretrained_dict = {k.replace(type, ""): v for k, v in pretrained_dict.items() if k.replace(type, "") in model_dict.keys()}

            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict)   # 只将权重加载给教师模型

            for k, _ in pretrained_dict.items():
                print("pretrained key: ", k)
            for key, _ in new_model.state_dict().items():
                print("key: ", key)

        return new_model
    

    @torch.no_grad()
    def forward(self, input, label):

        if self.encoder is not None:
            if self.teacher_backbone_class == "Swin_base":
                x, logits = self.encoder.forward(input, label)
                return x, logits
        else:
            return None, None




class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.backbone_class == "Resnet12":
            from resnet12 import resnet12
            self.encoder = resnet12(num_classes=252903, cos_fc=True, cos_scale=32.0, cos_margin=0.1)
        # elif args.backbone_class == "Resnet20":
        #     from resnet20 import resnet20
        #     self.encoder = resnet20(num_class=252903)
        # elif args.backbone_class == "Resnet50":
        #     from resnet50 import resnet50
        #     self.encoder = resnet50(num_classes=252903)
        elif args.backbone_class == "Swin_tiny":    # Swin_tiny的结构做了一定修改，以匹配1024维输出特征
            # self.encoder = SwinTransformer(img_size=384, patch_size=4, num_classes=252903, embed_dim=96,
            #                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=12,
            #                     cos_fc=True, cos_scale=32.0, cos_margin=0.1,
            #                     ml=False, scoremap=False, one_classifier=False)

            self.encoder = SwinTransformer(img_size=384, patch_size=4, num_classes=252903, embed_dim=128,
                                depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16), window_size=12,
                                cos_fc=True, cos_scale=32.0, cos_margin=0.1,
                                ml=False, scoremap=False, one_classifier=False)
            
        else:
            raise ValueError('')
    
        self.distill_layer = DistillLayer(
            args.teacher_backbone_class,
            args.teacher_init_weights,
            args.is_distill,
            args.kd_loss,
        ).requires_grad_(False)

        if args.head_fixed:
            self.init_student_head()
            self.encoder.head.requires_grad_(False)   # 学生模型的分类器头固定梯度，不训练

    def init_student_head(self):  # 用教师模型的分类头权重初始化学生模型的分类头
        for k, v in self.distill_layer.state_dict().items():
            # print("distill weight: ", k, v.size())
            if 'head' in k:
                head_weight = {"weight": v}
        
        # for k, v in self.encoder.state_dict().items():
        #     print("encoder weight: ", k, v.size())
        
        msg = self.encoder.head.load_state_dict(head_weight)
        print("student head: ", msg)
        
        if len(msg.missing_keys) != 0:
            print("Missing keys:{}".format(msg.missing_keys), level="warning")
        if len(msg.unexpected_keys) != 0:
            print("Unexpected keys:{}".format(msg.unexpected_keys), level="warning")

    def forward(self, input, label):

        if self.args.backbone_class == "Swin_tiny" or self.args.backbone_class == "Swin_base":
            x, logits = self.encoder(input, label)
        elif self.args.backbone_class == "Resnet12":
            x, logits = self.encoder(input, label)
        # elif self.args.backbone_class == "Resnet20":
        #     x, logits = self.encoder(input)
        # elif self.args.backbone_class == "Resnet50":
        #     x, logits = self.encoder(input)
        else:
            assert 0

        if self.training:

            teacher_x, teacher_logits = self.distill_layer(input, label)
            return x, logits, teacher_x, teacher_logits
            
        else:
            return logits