# for swin-tiny
python3.6 main.py \
--backbone_class Swin_tiny \
--dataset TLD \
--max_epoch 200 \
--warmup_epoch 10 \
--batch_size 256 \
--lr 0.001 \
--optim AdamW \
--lr_scheduler cosine \
--augment \
--multi_gpu \
--is_distill \
--teacher_backbone_class Swin_base \
--teacher_init_weights "/apdcephfs/share_1324356/shared_info/paulhliu/so1so/landmark/models/swin_8gpu_cosfc_s32md0_bs200_loadRight_lrd01_TLDv3v4webfgv4ctrip_ftRight_lrd005_noWeightInit_ftmd2_label2flabel_md2_lrd001_md25/checkpoints/model_epoch_0030.pyth" \
--kd_loss KD \
--kd_weight 1.0 \
--head_fixed \

# for resnet12
# python3.6 main.py \
# --backbone_class Resnet12 \
# --dataset TLD \
# --max_epoch 40 \
# --warmup_epoch 0 \
# --batch_size 256 \
# --lr 0.01 \
# --optim SGD \
# --lr_scheduler cosine \
# --augment \
# --multi_gpu \
# --is_distill \
# --teacher_backbone_class Swin_base \
# --teacher_init_weights "/apdcephfs/share_1324356/shared_info/paulhliu/so1so/landmark/models/swin_8gpu_cosfc_s32md0_bs200_loadRight_lrd01_TLDv3v4webfgv4ctrip_ftRight_lrd005_noWeightInit_ftmd2_label2flabel_md2_lrd001_md25/checkpoints/model_epoch_0030.pyth" \
# --kd_loss KD \
# --kd_weight 1.0 \
# --head_fixed \