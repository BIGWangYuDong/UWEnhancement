train_name = 'Train'
val_name = 'Val'
test_name = 'Test'
# backbone 和 init_type 需要写
model = dict(type='DCPDehazeSimple',
             backbone=dict(type='DenseNew', pretrained=True),
             pretrained=True,
             init_weight_type=dict(type='normal_init',
                                   mean=0,
                                   std=1),
             get_parameter=True,
             )
dataset_type = 'AlignedDataset'

data_root_train = '/home/dong/python-project/Dehaze/DATA/Train/'                  # data root, default = DATA
data_root_test = '/home/dong/python-project/Dehaze/DATA/Test/'
train_ann_file_path = 'train.txt'        # txt file for loading images, default = train.txt
train_ann_finetune_path = 'train_finetune.txt'
train_ann_finetune_addNH_path = 'train_finetune_addNH.txt'
val_ann_file_path = 'val.txt'          # txt file for loading images (validate during training process), default = test.txt
test_ann_file_path = 'test.txt'         # txt file for loading images, default = test.txt


img_norm_cfg = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
train_pipeline = [dict(type='LoadImageFromFile', gt_type='color'),
                  dict(type='Resize', img_scale=(550, 550), keep_ratio=True),
                  dict(type='RandomCrop', img_scale=(512, 512)),
                  dict(type='RandomRotate90', ratio=0.5),
                  dict(type='RandomRotate180', ratio=0.5),
                  dict(type='RandomFlip', flip_ratio=0.5),
                  # dict(type='Pad', size_divisor=32, mode='resize'),
                  dict(type='ImageToTensor'),
                  dict(type='Normalize', **img_norm_cfg)]
test_pipeling = [dict(type='LoadImageFromFile', gt_type='color'),
                 # dict(type='Resize', img_scale=(256,256), keep_ratio=True),
                 # dict(type='Pad', size_divisor=32, mode='resize'),
                 dict(type='FlipEnsemble'),
                 dict(type='ImageToTensor'),
                 dict(type='Normalize', **img_norm_cfg)]

data = dict(
    samples_per_gpu=2,                                  # batch size, default = 4
    workers_per_gpu=0,                                  # multi process, default = 4, debug uses 0
    val_samples_per_gpu=1,                              # validate batch size, default = 1
    val_workers_per_gpu=0,                              # validate multi process, default = 4
    train=dict(                                         # load data in training process, debug uses 0
        type=dataset_type,
        # ann_file=data_root_train + train_ann_file_path,
        # ann_file=data_root_train + train_ann_finetune_path,
        ann_file=data_root_train + train_ann_finetune_addNH_path,
        img_prefix=data_root_train + 'train/',
        gt_prefix=data_root_train + 'gt/',
        pipeline=train_pipeline),
    val=dict(                                           # load data in validate process
        type=dataset_type,
        ann_file=data_root_test + val_ann_file_path,
        img_prefix=data_root_test + 'train/',
        gt_prefix=data_root_test + 'gt/',
        pipeline=test_pipeling),
    test=dict(                                          # load data in test process
        type=dataset_type,
        ann_file=data_root_test + val_ann_file_path,
        img_prefix=data_root_test + 'train/',
        gt_prefix=data_root_test + 'gt/',
        pipeline=test_pipeling,
        test_mode=True))

train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SSIM', 'MSE', 'PSNR'])

loss_ssim = dict(type='SSIMLoss', window_size=11,
                 size_average=True, loss_weight=1.0)
loss_l1 = dict(type='L1Loss', loss_weight=2.0)
loss_perc = dict(type='PerceptualLoss', loss_weight=0.5,
                 no_vgg_instance=False, vgg_mean=False,
                 vgg_choose='conv4_3', vgg_maxpooling=False)
loss_fft = dict(type='FFTLoss', loss_weight=0)
loss_tv = dict(type='TVLoss', loss_weight=2.0)

optimizer = dict(type='Adam', lr=1e-4, betas=[0.9, 0.999])    # optimizer with type, learning rate, and betas.

# 需要写iter
lr_config = dict(type='Epoch',          # Epoch or Iter
                 warmup='step',       # liner, step, exp,
                 step=[100, 700],          # start with 1
                 liner_end=0.00001,
                 step_gamma=0.1,
                 exp_gamma=0.9)

# 需要写
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(type='VisdomLoggerHook')
    ])

total_epoch = 1500
total_iters = None                      # epoch before iters,
work_dir = './checkpoints/wyd/New/znew/dehaze_4'      #
# load_from = None
load_from = '/home/dong/python-project/Dehaze/checkpoints/wyd/New/znew/dehaze_3/epoch_381.pth'                  # only load network parameters
resume_from = None                      # resume training
save_freq_iters = 500                   # saving frequent (saving every XX iters)
save_freq_epoch = 1                     # saving frequent (saving every XX epoch(s))
log_level = 'INFO'                      # The level of logging.

savepath = 'results/wyd/New/znew/dehaze_4'