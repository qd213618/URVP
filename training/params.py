TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "hrnet2_yolo",#hrnet2_yolo,hrnet_yolo,hrneth_yolo
        "backbone_pretrained": "", #  set empty to disable ../weights/mobilenetv2.pth,../hrnetv2_w48.pth
    },
    "yolo": {
        "anchors": [],
        "classes": 1,
        "batchnorm":"BN",
    },
    "lr": {
        "backbone_lr": 0.01,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 30,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "Adam",
        "weight_decay": 0.0005, # 1e-4,
    },
    "batch_size": 14,#for hrnet2yolo use 14,hg4 use 9,hg8 use 5
    "batch_size_test": 140,
    "confidence_threshold": 0.03,#train:heatmap_tr,heatmap_tr_rote,URVP_tr,URVP_plus
    "train_path": "/home/lyb/datasets/vp/URVP_rote.txt",#"/home/lyb/datasets/vp/new4class_line.txt",#20190225230108 is for 4 rote 416*416,20190225234301 is for 5 rote 416*416,20190226013410 is for 5 rote 416 lambda_xy = 25
    "test_path": "/home/lyb/datasets/vp/heatmap_kong_val.txt",#heatmap_val.txt heatmap_kong_val heatmap_PLVP_val.txt
    "epochs": 60,#for rote 50
    "img_h": 320,#20190224003815 is for 416, 20190226012413 is 5 rote 608
    "img_w": 320,
    "lambda_xy": 35,
    "lambda_xy_start": 45,
    "lambda_xy_end": 56,
    "lambda_xy_step": 10,
    "lambda_list": [2,5,7,8,15,25],#,41,43,45,47,85,35],#for l23 [155,185,15,25,85],[35,45,55,65,115,95,155,185,15,25,85],[35,37,41,43,45,47,85]
    "line_num_min":8,
    "line_num_max":24,
    "line_num_step":3,
    "parallels": [1,0],                         #  config GPU device
    "working_dir": "/hard/vp_hm",              #  replace with your working dir"/home/lyb/workspace/vp/YOLOv3_PyTorch"
    "pretrain_snapshot": "",#  load checkpoint ../mobilenetv2/size416x416_try0/20190421194007/model_min_avg.pth #for rote 5/hard/vp/m2/mobilenetv2/size416x416_try0/20190505215441/model_24_85_l26.pth   /hard/vp/m2/mobilenetv2/size416x416_try0/20190504112645/model_e28_s50_xy85_l25.pth
    "evaluate_type": "",
    "try": 0,
    "export_onnx": False,
}
