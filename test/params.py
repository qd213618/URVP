TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "hrnet2_yolo",#hrnet2_yolo
        "backbone_pretrained": "",
    },
    "yolo": {
        # "anchors": [[[116, 90], [156, 198], [373, 326]],
        #             [[30, 61], [62, 45], [59, 119]],
        #             [[10, 13], [16, 30], [33, 23]]],
        # "anchors": [[[347, 347], [194, 194], [104, 104]],#20190220035049
        #             [[126, 126], [96, 96], [42, 42]],
        #             [[27, 27], [19, 19], [10, 10]]],
        "anchors": [[ [104, 104]],#20190220035049,20190221185147
                    [[42, 42]],
                    [[10, 10]]],
        "classes": 1,
        "line_num":23,
        "batchnorm": "BN",
    },
    "batch_size": 1,#70#843,MAX 783,for m2 use 140, for res50 use 70
    "confidence_threshold": 0.03,
    "images_path": "/home/lyb/datasets/vp/val/flickrtx",#flickrtx,PLVPtx
    "test_path": "/home/lyb/datasets/vp/heatmap_kong_val.txt",#heatmap_kong_val
    # "images_path": "./imagesvp6",
    "classes_names_path": "../data/vp.names",
    "img_h": 320,
    "img_w": 320,
    "parallels": [0],
    "lambda_xy": 47,
    "pretrain_floder": "",
    "pretrain_floder_list": ['/hard/vp_hm/hrneth_yolo/size320x320_try0/20200118225337/*.pth'],#/hard/vp_hm/hrnet2_yolo/size320x320_try0/20200112183536/*.pth
}
