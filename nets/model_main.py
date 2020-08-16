import torch
import torch.nn as nn
from collections import OrderedDict
from .backbone import backbone_fn
import torchvision

class ModelMain_PL(nn.Module):
    def __init__(self, config, is_training=True):
        super(ModelMain_PL, self).__init__()
        self.config = config
        self.training = is_training
        self.model_params = config["model_params"]
        #  backbone
        _backbone_fn = backbone_fn[self.model_params["backbone_name"]]
        self.backbone = _backbone_fn(self.model_params["backbone_pretrained"])
        final_out_filter0 = 3
        self.embedding0 = self._make_embedding([64, 128], 48+1, final_out_filter0)# for hrnet2yolo DCD or h1 or hrnethyolo DCD

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))
    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m
    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
            return _in
        hml, hmh, out0 = self.backbone(x)
        out= _branch(self.embedding0, out0)

        return hml, hmh, out#for hrnet2 yolo
