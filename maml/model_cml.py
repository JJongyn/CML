import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    
def conv3x3_nomax(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU()
    )
    
"""
ConvNet - Feature extractor + meta-learner
"""
class ConvNet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size, wh_size):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        # Feature extractor
        self.conv1 = conv3x3(in_channels, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)
        self.conv4 = conv3x3_nomax(hidden_size, hidden_size)

        # Meta-learner
        self.meta_learner = MetaLinear(hidden_size*wh_size*wh_size, out_features)

    def forward(self, inputs, params=None):
        features = self.conv1(inputs, params=get_subdict(params, 'conv1'))
        features = self.conv2(features, params=get_subdict(params, 'conv2'))
        features = self.conv3(features, params=get_subdict(params, 'conv3'))
        features = self.conv4(features, params=get_subdict(params, 'conv4'))
        
        meta_features = F.max_pool2d(features,2)
        meta_features = meta_features.view((meta_features.size(0), -1))
        logits = self.meta_learner(meta_features, params=get_subdict(params, 'meta_learner'))
        
        return features, logits
    
"""
Co-learner
""" 
class CoLearner(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size, wh_size):
        super(CoLearner, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        # Co-learner
        self.fixed_conv1 = conv3x3_nomax(in_channels, hidden_size)
        self.fixed_conv2 = conv3x3(hidden_size, hidden_size)
        self.fixed_cls = MetaLinear(hidden_size*wh_size*wh_size, out_features)

    def forward(self, inputs, params=None):
        features = self.fixed_conv1(inputs, params=get_subdict(params, 'fixed_conv1'))
        features = self.fixed_conv2(features, params=get_subdict(params, 'fixed_conv2'))
       
        features = features.view((features.size(0), -1))
        logits = self.fixed_cls(features, params=get_subdict(params, 'fixed_cls'))
        
        return features, logits
  