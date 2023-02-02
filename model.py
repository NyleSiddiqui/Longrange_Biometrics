import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from models.baselines import ResNet50
from models.v1 import LongRangeTransformer

def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)

def build_model(version, input_size, num_subjects, num_classes, patch_size, hidden_dim, num_heads, num_layers):
    if version == 'resnet':
        model = ResNet50(input_size, num_subjects, num_classes)
    elif version == 'v1':
        model = LongRangeTransformer(input_size, num_subjects, num_classes, hidden_dim, num_heads, num_layers)
    model.apply(weights_init)
    return model



if __name__ == '__main__':
    model = build_model('resnet', 224, 104, 6, 0, 2048, 8, 3)
    model.cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    model.eval()
    inputs = Variable(torch.rand(2, 3, 224, 224)).cuda()
    
    output_subjects, output_actions, features = model(inputs)
    print(output_subjects.shape, output_actions.shape, features.shape, flush=True)


