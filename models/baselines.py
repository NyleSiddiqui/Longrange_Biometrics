import torch 
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
        
                
class ResNet50(nn.Module):
    def __init__(self, input_size, num_subjects, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.num_subjects = num_subjects
        self.num_classes = num_classes
        self.backbone_model = resnet50(weights=weights).cuda()
        self.extractor = create_feature_extractor(self.backbone_model, return_nodes={"layer4": "features"}) #extract from earlier layer?
        self.avg_pool = nn.AvgPool2d(kernel_size=[7, 7], stride=(1, 1)) # reduce spatial downsize to 7x7
        self.subject_classifier = nn.Linear(2048, self.num_subjects)
        self.classifier = nn.Linear(2048, self.num_classes)
        
    def forward(self, inputs):
        features = self.extractor(inputs)['features']
        features = self.avg_pool(features).squeeze(-1).squeeze(-1)
        output_subjects = self.subject_classifier(features)
        outputs = self.classifier(features)
        return output_subjects, outputs, features