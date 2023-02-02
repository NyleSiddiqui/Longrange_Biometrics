from dis import dis
import torch 
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from einops import rearrange, repeat

                
class LongRangeTransformer(nn.Module):
    def __init__(self, input_size, num_subjects, num_classes, hidden_dim, num_heads, num_layers):
        super(LongRangeTransformer, self).__init__()
        self.num_subjects = num_subjects
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        weights = ResNet50_Weights.DEFAULT
        vweights = R3D_18_Weights
        #self.backbone_model = resnet50(weights=weights).cuda()
        self.backbone_model = r3d_18(weights=weights).cuda()
        self.extractor = create_feature_extractor(self.backbone_model, return_nodes={"layer4": "features"}) #extract from earlier layer?
        self.distance_tokens =  nn.Parameter(torch.randn(self.num_classes, hidden_dim), requires_grad=True)
        self.positional_encoding = nn.Parameter(torch.randn(49 + self.num_classes, hidden_dim), requires_grad=True)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True, dropout=0.0, norm_first=False)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)
        self.subject_classifier = nn.Linear(2048, self.num_subjects)
        self.classifier = nn.Linear(2048, 1)
        
    def forward(self, inputs):
        bs = inputs.shape[0]
        features = self.extractor(inputs)['features']
        features = rearrange(features, 'b f h w -> b (h w) f')
        distance_tokens = repeat(self.distance_tokens, 'n f -> b n f', b=bs)
        _features = torch.cat((features, distance_tokens), axis=1)
        _features += self.positional_encoding[None, :, :]
        _features = self.encoder(_features)
        features_subject = _features[:, -self.num_classes:, :]
        features_distance = _features[:, :-self.num_classes, :]
        output_subjects = self.subject_classifier(torch.mean(features_subject, axis=1))
        outputs = self.classifier(features_distance).squeeze(-1)
        return output_subjects, outputs, torch.mean(features_subject, axis=1)