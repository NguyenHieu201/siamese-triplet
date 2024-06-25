import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    
class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class ShallowNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ShallowNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(nn.Linear(input_dim, 256),
                                 nn.PReLU(),
                                 nn.Linear(256, 256),
                                 nn.PReLU(),
                                 nn.Linear(256, output_dim),
                                 NormalizeLayer()
                                )


    def forward(self, x1, x2, x3):
        output1 = self.net(x1)
        output2 = self.net(x2)
        output3 = self.net(x3)
        # output1 /= output1.pow(2).sum(1, keepdim=True).sqrt()
        # output2 /= output2.pow(2).sum(1, keepdim=True).sqrt()
        # output3 /= output3.pow(2).sum(1, keepdim=True).sqrt()
        return output1, output2, output3


    def get_embedding(self, x):
        return self.net(x)
    
    
class ShallowEmbeddNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ShallowEmbeddNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(),
        )
        self.net = nn.Sequential(nn.Linear(4096, 128),
                                 nn.PReLU(),
                                 nn.Linear(128, 128),
                                 nn.PReLU(),
                                 nn.Linear(128, 256),
                                 nn.PReLU(),
                                 nn.Linear(256, output_dim),
                                 NormalizeLayer()
                                )


    def forward(self, x):
        return self.get_embedding(x)


    def get_embedding(self, x):
        embedds = self.cnn(x)
        embedds = embedds.flatten(1)
        embedds = self.net(embedds)
        return embedds
    
    
class FlattenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs):
        return inputs.flatten(1)

class CombineNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_classes: int) -> None:
        super(CombineNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(),
            FlattenLayer(),
            nn.Linear(8192, 2048),
            nn.PReLU(),
            nn.Linear(2048, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU()
        )
        # self.backbone = self.__init_backbone()
        self.contrastive_branch = nn.Sequential(
            nn.Linear(256, output_dim),
            NormalizeLayer()
        )
        self.classification_branch = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        
    def __init_backbone(self):
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        backbone.features[0] = nn.Identity()
        backbone.features[1] = nn.Identity()
        backbone.features[2] = nn.Conv2d(128, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        backbone.classifier[3] = nn.Linear(1024, 256, bias=True)
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        embedds = self.contrastive_branch(x)
        confs = self.classification_branch(x)
        return embedds, confs
    
    def get_embedding(self, x):
        x = self.backbone(x)
        embedds = self.contrastive_branch(x)
        return embedds