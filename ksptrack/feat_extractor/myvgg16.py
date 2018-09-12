import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG
from torchvision import transforms
import os

# Removes the classification layer for feature extraction
class MyVGG16(nn.Module):
    def __init__(self, load_model=os.path.join('models', 'vgg16.pth'),
                 cuda=True):

        super(MyVGG16, self).__init__()

        # Download original pre-trained VGG16 if not given
        if(not os.path.exists(load_model)):
            self.model = models.vgg16(pretrained=True)
            os.makedirs(os.path.split(load_model)[0])
            torch.save(self.model, load_model)
        else:
            self.model = torch.load(load_model)

        if(cuda):
            self.model.cuda()
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=range(torch.cuda.device_count()))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))])

        self.feat_size = 4096
        torch.set_num_threads(4)
        self.model.eval()

        # remove unncessary classification layer
        #feat_model = self.model.fe
        classif_model = list(self.model.classifier.children())[0]
        self.model.classifier = nn.Sequential(classif_model)

    def get_features(self, img_stack):

        # Initialize feature image as zeros
        features = torch.stack([torch.zeros((1, self.feat_size))
                                for i in range(img_stack.shape[0])]).squeeze()

        def copy_data(m, i, o):
            features.copy_(o.data)

        # Create hook feat_layer
        h = self.model.register_forward_hook(copy_data)

        self.model(img_stack)

        h.remove()

        return features
