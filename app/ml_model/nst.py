import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from torchvision.utils import save_image

logger = logging.getLogger("ml_model")

SIZE_IMG = 512
NUM_STEPS = 300
CNN_NORMALIZATION_MEAN_VALUES = [0.485, 0.456, 0.406]
CNN_NORMALIZATION_STD_VALUES = [0.229, 0.224, 0.225]


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = NeuralStyleTransferModel.get_gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        G = NeuralStyleTransferModel.get_gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class NeuralStyleTransferModel:
    def __init__(self):
        self._size_img = SIZE_IMG
        self._loader = transforms.Compose([
            transforms.Resize(self._size_img),
            transforms.CenterCrop(self._size_img),
            transforms.ToTensor()
        ])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._content_layers_default = ['conv_4']
        self._style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self._cnn = models.vgg19(pretrained=True).features.to(self._device).eval()
        self._cnn_normalization_mean = torch.tensor(CNN_NORMALIZATION_MEAN_VALUES).to(self._device)
        self._cnn_normalization_std = torch.tensor(CNN_NORMALIZATION_STD_VALUES).to(self._device)

    def _image_loader(self, image_name):
        image = Image.open(image_name)
        image = self._loader(image).unsqueeze(0)
        return image.to(self._device, torch.float)

    @staticmethod
    def get_gram_matrix(input):
        batch_size, h, w, f_map_num = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        gram_matrix = torch.mm(features, features.t())
        return gram_matrix.div(batch_size * h * w * f_map_num)

    def _get_style_model_and_losses(self, style_img, content_img):
        cnn = copy.deepcopy(self._cnn)
        normalization = Normalization(
            self._cnn_normalization_mean,
            self._cnn_normalization_std
        ).to(self._device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            if name in self._content_layers_default:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self._style_layers_default:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def _get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def _run_style_transfer(
            self,
            content_img,
            style_img,
            input_img,
            num_steps=500,
            style_weight=100000,
            content_weight=1
    ):
        """Run the style transfer."""
        logger.debug("Run model: (NST). Building the style transfer model...")
        model, style_losses, content_losses = self._get_style_model_and_losses(style_img, content_img)
        optimizer = self._get_input_optimizer(input_img)
        logger.debug("Run model: (NST). Optimizing...")
        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss

                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    logger.debug("Run model: (NST). run (%s):", run)
                    logger.debug(
                        "Run model: (NST). Style Loss : (%s) Content Loss: (%s)",
                        round(style_score.item(), 4),
                        round(content_score.item(), 4)
                    )

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img

    def predict(self, content_image_pth: str, style_image_pth: str, save_path: str) -> None:
        logger.debug("Start model: (NST) prediction")
        logger.debug("Model: (NST). Start load and prepare content image with path: (%s)", content_image_pth)
        content_image = self._image_loader(content_image_pth)
        logger.debug(
            "Model: (NST). Successfully load and prepare content image with shapes: (%s)",
            content_image.shape
        )
        logger.debug("Model: (NST). Start load and prepare style image with path: (%s)", style_image_pth)
        style_image = self._image_loader(style_image_pth)
        logger.debug(
            "Model: (NST). Successfully load and prepare style image with shapes: (%s)",
            style_image.shape
        )
        input_image = content_image.clone()
        logger.debug("Model: (NST). Start prepare stylized image")
        stylized_image = self._run_style_transfer(
            content_image,
            style_image,
            input_image,
            num_steps=NUM_STEPS
        )
        logger.debug("Model: (NST). Successfully prepare stylized image")
        save_image(stylized_image, save_path)
        logger.debug("Model: (NST). Successfully saved stylized image along the way: (%s)", save_path)
