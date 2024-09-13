import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
from torchvision.utils import save_image
import requests
from requests.exceptions import RequestException
import tempfile
import os
from urllib.parse import urlparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'🖥️ Using device: {device}')

# Desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128
logging.debug(f'🖼️ Image size set to: {imsize}')

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

def image_loader(image_name):
    logging.info(f'🖼️ Loading image: {image_name}...')
    image = Image.open(image_name).convert('RGB')
    logging.debug(f'📏 Original image size: {image.size}')
    
    # Define a sequence of transforms
    transform = transforms.Compose([
        transforms.Resize(imsize + 32),  # Resize to slightly larger
        transforms.CenterCrop(imsize),   # Then center crop
        transforms.ToTensor()
    ])
    
    image = transform(image).unsqueeze(0)
    logging.info(f'✅ Loaded and processed image: {image_name}')
    logging.debug(f'🔢 Tensor shape: {image.shape}')
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        logging.debug(f'🎯 Content target shape: {self.target.shape}')

    def forward(self, input):
        # Ensure input and target have the same size
        if input.size() != self.target.size():
            logging.warning(f'⚠️ Input size {input.size()} does not match target size {self.target.size()}. Resizing input.')
            input = F.interpolate(input, size=self.target.size()[2:], mode='bilinear', align_corners=False)
        self.loss = F.mse_loss(input, self.target)
        logging.debug(f'📊 Content loss: {self.loss.item()}')
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    logging.debug(f'📐 Gram matrix input size: {input.size()}')
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        logging.debug(f'🎨 Style target shape: {self.target.shape}')

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        logging.debug(f'📊 Style loss: {self.loss.item()}')
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        logging.debug(f'🔢 Normalization mean: {self.mean}, std: {self.std}')

    def forward(self, img):
        # Expand mean and std to match the number of channels in the input image
        mean = self.mean.expand(img.size(1), -1, -1)
        std = self.std.expand(img.size(1), -1, -1)
        return (img - mean) / std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    logging.info('🏗️ Building the style transfer model...')
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)
        logging.debug(f'➕ Added layer: {name}')

        if name in content_layers:
            target = model(content_img).detach()
            logging.debug(f'Content target shape at {name}: {target.shape}')
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)
            logging.debug(f'➕ Added content loss at layer: {name}')

        if name in style_layers:
            target_feature = model(style_img).detach()
            logging.debug(f'Style target shape at {name}: {target_feature.shape}')
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)
            logging.debug(f'➕ Added style loss at layer: {name}')

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    logging.info(f'✅ Model built with {len(model)} layers')

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    logging.info('🏃 Starting style transfer process...')
    logging.debug(f'Content image shape: {content_img.shape}')
    logging.debug(f'Style image shape: {style_img.shape}')
    logging.debug(f'Input image shape: {input_img.shape}')
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    logging.info('🔄 Running optimization...')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

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
                logging.info(f'🔄 Run {run[0]}:')
                logging.info(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    logging.info('✅ Style transfer complete!')
    return input_img

def stack_images_horizontally(images):
    logging.info('🔗 Stacking images horizontally...')
    widths, heights = zip(*(i.size for i in images))
    max_height = max(heights)
    total_width = sum(widths)
    
    new_image = Image.new('RGB', (total_width, max_height))
    
    logging.debug(f'📏 New image dimensions: {new_image.size}')
    
    x_offset = 0
    for img in images:
        logging.debug(f'📍 Pasting image at x_offset: {x_offset}')
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    logging.info('✅ Images stacked horizontally.')
    return new_image

def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor.cpu().clone().squeeze(0))

def run_style_transfer_process(content_path, style_path, output_path):
    logging.info('🚀 Starting style transfer process...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'🖥️ Using device: {device}')

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    logging.info('✅ Loaded VGG19 model')

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    # Convert tensor images to PIL images
    content_pil = tensor_to_pil(content_img)
    style_pil = tensor_to_pil(style_img)
    output_pil = tensor_to_pil(output)

    # Stack images horizontally
    stacked_image = stack_images_horizontally([content_pil, style_pil, output_pil])

    logging.info(f'💾 Saving stacked output image to {output_path}...')
    stacked_image.save(output_path)
    logging.info(f'✅ Stacked output image saved to {output_path}')

def download_image(url, filename):
    logging.info(f'📥 Attempting to download image from {url}...')
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type')
        if 'image' not in content_type.lower():
            raise ValueError(f'URL does not point to an image. Content-Type: {content_type}')
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        logging.info(f'✅ Image successfully downloaded and saved as {filename}')
    except RequestException as e:
        parsed_url = urlparse(url)
        error_msg = f'❌ Failed to download image from {parsed_url.netloc}. Error: {str(e)}'
        if isinstance(e, requests.HTTPError):
            error_msg += f' Status code: {e.response.status_code}'
        elif isinstance(e, requests.Timeout):
            error_msg += ' Request timed out.'
        elif isinstance(e, requests.ConnectionError):
            error_msg += ' Connection error. Please check your internet connection.'
        logging.error(error_msg)
        raise Exception(error_msg)
    except ValueError as e:
        logging.error(f'❌ {str(e)}')
        raise
    except Exception as e:
        logging.error(f'❌ An unexpected error occurred while downloading the image: {str(e)}')
        raise

def run_multiple_style_transfers(content_path, style_paths, output_folder):
    logging.info('🚀 Starting multiple style transfers...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'🖥️ Using device: {device}')

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    logging.info('✅ Loaded VGG19 model')

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_img = image_loader(content_path)
    
    output_paths = []
    for i, style_path in enumerate(style_paths):
        logging.info(f'🎨 Processing style {i+1}/{len(style_paths)}...')
        style_img = image_loader(style_path)
        input_img = content_img.clone()

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img)

        # Convert tensor images to PIL images
        content_pil = tensor_to_pil(content_img)
        style_pil = tensor_to_pil(style_img)
        output_pil = tensor_to_pil(output)

        # Stack images horizontally
        stacked_image = stack_images_horizontally([content_pil, style_pil, output_pil])

        output_filename = f'output_{i+1}.jpg'
        output_path = os.path.join(output_folder, output_filename)
        logging.info(f'💾 Saving stacked output image to {output_path}...')
        stacked_image.save(output_path)
        logging.info(f'✅ Stacked output image saved to {output_path}')
        output_paths.append(output_filename)

    logging.info('✅ All style transfers complete!')
    return output_paths
