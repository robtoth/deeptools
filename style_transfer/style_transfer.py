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
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def load_image(image_path, imsize=None):
    logging.info(f'🖼️ Loading image from {image_path}...')
    image = Image.open(image_path).convert('RGB')
    
    if imsize:
        image = transforms.Resize(imsize)(image)
    
    loader = transforms.ToTensor()
    image = loader(image).unsqueeze(0)
    
    logging.info(f'✅ Loaded image from {image_path}.')
    logging.debug(f'📊 {image.size() = }')
    return image.to(device, torch.float)

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

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    logging.info('✅ Style transfer model built.')
    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    logging.info('🎨 Running style transfer...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

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
                logging.info(f'🔄 Iteration: {run[0]}')
                logging.info(f'📊 Style Loss: {style_score.item():4f}, Content Loss: {content_score.item():4f}')

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    logging.info('✅ Style transfer completed.')
    return input_img

def stack_images_horizontally(images):
    logging.info('🔗 Stacking images horizontally...')
    # Convert tensors to PIL images if necessary
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pil_images.append(transforms.ToPILImage()(img.squeeze(0).cpu().clamp(0, 1)))
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise TypeError(f'Unsupported image type: {type(img)}')
    
    # Get the maximum height
    max_height = max(img.size[1] for img in pil_images)
    
    # Resize images to have the same height
    resized_images = [img.resize((int(img.size[0] * max_height / img.size[1]), max_height)) for img in pil_images]
    
    # Calculate total width
    total_width = sum(img.size[0] for img in resized_images)
    
    # Create a new image
    stacked_image = Image.new('RGB', (total_width, max_height))
    
    # Paste images
    x_offset = 0
    for img in resized_images:
        stacked_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    logging.info('✅ Stacked images horizontally.')
    return stacked_image

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('🚀 Starting neural style transfer process...')
    
    parser = argparse.ArgumentParser(description='Perform neural style transfer')
    parser.add_argument('--content', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style', type=str, required=True, help='Path to the style image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save the output image')
    parser.add_argument('--steps', type=int, default=300, help='Number of optimization steps')
    parser.add_argument('--size', type=int, default=512, help='Size to resize images to')
    
    logging.info('🔍 Parsing command line arguments...')
    args = parser.parse_args()
    logging.info('✅ Parsed command line arguments.')
    
    logging.debug(f'📁 {args.content = }')
    logging.debug(f'🎨 {args.style = }')
    logging.debug(f'💾 {args.output = }')
    logging.debug(f'🔢 {args.steps = }')
    logging.debug(f'📏 {args.size = }')
    
    # Load and resize images to the same size
    content_img = load_image(args.content, (args.size, args.size))
    style_img = load_image(args.style, (args.size, args.size))
    input_img = content_img.clone()
    
    # Add these debug logs
    logging.debug(f'📊 Content image shape: {content_img.shape}')
    logging.debug(f'📊 Style image shape: {style_img.shape}')
    logging.debug(f'📊 Input image shape: {input_img.shape}')
    
    # Ensure all images have 3 channels and the same size
    assert content_img.shape == style_img.shape == input_img.shape, "Images must have the same shape"
    assert content_img.shape[1] == 3, "Images must have 3 channels (RGB)"
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=args.steps)
    
    # Stack images horizontally
    stacked_image = stack_images_horizontally([content_img, style_img, output])

    logging.info(f'💾 Saving stacked output image to {args.output}...')
    stacked_image.save(args.output)
    logging.info(f'✅ Saved stacked output image to {args.output}.')

    logging.info('✨ Neural style transfer process completed.')

if __name__ == '__main__':
    main()
