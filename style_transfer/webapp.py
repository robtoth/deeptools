import logging
from flask import Flask, render_template, request, jsonify, send_file
import os
import requests
from werkzeug.utils import secure_filename
from style_transfer import run_style_transfer, load_image, stack_images_horizontally
import torch
import torchvision.models as models
import io
from PIL import Image
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load VGG19 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Famous artworks
famous_artworks = [
    ('Starry Night', 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'),
    ('The Scream', 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/1280px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg'),
    ('The Persistence of Memory', 'https://uploads6.wikiart.org/images/salvador-dali/the-persistence-of-memory-1931.jpg'),
    ('The Great Wave off Kanagawa', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Great_Wave_off_Kanagawa2.jpg/1280px-Great_Wave_off_Kanagawa2.jpg'),
    ('Girl with a Pearl Earring', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg'),
]

def download_image(url, filename):
    logging.info(f'🌐 Downloading image from {url}...')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        logging.info(f'✅ Downloaded image to {filename}.')
    else:
        logging.error(f'❌ Failed to download image from {url}. Status code: {response.status_code}')

# Download famous artworks
for i, (name, url) in enumerate(famous_artworks):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], f'famous_artwork_{i+1}.jpg')
    if not os.path.exists(filename):
        download_image(url, filename)

@app.route('/')
def index():
    logging.info('🏠 Rendering index page...')
    return render_template('index.html')

@app.route('/api/style_transfer', methods=['POST'])
def style_transfer():
    logging.info('🎨 Handling style transfer request...')
    
    if 'content' not in request.files or 'style' not in request.files:
        logging.error('❌ Missing content or style image')
        return jsonify({'error': 'Missing content or style image'}), 400
    
    content_file = request.files['content']
    style_file = request.files['style']
    
    if content_file.filename == '' or style_file.filename == '':
        logging.error('❌ No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
    
    logging.info('💾 Saving uploaded files...')
    content_file.save(content_path)
    style_file.save(style_path)
    
    logging.info('🖼️ Loading images...')
    content_img = load_image(content_path, (512, 512))
    style_img = load_image(style_path, (512, 512))
    
    output_images = []
    
    # Apply user-uploaded style
    logging.info('🎨 Running style transfer with user-uploaded style...')
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, content_img.clone(), num_steps=app.config['NUM_STEPS'])
    output_images.append(('User Style', style_img, output))
    
    # Apply famous artwork styles
    for i, (name, _) in enumerate(famous_artworks):
        artwork_path = os.path.join(app.config['UPLOAD_FOLDER'], f'famous_artwork_{i+1}.jpg')
        artwork_img = load_image(artwork_path, (512, 512))
        logging.info(f'🎨 Running style transfer with {name}...')
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, artwork_img, content_img.clone(), num_steps=app.config['NUM_STEPS'])
        output_images.append((name, artwork_img, output))
    
    logging.info('🔗 Stacking images...')
    stacked_images = [stack_images_horizontally([content_img, style_img, output]) for _, style_img, output in output_images]
    final_stacked_image = stack_images_horizontally(stacked_images)
    
    logging.info('🖼️ Converting image to bytes...')
    img_byte_arr = io.BytesIO()
    final_stacked_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    logging.info('✅ Style transfer completed.')
    return send_file(io.BytesIO(img_byte_arr), mimetype='image/png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Neural Style Transfer web application')
    parser.add_argument('--steps', type=int, default=300, help='Number of optimization steps for style transfer')
    args = parser.parse_args()
    
    app.config['NUM_STEPS'] = args.steps
    
    logging.info(f'🚀 Starting Flask application with {app.config["NUM_STEPS"]} style transfer steps...')
    app.run(debug=True)
    logging.info('👋 Flask application stopped.')
