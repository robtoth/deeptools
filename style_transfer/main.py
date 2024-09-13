import logging
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from style_transfer import run_style_transfer_process, run_multiple_style_transfers, download_image
import tempfile
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Add this list of famous artworks
FAMOUS_ARTWORKS = [
    ('Starry Night', 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'),
    ('The Scream', 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/1280px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg'),
    ('The Persistence of Memory', 'https://uploads6.wikiart.org/images/salvador-dali/the-persistence-of-memory-1931.jpg'),
    ('The Great Wave off Kanagawa', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Great_Wave_off_Kanagawa2.jpg/1280px-Great_Wave_off_Kanagawa2.jpg'),
    ('Girl with a Pearl Earring', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg'),
]

@app.route('/')
def home():
    logging.info('🏠 Serving home page...')
    return render_template('index.html')

@app.route('/api/style_transfer', methods=['POST'])
def style_transfer():
    logging.info('🎨 Received style transfer request...')
    
    if 'content' not in request.files or 'style' not in request.files:
        logging.error('❌ Missing content or style image')
        return jsonify({'error': 'Missing content or style image'}), 400
    
    content_file = request.files['content']
    user_style_file = request.files['style']
    
    if content_file.filename == '' or user_style_file.filename == '':
        logging.error('❌ No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
    user_style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(user_style_file.filename))
    
    logging.info('💾 Saving uploaded files...')
    content_file.save(content_path)
    user_style_file.save(user_style_path)
    
    # Download famous artworks
    with tempfile.TemporaryDirectory() as temp_dir:
        style_paths = [user_style_path]
        failed_downloads = []
        for name, url in FAMOUS_ARTWORKS:
            temp_file = os.path.join(temp_dir, f'{name}.jpg')
            try:
                download_image(url, temp_file)
                style_paths.append(temp_file)
            except Exception as e:
                logging.error(f'Failed to download {name}: {str(e)}')
                failed_downloads.append({'name': name, 'error': str(e)})
        
        if failed_downloads:
            error_message = "Some artworks failed to download:\n" + "\n".join([f"{d['name']}: {d['error']}" for d in failed_downloads])
            logging.warning(error_message)
        
        if not style_paths:
            logging.error('❌ No style images available for transfer')
            return jsonify({'error': 'No style images available for transfer'}), 500
        
        logging.info('🏃 Running multiple style transfers...')
        try:
            output_filenames = run_multiple_style_transfers(content_path, style_paths, app.config['OUTPUT_FOLDER'])
        except Exception as e:
            logging.error(f'❌ Error during style transfer: {str(e)}')
            return jsonify({'error': f'Style transfer failed: {str(e)}'}), 500
    
    logging.info('✅ All style transfers complete')
    return jsonify({
        'outputs': output_filenames,
        'failed_downloads': failed_downloads
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logging.info(f'📁 Serving uploaded file: {filename}...')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    logging.info(f'📁 Serving output file: {filename}...')
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    logging.info('🚀 Starting Flask app...')
    app.run(debug=True)
