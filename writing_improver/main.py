import logging
import yaml
import nltk
from typing import Dict, Optional, List, Tuple
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
from tqdm import tqdm
import time

# Download required NLTK data
nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_critic_prompt() -> str:
    """Load the critic prompt from file."""
    logger.info('📖 Loading critic prompt...')
    try:
        with open('critic.txt', 'r') as file:
            prompt = file.read()
            logger.info('✅ Critic prompt loaded successfully')
            return prompt
    except Exception as e:
        logger.error(f'❌ Failed to load critic prompt: {e}')
        raise

def load_api_keys() -> Dict[str, Dict[str, str]]:
    """Load API keys from yaml file."""
    logger.info('🔑 Loading API keys...')
    try:
        with open('api_keys.yaml', 'r') as file:
            keys = yaml.safe_load(file)
            logger.info('✅ API keys loaded successfully')
            return keys['llm_api_keys']
    except Exception as e:
        logger.error(f'❌ Failed to load API keys: {e}')
        raise

def create_openai_client() -> OpenAI:
    """Create and return OpenAI client."""
    logger.info('🤖 Initializing OpenAI client...')
    api_keys = load_api_keys()
    client = OpenAI(api_key=api_keys['openai']['api_key'])
    logger.info('✅ OpenAI client initialized')
    return client

def get_sentences(text: str) -> List[str]:
    """Extract sentences from text using NLTK."""
    return nltk.sent_tokenize(text)

def estimate_chunk_size(sentences: List[str]) -> int:
    """
    Estimate optimal number of sentences per chunk based on average sentence length.
    
    Args:
        sentences: List of sentences from the text
    
    Returns:
        Number of sentences per chunk
    """
    if not sentences:
        return 5  # Default fallback
        
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
    logger.debug(f'📏 {avg_sentence_length = }')
    
    # Target ~1500 characters per chunk
    # This is slightly lower than before because sentence boundaries 
    # give us more natural breaks
    target_chunk_size = 1500
    optimal_sentences = max(3, min(10, int(target_chunk_size / avg_sentence_length)))
    
    logger.info(f'📏 Estimated optimal chunk size: {optimal_sentences} sentences')
    return optimal_sentences

def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks based on sentences.
    
    Args:
        text: The input text to chunk
    
    Returns:
        List of text chunks
    """
    logger.info('📝 Starting text chunking process')
    
    # Get sentences using NLTK
    sentences = get_sentences(text)
    logger.info(f'📊 Found {len(sentences)} sentences')
    
    # Calculate optimal chunk size
    sentences_per_chunk = estimate_chunk_size(sentences)
    
    # Create chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        
        # Check if we should create a new chunk
        if len(current_chunk) >= sentences_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    # Add any remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f'✅ Text split into {len(chunks)} chunks')
    return chunks

def process_text_with_openai(client: OpenAI, text: str, instruction: str) -> Optional[str]:
    """Process text using OpenAI API with given instruction."""
    logger.info(f'🔄 Processing text chunk with instruction: {instruction}')
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]
        )
        processed_text = response.choices[0].message.content
        logger.debug('✅ Chunk processed successfully')
        return processed_text
    except Exception as e:
        logger.error(f'❌ Error processing text chunk: {e}')
        return None

def process_text_in_chunks(client: OpenAI, text: str, instruction: str) -> str:
    """Process text in chunks and combine results."""
    logger.info('🔄 Starting chunked text processing')
    
    # Split text into chunks
    chunks = chunk_text(text)
    processed_chunks = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks, 1):
        logger.info(f'📝 Processing chunk {i} of {len(chunks)}')
        processed_chunk = process_text_with_openai(client, chunk, instruction)
        if processed_chunk:
            chunk_header = f"\n\n--------CHUNK {i}--------\n\n"
            processed_chunks.append(chunk_header + processed_chunk)
    
    # Combine processed chunks
    result = ''.join(processed_chunks)
    logger.info('✅ Completed processing all chunks')
    return result

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def recursive_condense(client: OpenAI, text: str, target_words: int = 500) -> Tuple[str, List[str]]:
    """
    Recursively condense text until target word count is reached.
    Returns the final text and a list of intermediate versions.
    """
    logger.info(f'🔄 Starting recursive condensing to {target_words} words')
    current_text = text
    versions = [text]
    
    while count_words(current_text) > target_words:
        chunks = chunk_text(current_text)
        processed_chunks = []
        
        for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
            processed_chunk = process_text_with_openai(
                client,
                chunk,
                "Condense this text into half the number of words while maintaining key information:"
            )
            if processed_chunk:
                processed_chunks.append(processed_chunk)
        
        current_text = ' '.join(processed_chunks)
        versions.append(current_text)
        
        logger.info(f'📊 Current word count: {count_words(current_text)}')
    
    return current_text, versions

# Initialize Flask app and load resources
app = Flask(__name__)
openai_client = create_openai_client()
critic_prompt = load_critic_prompt()

@app.route('/')
def index():
    """Render the main page."""
    logger.info('📱 Rendering index page')
    return render_template('index.html')

@app.route('/progress')
def progress():
    """SSE endpoint for progress updates."""
    def generate():
        for i in range(0, 101, 5):
            time.sleep(0.1)  # Simulate processing time
            yield f"data: {{'progress': {i}}}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/process', methods=['POST'])
def process():
    """Process text through OpenAI API."""
    logger.info('🔄 Processing text request received')
    
    data = request.json
    original_text = data.get('text', '')
    
    # Process condensed text recursively
    condensed_text, versions = recursive_condense(openai_client, original_text)
    
    # Process other columns with progress bars
    with tqdm(total=2, desc="Processing additional columns") as pbar:
        reorganized_text = process_text_in_chunks(
            openai_client,
            original_text,
            "Reorganize the following text to be more coherent and well-structured:"
        )
        pbar.update(1)
        
        critique_instruction = critic_prompt + original_text
        critique_text = process_text_in_chunks(
            openai_client,
            original_text,
            critique_instruction
        )
        pbar.update(1)
    
    return jsonify({
        'condensed': condensed_text,
        'condensed_versions': versions,
        'reorganized': reorganized_text,
        'critique': critique_text
    })

if __name__ == '__main__':
    logger.info('🚀 Starting Flask application...')
    app.run(debug=True)
