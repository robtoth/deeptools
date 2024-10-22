import aiohttp
import asyncio
from openai import AsyncOpenAI
import os
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """
    Load configuration from YAML file.
    
    Returns:
        dict: Configuration dictionary.
    """
    logger.info('📂 Loading configuration from YAML file')
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info('✅ Configuration loaded successfully')
        return config
    except Exception as e:
        logger.error(f'❌ Error loading configuration: {e}')
        raise

# Load configuration
config = load_config()

# Set up Perplexity client
perplexity_client = AsyncOpenAI(api_key=config['perplexity']['api_key'], base_url="https://api.perplexity.ai")

# Set up OpenAI client
openai_client = AsyncOpenAI(api_key=config['openai']['api_key'])

async def get_political_news() -> str:
    """
    Get today's political news using Perplexity API asynchronously.
    
    Returns:
        str: A summary of today's political news.
    """
    logger.info('🗞️ Fetching today\'s political news using Perplexity API')
    try:
        response = await perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise summaries of current political news."},
                {"role": "user", "content": "What's today's political news?"}
            ]
        )
        news_summary = response.choices[0].message.content.strip()
        logger.info('✅ Political news summary generated')
        return news_summary
    except Exception as e:
        logger.error(f'❌ Error fetching political news: {str(e)}')
        return "Unable to fetch today's political news. Please try again later."

async def generate_questions(perspective: str, news: str) -> list:
    """
    Generate questions about the news from a specific perspective using ChatGPT asynchronously.
    
    Args:
        perspective (str): 'liberal' or 'conservative'
        news (str): The news summary
    
    Returns:
        list: A list of 5 questions
    """
    logger.info(f'🤔 Generating {perspective} questions using ChatGPT')
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {perspective} political analyst. Generate exactly 5 questions about the following news from a {perspective} perspective. Format your response as a numbered list, with each question on a new line, prefixed by 'Q1:', 'Q2:', 'Q3:', 'Q4:', and 'Q5:'. Do not include any additional text or explanations."},
                {"role": "user", "content": news}
            ]
        )
        raw_response = response.choices[0].message.content.strip()
        
        # Split the response into lines and process each line
        questions = []
        for line in raw_response.split('\n'):
            if line.strip().startswith(('Q1:', 'Q2:', 'Q3:', 'Q4:', 'Q5:')):
                question = line.split(':', 1)[1].strip()
                questions.append(question)
        
        logger.info(f'✅ {perspective.capitalize()} questions generated')
        
        # Ensure we have exactly 5 questions
        if len(questions) < 5:
            questions.extend([f"Additional {perspective} question {i+1}" for i in range(5 - len(questions))])
        elif len(questions) > 5:
            questions = questions[:5]
        
        return questions
    except Exception as e:
        logger.error(f'❌ Error generating {perspective} questions: {str(e)}')
        return [f"Unable to generate {perspective} question {i+1}" for i in range(5)]

async def get_perplexity_answer(question: str) -> str:
    """
    Get an answer to a question using Perplexity API asynchronously.
    
    Args:
        question (str): The question to ask
    
    Returns:
        str: The answer from Perplexity
    """
    logger.info(f'🤖 Getting answer from Perplexity API for question: {question}')
    try:
        response = await perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise answers to political questions."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content.strip()
        logger.info('✅ Answer generated from Perplexity')
        return answer
    except Exception as e:
        logger.error(f'❌ Error getting answer from Perplexity: {str(e)}')
        return "Unable to get an answer at this time. Please try again later."

async def process_questions_with_answers(questions: list) -> list:
    """
    Process a list of questions by getting answers from Perplexity for each asynchronously.
    
    Args:
        questions (list): List of questions
    
    Returns:
        list: List of dictionaries containing questions and their answers
    """
    tasks = [get_perplexity_answer(question) for question in questions]
    answers = await asyncio.gather(*tasks)
    return [{"question": q, "answer": a} for q, a in zip(questions, answers)]

# Add this new function at the end of the file

async def generate_final_summary(liberal_qa: list, conservative_qa: list) -> str:
    """
    Generate a final summary of the news and answers from both perspectives.
    
    Args:
        liberal_qa (list): List of liberal Q&A pairs
        conservative_qa (list): List of conservative Q&A pairs
    
    Returns:
        str: A summary of today's news and perspectives
    """
    logger.info('📊 Generating final summary')
    try:
        combined_content = "Liberal perspective:\n"
        for qa in liberal_qa:
            combined_content += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        combined_content += "\nConservative perspective:\n"
        for qa in conservative_qa:
            combined_content += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a neutral political analyst. Summarize the key points from both liberal and conservative perspectives based on the provided Q&A pairs. Highlight areas of agreement and disagreement."},
                {"role": "user", "content": combined_content}
            ]
        )
        summary = response.choices[0].message.content.strip()
        logger.info('✅ Final summary generated')
        return summary
    except Exception as e:
        logger.error(f'❌ Error generating final summary: {str(e)}')
        return "Unable to generate a final summary at this time. Please try again later."
