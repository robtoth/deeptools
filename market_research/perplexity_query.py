import yaml
import requests
import logging
import os
from openai import OpenAI
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_keys(file_path: str = 'api_keys.yaml') -> dict:
    """
    Load API keys from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing API keys.

    Returns:
        dict: A dictionary containing the API keys.
    """
    logging.info(f'🔑 Loading API keys from {file_path}...')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_file_path = os.path.join(script_dir, file_path)
        logging.debug(f'🗂️ Full path to API keys file: {full_file_path = }')
        with open(full_file_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info('✅ Successfully loaded API keys.')
        return config['api_keys']
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        logging.error(f'❌ Error loading API keys: {e}')
        raise ValueError(f'Failed to load API keys from {file_path}') from e

def generate_market_research_queries(company_name: str) -> list:
    """
    Generate a list of market research queries for a given company using OpenAI API.

    Args:
        company_name (str): The name of the company to research.

    Returns:
        list: A list of market research queries.
    """
    logging.info(f'🧠 Generating market research queries for: {company_name = }...')
    
    logging.info('🔑 Loading API keys...')
    api_keys = load_api_keys()
    logging.info('✅ API keys loaded.')

    logging.info('🔧 Setting up OpenAI client...')
    client = OpenAI(
        api_key=api_keys['openai'],
    )
    logging.info('✅ OpenAI client set up.')
    
    prompt = f'''Generate a list of 5 specific market research queries about the company {company_name}. 
    These queries should cover various aspects such as the company's market position, competitors, 
    recent developments, financial performance, and future prospects. 
    Format the output as a Python list of strings.'''

    logging.info('🤖 Creating chat completion...')
    try:
        logging.info('📅 Getting current date and recent months...')
        current_date = datetime.now()
        current_month_year = current_date.strftime('%B %Y')
        previous_months = [(current_date - relativedelta(months=i)).strftime('%B %Y') for i in range(1, 4)]
        date_info = f'Current month: {current_month_year}. Previous 3 months: {", ".join(previous_months)}.'
        logging.debug(f'📝 Date information: {date_info = }')

        logging.info('🤖 Creating chat completion...')
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': f'You are a helpful assistant that generates market research queries. Output the queries as a Python list of strings, without any additional formatting or explanation. Today\'s date is {current_date.strftime("%Y-%m-%d")}. {date_info}'
                },
                {
                    'role': 'user',
                    'content': f'{prompt} Include recent information, considering the following date context: {date_info}'
                }
            ],
            model='gpt-4o-mini',
        )
        logging.info('✅ Chat completion created.')
        logging.debug(f'📝 Raw chat completion response: {chat_completion = }')
        
        queries_str = chat_completion.choices[0].message.content.strip()
        logging.debug(f'📝 Stripped query string: {queries_str = }')
        
        queries = eval(queries_str)
        logging.info('✅ Successfully generated market research queries.')
        logging.debug(f'📝 Generated queries: {queries = }')
        
        return queries
    except Exception as e:
        logging.error(f'❌ Error generating market research queries: {e}')
        return []

def query_perplexity(query: str) -> str:
    """
    Query the Perplexity API and return the answer.

    Args:
        query (str): The question to ask Perplexity.

    Returns:
        str: The answer from Perplexity.
    """
    logging.info(f'🤔 Preparing to query Perplexity API with: {query = }...')
    
    api_keys = load_api_keys()
    api_key = api_keys['perplexity']

    url = 'https://api.perplexity.ai/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'llama-3.1-sonar-small-128k-chat',
        'messages': [{'role': 'user', 'content': query}]
    }

    logging.info('🌐 Sending request to Perplexity API...')
    logging.debug(f'📝 Request payload: {payload = }')
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info('✅ Received response from Perplexity API.')
    except requests.exceptions.RequestException as e:
        logging.error(f'❌ Error querying Perplexity API: {e}')
        logging.error(f'📝 Response content: {e.response.content if e.response else "No response content"}')
        return f'Error: {str(e)}'

    logging.debug(f'📊 Response status code: {response.status_code = }')
    logging.debug(f'📝 Response content: {response.content = }')
    
    data = response.json()
    answer = data['choices'][0]['message']['content']
    
    logging.info('🎉 Successfully extracted answer from API response.')
    logging.debug(f'📝 {answer = }')

    return answer


def save_to_markdown(company_name: str, queries: list, results: list):
    """
    Save the queries and results to a Markdown file.

    Args:
        company_name (str): The name of the company researched.
        queries (list): List of queries asked.
        results (list): List of results obtained from Perplexity.
    """
    logging.info(f'💾 Saving results for {company_name = } to Markdown...')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{company_name}_market_research_{timestamp}.md'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(f'# Market Research for {company_name}\n\n')
        for i, (query, result) in enumerate(zip(queries, results), 1):
            f.write(f'## Query {i}: {query}\n\n')
            f.write(f'{result}\n\n')
    
    logging.info(f'✅ Results saved to {full_path = }')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logging.error('❌ Error: Please provide a company name as a command line argument.')
        sys.exit(1)
    
    company_name = sys.argv[1]
    logging.info(f'🏢 Starting market research for: {company_name = }')
    
    logging.info('🔍 Generating market research queries...')
    queries = generate_market_research_queries(company_name)
    logging.info(f'📊 Generated {len(queries) = } market research queries.')
    
    results = []
    for i, query in enumerate(queries, 1):
        logging.info(f'🔍 Querying Perplexity with query {i}: {query = }')
        result = query_perplexity(query)
        results.append(result)
        logging.info(f'🏁 Result for query {i}:\n{result}\n')
    
    save_to_markdown(company_name, queries, results)
    logging.info('🎉 Market research completed successfully.')
