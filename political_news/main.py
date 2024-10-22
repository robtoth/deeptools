from flask import Flask, render_template, jsonify
from flask_cors import CORS
import logging
from api_requests import get_political_news, generate_questions, get_perplexity_answer, process_questions_with_answers, generate_final_summary
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    logger.info('🏠 Rendering index page')
    return render_template('index.html')

@app.route('/get_news')
async def get_news():
    news = await get_political_news()
    return jsonify({"news": news})

@app.route('/get_questions/<perspective>')
async def get_questions(perspective):
    news = await get_political_news()
    questions = await generate_questions(perspective, news)
    return jsonify({"questions": questions})

@app.route('/get_answer/<perspective>/<int:question_index>')
async def get_answer(perspective, question_index):
    news = await get_political_news()
    questions = await generate_questions(perspective, news)
    if 0 <= question_index < len(questions):
        answer = await get_perplexity_answer(questions[question_index])
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Invalid question index"}), 400

@app.route('/get_final_summary')
async def get_final_summary():
    news = await get_political_news()
    liberal_questions = await generate_questions('liberal', news)
    conservative_questions = await generate_questions('conservative', news)
    
    liberal_qa = await process_questions_with_answers(liberal_questions)
    conservative_qa = await process_questions_with_answers(conservative_questions)
    
    summary = await generate_final_summary(liberal_qa, conservative_qa)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    logger.info('🚀 Starting Flask application')
    app.run(debug=True)
    logger.info('✅ Flask application stopped')
