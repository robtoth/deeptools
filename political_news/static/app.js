async function fetchNews() {
    const response = await fetch('/get_news');
    const data = await response.json();
    document.getElementById('news-content').textContent = data.news;
    document.getElementById('news-content-right').textContent = data.news;
}

async function fetchQuestions(perspective) {
    const response = await fetch(`/get_questions/${perspective}`);
    const data = await response.json();
    const container = document.getElementById(`${perspective}-qa`);
    container.innerHTML = '';
    data.questions.forEach((question, index) => {
        const qaDiv = document.createElement('div');
        qaDiv.className = 'qa-pair';
        qaDiv.innerHTML = `
            <p class="question">Q: ${question}</p>
            <p class="answer" id="${perspective}-answer-${index}">Loading answer...</p>
        `;
        container.appendChild(qaDiv);
        fetchAnswer(perspective, index);
    });
}

async function fetchAnswer(perspective, questionIndex) {
    const response = await fetch(`/get_answer/${perspective}/${questionIndex}`);
    const data = await response.json();
    document.getElementById(`${perspective}-answer-${questionIndex}`).textContent = `A: ${data.answer}`;
}

async function fetchFinalSummary() {
    const summaryContainer = document.getElementById('final-summary');
    summaryContainer.textContent = 'Generating final summary...';
    
    const response = await fetch('/get_final_summary');
    const data = await response.json();
    
    summaryContainer.textContent = data.summary;
}

window.onload = function() {
    fetchNews();
    fetchQuestions('liberal');
    fetchQuestions('conservative');
    fetchFinalSummary();
};
