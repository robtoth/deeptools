class TextProcessor {
    constructor() {
        this.originalTextArea = document.getElementById('original-text');
        this.condensedTextArea = document.getElementById('condensed-text');
        this.reorganizedTextArea = document.getElementById('reorganized-text');
        this.critiqueTextArea = document.getElementById('critique-text');
        this.loadingDiv = document.getElementById('loading');
        this.progressBar = document.getElementById('progress-bar-fill');
    }

    updateProgress(progress) {
        this.progressBar.style.width = `${progress}%`;
    }

    async processText() {
        const originalText = this.originalTextArea.value;
        
        if (!originalText.trim()) {
            alert('Please enter some text first');
            return;
        }
        
        this.loadingDiv.style.display = 'block';
        this.updateProgress(0);
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: originalText }),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Set up SSE for progress updates
            const eventSource = new EventSource('/progress');
            eventSource.onmessage = (event) => {
                const progress = JSON.parse(event.data).progress;
                this.updateProgress(progress);
            };
            
            const data = await response.json();
            
            this.condensedTextArea.value = data.condensed;
            this.reorganizedTextArea.value = data.reorganized;
            this.critiqueTextArea.value = data.critique;

            eventSource.close();
            this.updateProgress(100);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the text');
        } finally {
            this.loadingDiv.style.display = 'none';
            setTimeout(() => this.updateProgress(0), 1000);
        }
    }
}

// Initialize the text processor when the document loads
document.addEventListener('DOMContentLoaded', () => {
    window.textProcessor = new TextProcessor();
});

// Global function to be called from HTML
function processText() {
    window.textProcessor.processText();
}
