import os
import string
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# Get API Key from Environment Variable (Best practice for Render)
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"

# Configure API immediately
if API_KEY:
    genai.configure(api_key=API_KEY)

def preprocess_text(text):
    """Same preprocessing logic as CLI."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not API_KEY:
        return jsonify({'error': 'API Key is missing on the server.'}), 500

    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # 1. Preprocess
    processed_question = preprocess_text(question)

    try:
        # 2. Call LLM
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(processed_question)
        answer = response.text
        
        # 3. Return JSON
        return jsonify({
            'original': question,
            'processed': processed_question,
            'answer': answer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 for Render support
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)