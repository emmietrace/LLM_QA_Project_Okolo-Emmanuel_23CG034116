import os
import re
import string
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (optional, if using a .env file locally)
load_dotenv()

# --- CONFIGURATION ---
# NOTE: Replace 'YOUR_API_KEY' with your actual key if not using environment variables.
API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL_NAME = "gemini-2.5-pro"  # As requested

def configure_genai():
    """Configures the Gemini API."""
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found. Please set it in your environment or script.")
        return False
    genai.configure(api_key=API_KEY)
    return True

def preprocess_text(text):
    """
    Performs basic NLP preprocessing:
    1. Lowercasing
    2. Punctuation removal
    3. Basic Tokenization (splitting by whitespace)
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenization (just for demonstration, we rejoin for the API)
    tokens = text.split()
    
    # Return both the cleaned string and tokens for display
    return " ".join(tokens), tokens

def get_llm_response(prompt):
    """Sends the processed prompt to the Gemini LLM."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to LLM: {e}"

def main():
    if not configure_genai():
        return

    print("="*60)
    print(f"NLP Q&A System CLI (Model: {MODEL_NAME})")
    print("="*60)
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nUser Question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting application. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        # Preprocessing
        processed_text, tokens = preprocess_text(user_input)
        
        print("-" * 30)
        print(f"Preprocessing Info:")
        print(f"  > Lowercase & Cleaned: {processed_text}")
        print(f"  > Tokens: {tokens}")
        print("-" * 30)

        print("Querying Gemini API...", end="\r")
        
        # Get Answer
        answer = get_llm_response(processed_text)
        
        print(" " * 30, end="\r") # Clear loading text
        print(f"LLM Answer:\n{answer}")
        print("="*60)

if __name__ == "__main__":
    main()