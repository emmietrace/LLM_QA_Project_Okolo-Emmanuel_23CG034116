import os
import string
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------------------------
# PART A: PYTHON CLI APPLICATION
# -------------------------------------------------------------------
# This script fulfills the requirement to build a CLI that:
# 1. Accepts natural language questions.
# 2. Performs preprocessing (lowercase, punctuation removal, tokenization).
# 3. Sends the prompt to the Gemini LLM API.
# 4. Displays the final answer.
# -------------------------------------------------------------------

# Load environment variables from .env file (if available)
load_dotenv()

# --- CONFIGURATION ---
# NOTE: ensure GEMINI_API_KEY is set in your environment or .env file
API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL_NAME = "gemini-2.5-pro"

def configure_genai():
    """
    Configures the Google Gemini API with the provided API key.
    Returns True if successful, False otherwise.
    """
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found.")
        print("Please create a .env file or set the environment variable.")
        return False
    
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        print(f"Configuration Error: {e}")
        return False

def preprocess_text(text):
    """
    Performs the required NLP preprocessing steps:
    1. Lowercasing: Converts all text to lowercase.
    2. Punctuation Removal: Strips out special characters.
    3. Tokenization: Splits text into individual words (tokens).
    
    Returns:
        cleaned_text (str): Rejoined text for the LLM.
        tokens (list): List of strings for display.
    """
    # Step 1: Lowercasing
    text_lower = text.lower()
    
    # Step 2: Punctuation removal
    # Uses a translation table to remove all characters in string.punctuation
    cleaned_text = text_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Step 3: Tokenization
    # Splits the string by whitespace into a list of tokens
    tokens = cleaned_text.split()
    
    # We rejoin tokens for the API prompt, but return both for rubric compliance
    final_prompt = " ".join(tokens)
    
    return final_prompt, tokens

def get_llm_response(prompt):
    """
    Sends the processed prompt to the Gemini LLM and retrieves the text response.
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Return the text part of the response
        return response.text
    except Exception as e:
        return f"API Error: An error occurred while contacting the LLM.\nDetails: {e}"

def main():
    """
    Main application loop for the CLI.
    """
    if not configure_genai():
        return

    # Display Welcome Banner
    print("="*70)
    print(f"NLP QUESTION-AND-ANSWERING SYSTEM (CLI)")
    print(f"Model: {MODEL_NAME}")
    print("="*70)
    print("Instructions: Type your question below. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            # Input handling
            print("\n" + "-"*30)
            user_input = input("User Question: ")
            
            # Check for exit condition
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting application. Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # --- RUBRIC REQUIREMENT: PREPROCESSING ---
            cleaned_text, tokens = preprocess_text(user_input)
            
            print("\n[SYSTEM] Processing your input...")
            print(f" > Lowercase & Cleaned: '{cleaned_text}'")
            print(f" > Tokenization Output: {tokens}") # Explicitly showing tokens
            
            print("[SYSTEM] Sending to LLM...", end="\r")
            
            # --- RUBRIC REQUIREMENT: LLM INTERACTION ---
            answer = get_llm_response(cleaned_text)
            
            # Clear the loading line
            print(" " * 40, end="\r")
            
            # Display Answer
            print("="*70)
            print("LLM RESPONSE:")
            print(answer)
            print("="*70)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break

if __name__ == "__main__":
    main()
