import google.generativeai as genai
import os

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBr_CAjTXhZvW6Rbo-3qp7nyNhPQaKamRU"
genai.configure(api_key=GEMINI_API_KEY)

def generate_sentence_from_signs(sign_sequence):
    """
    Convert a sequence of sign language words into a grammatical English sentence.
    
    Args:
        sign_sequence: List of sign words, e.g., ["I", "love", "you"]
        
    Returns:
        str: Grammatical sentence
    """
    if not sign_sequence:
        return ""
    
    # If only one sign, return it as-is
    if len(sign_sequence) == 1:
        return sign_sequence[0]
    
    # Create prompt for Gemini
    signs = " ".join(sign_sequence)
    prompt = f"""Convert the following sign language words into a proper grammatical English sentence.
    
Sign words: {signs}

Rules:
- Keep the meaning of the signs
- Add necessary articles, pronouns, and grammar
- Make it natural and fluent
- Output ONLY the sentence, nothing else

Sentence:"""
    
    try:
        # Use Gemini 1.5 Flash for fast processing
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract the sentence
        sentence = response.text.strip()
        
        # Remove quotes if Gemini added them
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        
        return sentence
        
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        # Fallback: just join the signs with spaces
        return " ".join(sign_sequence)

