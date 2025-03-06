"""
Mistral Code Reviewer - Get AI feedback on your code using Mistral AI with streaming output
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def get_language(file_path):
    """Get language from file extension."""
    ext = Path(file_path).suffix.lower()
    return {'.py': 'python', '.js': 'javascript', '.java': 'java', 
            '.html': 'html', '.css': 'css', '.cpp': 'cpp', 
            '.go': 'go', '.ts': 'typescript'}.get(ext, 'code')

def review_code_streaming(code, language):
    """Review code using Mistral AI with streaming output."""
    endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    messages = [
        {"role": "system", "content": f"You are an expert {language} code reviewer. Be concise but thorough."},
        {"role": "user", "content": f"Review this {language} code for bugs, improvements, and best practices:\n\n```{language}\n{code}\n```"}
    ]
    
    payload = {
        "model": "codestral-latest",
        "messages": messages,
        "temperature": 0.2,
        "stream": True  # Enable streaming
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        print("\n--- CODE REVIEW ---")
        
        # Collect the complete response while streaming
        full_response = []
        
        for chunk in response.iter_lines():
            if chunk:
                line = chunk.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                if line == '[DONE]':
                    break
                    
                try:
                    chunk_data = json.loads(line)
                    content = chunk_data['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end="", flush=True)
                        time.sleep(0.01)  # Small delay for typing effect
                        full_response.append(content)
                except json.JSONDecodeError:
                    pass
        
        print("\n")  # Final newline
        return ''.join(full_response)
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python samples/mistral-code-reviewer.py <code_file>")
        return
    
    try:
        file_path = sys.argv[1]
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        language = get_language(file_path)
        print(f"Reviewing {language} code in: {file_path}")
        
        review_code_streaming(code, language)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()