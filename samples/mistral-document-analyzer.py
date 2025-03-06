"""
Mistral Document Analyzer - Summarize text files using Mistral AI with streaming output
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def summarize_text_streaming(text):
    """Get a summary of text using Mistral AI with streaming output."""
    endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    messages = [
        {"role": "system", "content": "You are an expert summarizer. Be concise but comprehensive."},
        {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
    ]
    
    payload = {
        "model": "mistral-medium-latest",
        "messages": messages,
        "temperature": 0.3,
        "stream": True  # Enable streaming
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        print("\n--- SUMMARY ---")
        
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
        print("Usage: python samples/mistral-document-analyzer.py <filename> [output_filename]")
        return
    
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Analyzing: {sys.argv[1]}")
        summary = summarize_text_streaming(text)
        
        # Save summary if requested
        if len(sys.argv) > 2 and summary:
            with open(sys.argv[2], 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\nSummary saved to: {sys.argv[2]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()