"""
Mistral Content Creator - Generate blog posts and social media content using Mistral AI with streaming output
"""
import os
import json
import time
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def generate_content_streaming(topic, content_type="blog", tone="professional"):
    """Generate content using Mistral AI with streaming output."""
    endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    # Define the prompt based on content type
    prompts = {
        "blog": f"Write a blog post about {topic}. Keep it concise but informative.",
        "social": f"Create a set of 3 social media posts about {topic} for Twitter, LinkedIn, and Instagram.",
        "email": f"Write a short marketing email about {topic} with a subject line and call to action."
    }
    
    # Define the tone instruction
    tones = {
        "professional": "Use a professional and authoritative tone.",
        "friendly": "Use a warm, conversational tone.",
        "persuasive": "Use a persuasive tone that creates urgency."
    }
    
    prompt = f"{prompts.get(content_type, prompts['blog'])}\n{tones.get(tone, tones['professional'])}"
    
    messages = [
        {"role": "system", "content": "You are an expert content creator."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": "mistral-medium-latest",
        "messages": messages,
        "temperature": 0.7,
        "stream": True  # Enable streaming
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        print(f"\nGenerating {tone} {content_type} content about '{topic}'...")
        print("\n--- CONTENT ---")
        
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
    parser = argparse.ArgumentParser(description="Generate content using Mistral AI")
    parser.add_argument("topic", help="Topic for the content")
    parser.add_argument("--type", choices=["blog", "social", "email"], default="blog", help="Content type")
    parser.add_argument("--tone", choices=["professional", "friendly", "persuasive"], default="professional", help="Content tone")
    
    args = parser.parse_args()
    
    content = generate_content_streaming(args.topic, args.type, args.tone)
    
    if content:
        # Save the content to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.type}_{args.topic.replace(' ', '_')}_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\nContent saved to: {filename}")

if __name__ == "__main__":
    main()