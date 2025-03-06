This is a template project for Mistral AI applications in [Codeanywhere](https://codeanywhere.com/). [Try it out](https://app.codeanywhere.com/#https://github.com/codeanywhere-templates/mistral-ai)

## Getting Started

Set your Mistral API key:
```bash 
export MISTRAL_API_KEY=your_api_key_here
```

Open the terminal and run one of the example scripts:

```bash
python samples/mistral-document-analyzer.py util/report.txt
python samples/mistral-code-reviewer.py util/script.js
python samples/mistral-content-creator.py "AI Applications" --type blog --tone professional
```

All dependencies are pre-installed in the devcontainer. You only need to create a `.env` file with your API key:

```
MISTRAL_API_KEY=your_api_key_here
```

## Features

- Development container for Mistral AI applications
- Pre-installed dependencies
- Document analysis and summarization tools
- Code review and improvement utilities
- Content generation capabilities

## Usage Examples

### Document Analysis

```bash
# Summarize a document
python samples/mistral-document-analyzer.py util/report.txt

# Summarize and save to a file
python samples/mistral-document-analyzer.py util/report.txt summary.txt
```

### Code Review

```bash
# Review JavaScript code
python samples/mistral-code-reviewer.py util/script.js
```

### Content Creation

```bash
# Create a blog post
python samples/mistral-content-creator.py "Machine Learning Trends" --type blog

# Create social media content with a friendly tone
python samples/mistral-content-creator.py "Product Launch" --type social --tone friendly

# Create a marketing email with a persuasive tone
python samples/mistral-content-creator.py "Summer Sale" --type email --tone persuasive
```

## Learn More

To learn more about Mistral AI, take a look at the following resources:

- [Mistral AI Documentation](https://docs.mistral.ai/) - learn about Mistral AI features and API
- [Mistral AI Console](https://console.mistral.ai/) - try out models in your browser
- [La Plateforme](https://docs.mistral.ai/platform/overview/) - learn about Mistral's platform
- [Mistral API Reference](https://docs.mistral.ai/api/) - detailed API documentation

You can check out [the official Mistral AI GitHub repository](https://github.com/mistralai/mistral-python) - your feedback and contributions are welcome!

## Want to contribute?

Feel free to [open a PR](https://github.com/codeanywhere-templates/mistral-ai) with any suggestions for this template project 😃
