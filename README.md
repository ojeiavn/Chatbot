# AI-Enhanced Detailed Lecture Notes Generation and Exam Preparation Tool

## Project Overview

This project is an AI-based tool specifically designed to enhance lecture note generation and exam preparation for the University of Warwick's Computer Science curriculum. Using Retrieval-Augmented Generation (RAG) techniques, this tool retrieves real-time data from module-specific content, ensuring accurate and contextually relevant responses tailored to students' academic needs.

## Problem Statement

The increasing use of generative AI tools by university students has highlighted the limitations of these models in providing accurate, detailed responses based on specific academic content. Many existing models, like ChatGPT and Google Gemini, are not fine-tuned for university-specific content, leading to generic responses and potential misinformation. This project addresses these gaps by integrating module-specific data into a generative AI model.

## Objectives

1. **Lecture Content Interaction**: Allow users to input lecture slides and transcripts for deeper understanding.
2. **Lecture Notes Generation**: Automatically generate comprehensive lecture notes based on the provided content.
3. **Exam Preparation**: Retrieve past exam questions and provide feedback related to them.
4. **Multimodal Input Processing**: Enhance the tool to process various input formats, including images and potentially video.

## Methodology

The development follows an iterative approach, incorporating user feedback after each sprint. Key phases include:
- Research and literature review on model adaptation and data collection.
- Agile development cycles focusing on core features, with User Acceptance Testing (UAT) after each iteration.

## Getting Started

### Prerequisites

- Python 3.x
- Streamlit
- LangChain
- dotenv
- Additional dependencies (see below)

### Dependencies

To set up the project environment, install the following packages using pip:

```bash
pip install streamlit langchain langchain_community langchain_openai python-dotenv
```

### Environment Variables

You will need to create a `.env` file in the root of your project directory. This file should contain your API keys and other sensitive information required for the application.

**Example of `.env` file:**
```plaintext
OPENAI_API_KEY=your_openai_api_key
```

### Usage

1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd [your-project-directory]
   ```

2. Ensure you have your `.env` file set up with the required API keys.

3. Run the application using Streamlit:
   ```bash
   streamlit run [your_script.py]
   ```

4. Access the application in your web browser at `http://localhost:8501`.

## Features

- **Lecture Note Generation**: The tool generates detailed notes from provided lecture slides and transcripts.
- **Content Summarization**: Users can query specific lecture topics, and the model provides clear, concise summaries.
- **Exam Preparation Assistance**: Retrieval of past exam questions related to lecture content, with options for feedback and guided study.

## Conclusion

This project aims to enhance students' access to accurate, detailed academic content, providing valuable tools for study and exam preparation. User feedback will be essential to refining the tool throughout development.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Special thanks to the University of Warwick for their support and resources for this project.

