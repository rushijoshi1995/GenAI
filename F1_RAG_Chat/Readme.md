# Formula 1 RAG (Retrieval-Augmented Generation) Chat App

![RAG App Preview](F1_app_recording.gif)

## Description

This Formula 1 RAG Chat App is an interactive question-answering system that leverages Retrieval-Augmented Generation to provide accurate and contextual answers to queries about Formula 1 racing. The app uses a vector database to store and retrieve relevant information, employs advanced language models to refine queries and generate responses, and features a user-friendly Streamlit interface for easy interaction.

## Features

- Interactive chat interface for Formula 1 related questions
- Contextual question-answering about Formula 1 racing
- Query refinement for more accurate information retrieval
- Retrieval of relevant documents from a Chroma vector database
- Integration with OpenAI's language models for natural language processing
- Conversation history tracking for improved context awareness
- Source citation for answers

## Technologies Used

- Python
- Streamlit
- LangChain
- OpenAI GPT models (gpt-4o-mini)
- Chroma vector database
- OpenAI Embeddings (text-embedding-ada-002)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rushijoshi1995/GenAI.git
   cd F1_RAG_Chat
   ```

2. Install the required packages:
   ```
   pip install langchain langchain_openai chromadb streamlit
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

4. Prepare your Chroma database:
   - Ensure you have a `chroma_db` directory in your project root with the necessary Formula 1 data embedded

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`)

3. Start chatting! Ask questions about Formula 1, and the app will provide answers based on the information in its database.

## How It Works

1. The user inputs a question about Formula 1 in the chat interface.
2. The app refines the query using the conversation history and current input.
3. The refined query is used to retrieve relevant documents from the Chroma vector database.
4. The retrieved documents and conversation history are fed into a language model to generate a contextual response.
5. The response is displayed in the chat interface with a simulated streaming effect.

## File Structure

- `app.py`: The main Streamlit application file
- `F1_conv_bot.py`: Contains the core RAG functionality (query refinement, document retrieval, and response generation)
- `chroma_db/`: Directory containing the Chroma vector database with Formula 1 information
