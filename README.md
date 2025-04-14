# RAG-Bot-with-Live-Agent-Support:
An AI-powered Interactive RAG Bot with seamless live agent integration. Leverages Retrieval-Augmented Generation (RAG) for context-aware responses in web development, digital marketing, and more. Features a Streamlit-based Admin Dashboard to monitor and manage conversations in real-time. Built with LangChain, OpenAI, and Chroma.

# Features:
Retrieval-Augmented Generation (RAG): Provides accurate, context-rich responses using a Chroma vector store and OpenAI embeddings.
Live Agent Support: Allows human agents to take over bot conversations seamlessly via an Admin Dashboard.
Real-Time Monitoring: Admins can view and manage the latest user session with timestamps.
Persistent Conversations: Stores chat history in a local file for continuity.
Streamlit Interface: Intuitive UI for both chat and admin modes, with a sidebar for mode selection.
Customizable: Easily extendable for additional domains or services.

# Tech Stack :
Python 3.8+
Streamlit: For the web interface
LangChain: For RAG pipeline and prompt management
OpenAI API: For embeddings and LLM (GPT-4o-mini)
Chroma: For vector storage
dotenv: For environment variable management

# Installation : 
git clone https://github.com/your-username/RAG-Bot-with-Live-Agent-Support
cd starspark

# Set up a virtual environment (optional but recommended) : 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Create a .env file in the root directory and add your OpenAI API key :
OPENAI_API_KEY=your-api-key-here

# Run the application: 
streamlit run app.py

# Usage : 
Chat Mode:
Access the app at http://localhost:8501.
1 Select "Chat" from the sidebar to interact with the bot.
2 Ask about web development, digital marketing, or other services. The bot responds using RAG or a friendly greeting for simple inputs (e.g., "hi").

#Admin Dashboard:
1 Select "Admin Dashboard" from the sidebar.
2 View the latest user session, including messages and timestamps.
3 Take over a session to respond as a human agent or release it back to the bot.

#Conversation Storage:
Chats are saved in Conversation/conversations.txt for persistence.

# Project Structure :  RAG-Bot-with-Live-Agent-Support/
├── app.py                # Main Streamlit application
├── Conversation/         # Directory for conversation storage
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (not tracked)
└── README.md             # This file



