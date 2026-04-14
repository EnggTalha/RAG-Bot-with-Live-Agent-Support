# 🤖 RAG Bot with Live Agent Support

An AI-powered conversational bot that combines **Retrieval-Augmented Generation (RAG)** with seamless **live human agent takeover**. Built for Digital Graphiks — a Dubai-based web development and digital marketing agency — it answers context-aware questions about services while letting human agents step in at any time via a real-time Admin Dashboard.

Built with **LangChain**, **OpenAI GPT-4o-mini**, **Chroma**, and **Streamlit**.

---

## ✨ Features

- **RAG-Powered Responses** — Retrieves relevant context from a Chroma vector store using OpenAI embeddings, then generates accurate, grounded answers via GPT-4o-mini.
- **Live Agent Takeover** — Admins can take over any active user session instantly from the dashboard, replacing bot responses with human replies.
- **Admin Dashboard** — Streamlit-based interface to monitor the latest user session with full message history and timestamps.
- **Graceful Handoff** — Notifies users when a human agent joins or releases a session back to the bot.
- **Persistent Conversations** — Chat history is saved to a local file (`Conversation/conversations.txt`) so sessions survive page refreshes.
- **Stable Session IDs** — Each browser session gets a UUID-based ID, ensuring isolated, consistent conversation tracking.
- **Smart Greeting Detection** — Responds to casual openers like "hi" or "hello" with a friendly message instead of triggering the full RAG pipeline.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **UI Framework** | Streamlit |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI Embeddings |
| **Vector Store** | Chroma |
| **RAG Framework** | LangChain |
| **Env Management** | python-dotenv |
| **Session Tracking** | UUID |

---

## 📁 Project Structure

```
RAG-Bot-with-Live-Agent-Support/
├── app.py                  # Main Streamlit application
├── Conversation/
│   └── conversations.txt   # Persistent session & message storage
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not tracked by Git)
├── Demo.webm               # Demo video
├── Demo with history.webm  # Demo video with conversation history
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/EnggTalha/RAG-Bot-with-Live-Agent-Support
cd RAG-Bot-with-Live-Agent-Support
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💬 Usage

### Chat Mode
1. Select **"Chat"** from the sidebar.
2. Ask anything about web development, digital marketing, mobile apps, SEO, branding, or other services offered by Digital Graphiks.
3. The bot uses RAG to retrieve relevant context and respond accurately.

### Admin Dashboard
1. Select **"Admin Dashboard"** from the sidebar.
2. View the most recently active user session with full message history and timestamps.
3. Click **"Take over"** to disable the bot and reply as a human agent.
4. Click **"Release bot"** to hand control back to the AI.

---

## ⚙️ How It Works

```
User Input
    │
    ▼
Is bot active for this session?
    │
    ├── YES → Similarity search on Chroma vector store
    │              │
    │              ▼
    │         Top-k relevant documents retrieved
    │              │
    │              ▼
    │         LangChain prompt + GPT-4o-mini generates response
    │              │
    │              ▼
    │         Response stored & displayed to user
    │
    └── NO  → "A human agent is handling your conversation."
                    │
                    ▼
              Admin types reply in dashboard → stored & displayed
```

---

## 🔒 Security Notes

- **Never commit your `.env` file.** It is listed in `.gitignore` by default.
- The `.env` file in this repository is a placeholder — replace it with your own key before running.
- For production deployments, use environment secrets via your hosting provider (e.g., Streamlit Cloud Secrets, Railway, or Heroku Config Vars).

---

## 🗺️ Roadmap / Possible Improvements

- [ ] Add multi-session support to the Admin Dashboard (view all active sessions, not just the latest)
- [ ] Replace file-based storage with a database (SQLite or PostgreSQL)
- [ ] Add authentication for the Admin Dashboard
- [ ] Support document ingestion from PDFs or URLs via the UI
- [ ] WebSocket or polling-based live refresh for the admin view
- [ ] Deploy to Streamlit Cloud with one-click setup

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Talha** — [GitHub @EnggTalha](https://github.com/EnggTalha)
