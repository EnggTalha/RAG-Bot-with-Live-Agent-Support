import streamlit as st
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from dotenv import load_dotenv
import uuid

# Set page config as the first Streamlit command
st.set_page_config(page_title="Interactive RAG Bot with Live Agent Support", layout="wide")

# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file!")
    st.stop()

# Directory and file for conversation storage
CONVO_DIR = "Conversation"
CONVO_FILE = os.path.join(CONVO_DIR, "conversations.txt")

# Initialize conversation storage
def init_convo_file():
    if not os.path.exists(CONVO_DIR):
        os.makedirs(CONVO_DIR)
    if not os.path.exists(CONVO_FILE):
        with open(CONVO_FILE, "w") as f:
            f.write("")

# Store message in conversation file
def store_message(session_id, message, sender, is_bot_active=True):
    init_convo_file()
    sessions = {}
    current_session = None
    with open(CONVO_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("SESSION:"):
                current_session = line.split("SESSION: ")[1]
                sessions[current_session] = {"messages": [], "is_bot_active": True}
            elif line.startswith("is_bot_active:"):
                if current_session:
                    sessions[current_session]["is_bot_active"] = line.split(": ")[1] == "True"
            elif current_session and ": " in line:
                msg_text = line.split(" [")[0]
                sender_msg = msg_text.split(": ", 1)
                if len(sender_msg) == 2:
                    sessions[current_session]["messages"].append({
                        "sender": sender_msg[0],
                        "message": sender_msg[1],
                        "timestamp": line.split("[")[1][:-1] if "[" in line else ""
                    })

    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "is_bot_active": is_bot_active}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sessions[session_id]["messages"].append({
        "sender": sender,
        "message": message,
        "timestamp": timestamp
    })
    sessions[session_id]["is_bot_active"] = is_bot_active

    with open(CONVO_FILE, "w") as f:
        for sid, data in sessions.items():
            f.write(f"SESSION: {sid}\n")
            for msg in data["messages"]:
                f.write(f"{msg['sender']}: {msg['message']} [{msg['timestamp']}]\n")
            f.write(f"is_bot_active: {data['is_bot_active']}\n")

# Retrieve conversation history
def get_conversation(session_id):
    init_convo_file()
    sessions = {}
    current_session = None
    with open(CONVO_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SESSION:"):
                current_session = line.split("SESSION: ")[1]
                sessions[current_session] = {"messages": [], "is_bot_active": True}
            elif line.startswith("is_bot_active:"):
                if current_session:
                    sessions[current_session]["is_bot_active"] = line.split(": ")[1] == "True"
            elif current_session and ": " in line:
                msg_text = line.split(" [")[0]
                sender_msg = msg_text.split(": ", 1)
                if len(sender_msg) == 2:
                    sessions[current_session]["messages"].append({
                        "sender": sender_msg[0],
                        "message": sender_msg[1],
                        "timestamp": line.split("[")[1][:-1] if "[" in line else ""
                    })

    return [(msg["message"], msg["sender"]) for msg in sessions.get(session_id, {"messages": []})["messages"]]

# Check if bot is active
def is_bot_active(session_id):
    init_convo_file()
    sessions = {}
    current_session = None
    with open(CONVO_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SESSION:"):
                current_session = line.split("SESSION: ")[1]
                sessions[current_session] = {"messages": [], "is_bot_active": True}
            elif line.startswith("is_bot_active:"):
                if current_session:
                    sessions[current_session]["is_bot_active"] = line.split(": ")[1] == "True"

    return sessions.get(session_id, {"is_bot_active": True})["is_bot_active"]

# Set bot active/inactive
def set_bot_active(session_id, active):
    init_convo_file()
    sessions = {}
    current_session = None
    with open(CONVO_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("SESSION:"):
                current_session = line.split("SESSION: ")[1]
                sessions[current_session] = {"messages": [], "is_bot_active": True}
            elif line.startswith("is_bot_active:"):
                if current_session:
                    sessions[current_session]["is_bot_active"] = line.split(": ")[1] == "True"
            elif current_session and ": " in line:
                msg_text = line.split(" [")[0]
                sender_msg = msg_text.split(": ", 1)
                if len(sender_msg) == 2:
                    sessions[current_session]["messages"].append({
                        "sender": sender_msg[0],
                        "message": sender_msg[1],
                        "timestamp": line.split("[")[1][:-1] if "[" in line else ""
                    })

    if session_id in sessions:
        sessions[session_id]["is_bot_active"] = active

    with open(CONVO_FILE, "w") as f:
        for sid, data in sessions.items():
            f.write(f"SESSION: {sid}\n")
            for msg in data["messages"]:
                f.write(f"{msg['sender']}: {msg['message']} [{msg['timestamp']}]\n")
            f.write(f"is_bot_active: {data['is_bot_active']}\n")

# Initialize RAG pipeline
@st.cache_resource
def init_rag():
    documents = [
        "Digital Graphiks is a premier web development and digital marketing agency based in Dubai, UAE, founded in 2009. It specializes in creating modern, responsive websites and delivering innovative digital solutions across Dubai, Abu Dhabi, and Sharjah. The company employs 11-50 professionals and has offices in London, UK.",
        "Digital Graphiks offers affordable, visually stunning, and responsive web design services tailored to brand identity. Expertise includes user experience (UX) design, wireframing, prototyping, and SEO-friendly websites. Tools used include Adobe Photoshop, Sketch, Adobe XD, Balsamiq, InVision, and Figma. Projects include Hawas TV and Access Middle East.",
        "Digital Graphiks provides user-friendly Android and hybrid mobile app development, creating attractive and simple apps. Notable projects include Burgerna, which ranks well on Google, showcasing expertise in mobile solutions.",
        "Digital Graphiks delivers data-driven digital marketing services, including SEO, PPC, SMM, and content creation. It manages enterprise-level analytics and campaigns to enhance client project performance and is recognized as a leader in digital marketing.",
        "Digital Graphiks employs professional content writers in Dubai to create high-quality, SEO-optimized website content, focusing on client products and target audiences to build brand awareness.",
        "Digital Graphiks offers B2B e-commerce solutions to drive growth and efficiency, and Learning Management System (LMS) solutions to transform training initiatives in Dubai.",
        "Digital Graphiks specializes in custom software development, DevOps, DevSecOps, penetration testing, and web application security audits, ensuring robust IT solutions.",
        "Digital Graphiks provides branding services, including logo design and brand strategy, to create cohesive and impactful brand identities for clients.",
        "Contact Digital Graphiks at: UAE - Office #904, Capital Golden Tower, Business Bay, Dubai, UAE; Phone: +971 55 252 0600; Email: info@digitalgraphiks.ae. UK - 20-22, Wenlock Road, London N1 7GU; Phone: +44 753 714 4665; Email: info@digitalgraphiks.co.uk. Website: www.digitalgraphiks.ae; WhatsApp support available."
    ]

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(documents, embeddings, collection_name="digital_graphiks")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """
    You are a helpful and knowledgeable assistant representing Digital Graphiks in Dubai. You specialize in logo design, web development, mobile app development, SEO, digital marketing, content writing, B2B e-commerce, Learning Management Systems (LMS), and custom software development.
    For questions about services or contact details, use the provided context when it's helpful. Otherwise, use your expertise to answer clearly and professionally.
    """
    prompt = PromptTemplate.from_template(
        system_prompt + "\n\nContext: {context}\n\nUser: {question}\n\nAnswer:"
    )

    def custom_rag_chain(input_text):
        docs = vectorstore.similarity_search(input_text, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        chain = (
            {"context": lambda x: context, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(input_text)

    return custom_rag_chain

# Streamlit App
def main():
    rag_chain = init_rag()

    # Initialize session state with stable session_id
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Stable unique ID
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Sidebar for mode selection
    st.sidebar.title("Mode")
    app_mode = st.sidebar.selectbox("Choose mode", ["Chat", "Admin Dashboard"])

    # Chat Interface
    if app_mode == "Chat":
        st.title("Interactive RAG Bot with Live Agent Support")
        st.write("Ask about our services or anything else!")

        # Display conversation history
        chat_container = st.container()
        with chat_container:
            history = get_conversation(st.session_state.session_id)
            for msg, sender in history:
                with st.chat_message(sender.lower()):
                    st.markdown(msg)

            # Initial greeting only if history is empty and not initialized
            if not history and not st.session_state.initialized:
                greeting = "Hello! Welcome to Spherestech. I'm here to help with web development, mobile apps, digital marketing, and more. What's on your mind?"
                store_message(st.session_state.session_id, greeting, "Assistant")
                st.session_state.initialized = True
                with st.chat_message("assistant"):
                    st.markdown(greeting)

        # User input
        if prompt := st.chat_input("Your message"):
            store_message(st.session_state.session_id, prompt, "User")
            with st.chat_message("user"):
                st.markdown(prompt)

            if is_bot_active(st.session_state.session_id):
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Check for greeting-like inputs
                        if prompt.lower() in ["hi", "hello", "hey"]:
                            response = "Hi there! I'm excited to assist you with your project needs. What's on your mind today?"
                        else:
                            response = rag_chain(prompt)
                        st.markdown(response)
                        store_message(st.session_state.session_id, response, "Assistant")
            else:
                with st.chat_message("assistant"):
                    st.markdown("A human agent is handling your conversation. Please wait.")

    # Admin Dashboard
    else:
        st.title("Admin Dashboard")
        st.write("View and manage the latest user conversation.")

        # Get all sessions
        init_convo_file()
        sessions = {}
        current_session = None
        with open(CONVO_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SESSION:"):
                    current_session = line.split("SESSION: ")[1]
                    sessions[current_session] = {"messages": [], "is_bot_active": True}
                elif line.startswith("is_bot_active:"):
                    if current_session:
                        sessions[current_session]["is_bot_active"] = line.split(": ")[1] == "True"
                elif current_session and ": " in line:
                    msg_text = line.split(" [")[0]
                    sender_msg = msg_text.split(": ", 1)
                    if len(sender_msg) == 2:
                        sessions[current_session]["messages"].append({
                            "sender": sender_msg[0],
                            "message": sender_msg[1],
                            "timestamp": line.split("[")[1][:-1] if "[" in line else ""
                        })

        # Find the latest session
        latest_session_id = None
        latest_timestamp = None
        for session_id, data in sessions.items():
            for msg in data["messages"]:
                msg_time = msg["timestamp"]
                try:
                    msg_time_parsed = datetime.strptime(msg_time, "%Y-%m-%d %H:%M:%S")
                    if latest_timestamp is None or msg_time_parsed > latest_timestamp:
                        latest_timestamp = msg_time_parsed
                        latest_session_id = session_id
                except ValueError:
                    continue

        # Display only the latest session
        if latest_session_id:
            with st.expander(f"Session: {latest_session_id}"):
                for msg in sessions[latest_session_id]["messages"]:
                    with st.chat_message(msg["sender"].lower()):
                        st.markdown(f"{msg['message']} *({msg['timestamp']})*")

                if is_bot_active(latest_session_id):
                    if st.button(f"Take over {latest_session_id}", key=f"takeover_{latest_session_id}"):
                        set_bot_active(latest_session_id, False)
                        store_message(latest_session_id, "A human agent has joined the conversation.", "Agent", False)
                        st.rerun()
                else:
                    prompt = st.chat_input(f"Reply in {latest_session_id}", key=f"agent_input_{latest_session_id}")
                    if prompt:
                        store_message(latest_session_id, prompt, "Agent", False)
                        st.rerun()
                    if st.button(f"Release bot for {latest_session_id}", key=f"release_{latest_session_id}"):
                        set_bot_active(latest_session_id, True)
                        store_message(latest_session_id, "The bot has resumed the conversation.", "Agent", True)
                        st.rerun()
        else:
            st.write("No sessions found.")

if __name__ == "__main__":
    main()