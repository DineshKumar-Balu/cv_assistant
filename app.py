import os
import PyPDF2
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from literalai import LiteralClient

# Load environment variables from .env file
load_dotenv()

# GROQ API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize GROQ chat
llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0.2
)

client = LiteralClient(api_key=os.getenv('LITERAL_API_KEY'))
# This will fetch the champion version, you can also pass a specific version
prompt = client.api.get_prompt(name="RAG prompt")

# Static PDF file path
pdf_path = 'dineshcvforbot.pdf'

def process_pdf(pdf_path):
    # Read the content of your PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    pdf_chunks = text_splitter.split_text(pdf_text)
    return pdf_chunks

def initialize_chain(pdf_chunks):
    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(pdf_chunks, embeddings)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return chain

@st.cache_resource(hash_funcs={RecursiveCharacterTextSplitter: id})
def load_pdf(pdf_path):
    return process_pdf(pdf_path)

@st.cache_resource(hash_funcs={ConversationalRetrievalChain: id})
def load_chain(pdf_chunks):
    return initialize_chain(pdf_chunks)

def main():
    st.set_page_config(
        page_title="virtual-agent",
        page_icon="image.png"
    )

    st.title("Hi! 👋 I'm Dinesh Kumar Balu ")

    # Load PDF and initialize chain
    pdf_chunks = load_pdf(pdf_path)
    chain = load_chain(pdf_chunks)

    # Display previous conversation history
    st.subheader("This is My CV Assistant You can ask questions like")
    # TODO: Add code to display previous conversation history

    # Display user input and bot response in a chat-like interface
    st.markdown("---")
    st.markdown("##### What is your name?")
    st.markdown("##### What is Dinesh's experience?")
    st.markdown("##### Tell me about Dinesh?")   
    st.markdown("---")

    # Create a container for the chat interface
    chat_container = st.container()

    # Create a column for the user input
    user_input_col = st.columns([1])[0]
    with user_input_col:
        user_input = st.text_input("User Input", help="Type your message here and press Enter to send.", key="user_input")

    # Check if the user input is not null or empty
    if user_input:
        # Call the chain with user's message content
        res = chain(user_input)
        answer = res["answer"]

        # Display user's message and bot's response in a chat bubble format
        with chat_container:
            st.markdown(f"#### **YOU**: {user_input}")
        with chat_container:
            st.markdown(f"> {answer}")

if __name__ == "__main__":
    main()
