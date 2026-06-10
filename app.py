import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
st.set_page_config(page_title="PDF Chatbot by Suman", page_icon="📄")

st.title("📄 Chat with your PDF - Live Demo")
st.write("Upload any PDF and ask questions. Get answers with page citations.")

# OpenAI API Key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Session state init
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

if uploaded_file is not None:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # Create conversation chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    
    st.success("PDF processed! Ask your questions below.")

# Chat interface
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", response['answer']))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Bot:** {message}")
            st.write("---")

st.markdown("---")
st.markdown("[**Order Your Custom Chatbot on Fiverr →**](https://www.fiverr.com/s/pd7l461)")
st.write("Built with LangChain + OpenAI GPT-4 + FAISS by Suman Lata")
