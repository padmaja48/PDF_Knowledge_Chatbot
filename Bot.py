import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunk_data(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorestore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")    
    vectorstore = FAISS.from_texts(texts = chunks,embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key= st.secrets["GROQ_API_KEY"],
        model_name="llama3-70b-8192",
        temperature=0.1
    ) 
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title= "Bot To Chat With Pdf's",page_icon=":robot_face:")
    st.markdown(css, unsafe_allow_html = True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with your PDF files :books:")
    user_question = st.text_input("Ask a question about your PDF")
    
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.subheader("Your PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF files Here and Click on 'Process'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDF files..."):
                    try:
                        # Get pdf text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Get the text chunks
                        chunk_data = get_chunk_data(raw_text)
                        
                        # Create vector store(knowledge base)
                        vectorStore = get_vectorestore(chunk_data)
                        
                        # Create Conversation Chain
                        st.session_state.conversation = get_conversation_chain(vectorStore)
                        
                        st.success("✅ PDF processing complete! You can now ask questions.")
                        
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
                        st.info("Please try again or check your PDF files.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()