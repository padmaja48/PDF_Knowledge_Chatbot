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
    try:
        llm = ChatGroq(
            groq_api_key= st.secrets["GROQ_API_KEY"],
            model_name="llama-3.3-70b-versatile",  # Current production model (Sept 2025)
            temperature=0.1
        ) 
        memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages = True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def handle_user_input(user_question):
    try:
        if st.session_state.conversation is None:
            st.error("Please process your PDF files first before asking questions.")
            return
            
        if not user_question.strip():
            st.warning("Please enter a valid question.")
            return

        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        
        # Provide specific error handling for common issues
        error_str = str(e).lower()
        if "badrequest" in error_str or "400" in error_str:
            st.info("This might be due to:")
            st.write("• Invalid API key")
            st.write("• Outdated model name")
            st.write("• Input too long for the model")
            st.write("• API rate limits")
        elif "rate limit" in error_str:
            st.warning("⚠️ Rate limit exceeded. Please wait a moment before trying again.")
        elif "timeout" in error_str:
            st.warning("⚠️ Request timed out. Please try again.")
        elif "model_decommissioned" in error_str:
            st.error("❌ The AI model is outdated. Please contact the developer to update the model.")
            
        # Option to reset conversation
        if st.button("🔄 Reset Conversation"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.success("✅ Conversation reset. Please process your PDFs again.")
            st.rerun()

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
                        
                        if not raw_text.strip():
                            st.error("❌ No text could be extracted from the PDF files. Please check if your PDFs contain readable text.")
                            return
                        
                        # Get the text chunks
                        chunk_data = get_chunk_data(raw_text)
                        
                        if not chunk_data:
                            st.error("❌ Failed to create text chunks from the PDF content.")
                            return
                        
                        # Create vector store(knowledge base)
                        vectorStore = get_vectorestore(chunk_data)
                        
                        # Create Conversation Chain
                        conversation_chain = get_conversation_chain(vectorStore)
                        
                        if conversation_chain:
                            st.session_state.conversation = conversation_chain
                            st.success("✅ PDF processing complete! You can now ask questions.")
                            st.info(f"📄 Processed {len(pdf_docs)} PDF file(s) with {len(chunk_data)} text chunks.")
                        else:
                            st.error("❌ Failed to create conversation chain. Please check your API key and try again.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {str(e)}")
                        
                        # Provide specific guidance based on error type
                        error_str = str(e).lower()
                        if "api" in error_str or "groq" in error_str:
                            st.info("💡 This might be an API issue. Please check:")
                            st.write("• Your GROQ_API_KEY in secrets.toml")
                            st.write("• Your internet connection")
                            st.write("• API rate limits")
                        elif "pdf" in error_str:
                            st.info("💡 This might be a PDF issue. Please try:")
                            st.write("• Uploading a different PDF file")
                            st.write("• Ensuring the PDF contains readable text")
                            st.write("• Using a smaller PDF file")
                        else:
                            st.info("💡 Please try again or contact support if the issue persists.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        # Add helpful information in sidebar
        with st.expander("ℹ️ How to use"):
            st.write("""
            1. **Upload PDFs**: Use the file uploader above
            2. **Process**: Click the 'Process' button to analyze your PDFs
            3. **Ask Questions**: Type your questions in the text input above
            4. **Get Answers**: The AI will answer based on your PDF content
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.write("""
            **Common Issues:**
            - **No response**: Make sure you've processed your PDFs first
            - **Error processing**: Check if your PDF contains readable text
            - **API errors**: Verify your GROQ API key is correct
            - **Slow responses**: Large PDFs take more time to process
            """)

if __name__ == '__main__':
    main()