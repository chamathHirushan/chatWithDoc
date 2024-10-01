import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# initialize the env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    '''extract the text from a folder which contains multiple pdfs'''
    text = ''
    for pdf in pdf_docs:
        # Initialize the PdfReader object for that particular pdf
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    '''chunk the given text and return the list of chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)  # Initialize the text_spliter function
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    '''vectorize the given chunks and store it in local'''
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(
        text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possinle from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide any wrong answers\n\n
    
    Context :\n {context}?\n
    Question : \n{question}\n
    
    Answer :
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # stuff helps to do internal text summarization
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # load stored vector index from the local
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)  # Display the response
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF(s)", type="pdf", accept_multiple_files=True)
        if st.button("Submit"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Completed")


if __name__ == "__main__":
    main()
