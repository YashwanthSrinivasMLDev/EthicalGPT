from langchain_groq import ChatGroq
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import  ChatPromptTemplate
# from dotenv import load_dotenv
import streamlit as st
# load_dotenv()

#create a llm
# groq_api_key= os.environ.get('GROQ_API_KEY')
groq_api_key= st.secrets['GROQ_API_KEY']
llm = ChatGroq(model_name='llama-3.1-8b-instant', temperature=0, max_tokens=None, timeout=None, api_key=groq_api_key)

#embedding
embedding = HuggingFaceEmbeddings()

# create vector database
def create_vector_retriever() :
    document_loader = UnstructuredWordDocumentLoader("./arguments for veganism.docx")
    rag_document = document_loader.load()
    #splitting text to fit into llm context window and for better context understanding
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap=200  )
    split_text = text_splitter.split_documents(rag_document)
    vector_db = FAISS.from_documents(split_text, embedding=embedding)
    vector_file_path= "./faiss_vector_db_for_ethical_chatbot"
    vector_db.save_local(vector_file_path)
    # creating retriever for the vector db
    vector_db_from_local = vector_db.load_local(vector_file_path, embeddings=embedding,allow_dangerous_deserialization=True)
    retriever = vector_db_from_local.as_retriever(score_threshold=0.8)
    return retriever


#creating prompt
prompt_template = """ 
Hi, you are a hardcore animal rights activist who is abolitionist in approach.
 Answer the query {input} first by referring the context provided which is 
 {context}.

if you don't find relevant answer in the context provided, then just 
say that you don't know the answer

"""
prompt = ChatPromptTemplate.from_template(prompt_template)

#creating agent chain using langchain
# agent_chain = prompt | llm

vector_db_retriever = create_vector_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector_db_retriever, document_chain)


# def create_retrieval_chain_custom() :
#     # vector_db_retriever = create_vector_retriever()
#     return temp_retrieval_chain

# retrieval_chain= create_retrieval_chain()
# query = "is eating an animal ethically correct?"
# response = retrieval_chain.invoke( {'input':query })

# print(response['answer'])

