import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
from langchain.document_loaders import  PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain

# Load the environment variables
load_dotenv()


llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),temperature=0.5)
chain = load_qa_chain(llm,chain_type='stuff')
pc = Pinecone(api_key=os.getenv('Pinecone_API_KEY'))
index = pc.Index(os.getenv('Pinecone_Index'))

# load PDF from document folder
def load_pdf(path):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    return documents

# convert PDF files to chunks
def create_chunks(doc, chunk_size = 800, chunk_overlap = 50):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = splitter.split_documents(doc)
    return chunks

# create vector Index
def create_index(chunk_doc, embeddings , Index_name):
    Index = PineconeVectorStore.from_documents(chunk_doc, embeddings,index_name = Index_name)
    return Index

def retrieve_query(query,docsearch,k=2):
    results= docsearch.similarity_search(query,k=k)
    return results




def check_document_exist(folder_path):
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist.")
        return
    files_list = os.listdir(folder_path)
    documents_found = any(file.endswith(('.doc', '.docx', '.pdf')) for file in files_list)
    return documents_found,files_list

def upload_pdf_to_folder(uploaded_file, folder_path):
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist.")
        return

    # Check if the uploaded file is a PDF
    if uploaded_file.type == 'application/pdf':
        # Save the uploaded PDF file to the folder
        with open(os.path.join(folder_path, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
    else:
        st.error("Please upload a PDF file.")



# Define the Streamlit app
def main():
    index_created = []
    # Define the header
    st.title('Document Q & A on Custome Documents')
    st.header('This is a Q & A custome document retrieval system')

    # Define the sidebar
    st.sidebar.title('Options')
    st.sidebar.text('Please select below options to proceed')

    # Add content to the main section
    st.write('Hello, this is the main content of the app.')

    input_text = st.text_input("Please enter the text")
    Analyze = st.button("Analyze")
    
    folder_path = 'documents'
    document_exist ,files_list= check_document_exist(folder_path)
    with st.sidebar:
        
        if document_exist:
            st.write("Documents found in the folder:")
            for file in files_list:
                if file.endswith(('.doc', '.docx', '.pdf')):
                    st.write(f"- {file}")

            delete_confirmation = st.checkbox("Confirm deletion of documents")
            delete_button = st.button("Delete documents")

            if delete_confirmation and delete_button:
                # Delete the documents
                for file in files_list:
                    if file.endswith(('.doc', '.docx', '.pdf')):
                        os.remove(os.path.join(folder_path, file))
                        index.delete(delete_all=True)
                st.success("Documents and Index deleted from Pinecone successfully.")
            else:
                st.warning("Deletion not confirmed.")
        else:
            st.write("No documents found in the folder.")

        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        if st.button("Upload File"):
            if uploaded_file is not None:
                upload_pdf_to_folder(uploaded_file, folder_path)
            else:
                st.warning("Please upload a file first.")
        document_exist ,files_list= check_document_exist(folder_path)
        if document_exist:
            add_radio = st.radio(
                        "Please select the method for document analysis",
                        ("OpenAI Langchain", "LlamaIndex"))
        else:
            st.warning("No documents found in the folder.")
    if document_exist and Analyze:
        if add_radio == "OpenAI Langchain":
            with st.spinner():
                document = load_pdf(folder_path)
                embeddings = OpenAIEmbeddings()
                chunks_document = create_chunks(document)
                # print(chunks_document)
                if index_created is not None:
                    index_created= create_index(chunks_document, embeddings, os.getenv('Pinecone_Index'))
                results =retrieve_query(input_text,index_created)
                print("Results",results)
                response=response=chain.run(input_documents=results,question=input_text)
                st.header("Analysis text")
                st.write(response)
                st.divider() 
                col1, col2 =st.columns(2)
                col1.subheader("Reference content")
                col1.write(results[0].page_content)
                col1.divider()
                col1.subheader("page Number")
                col1.write(results[0].metadata['page'])
                col2.subheader("Reference content")
                col2.write(results[1].page_content)
                col2.divider()
                col2.subheader("page Number")
                col2.write(results[1].metadata['page'])

            st.success("done")








        

        


if __name__ == '__main__':
    main()





