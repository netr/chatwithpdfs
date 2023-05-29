import os.path
import pickle

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


def main():
    load_dotenv()

    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    vector_store: FAISS

    if pdf is not None:
        pdf_name = pdf.name[:-4]

        if os.path.exists(f"{pdf_name}.pkl"):
            with open(f"{pdf_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_text(text=text)
            embeddings = OpenAIEmbeddings()

            vector_store = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{pdf_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        # accept user question / query
        query = st.text_input("Ask a question", value="")
        if query:
            docs = vector_store.similarity_search(query, k=3)

            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = qa_chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
