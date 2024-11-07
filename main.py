import os
import langchain
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()

# Load transcript and slide documents
#transcript_loader = TextLoader('C:\\Users\\ojei1\\Documents\\ML\\RAG\\Chatbot\\lec352.txt')
transcript_loader = TextLoader('C:\\Users\\ojei1\\Documents\\ML\\RAG\\Chatbot\\CS352 L2.txt')
slide_loader = PyMuPDFLoader('C:\\Users\\ojei1\\Documents\\ML\\RAG\\Chatbot\\02-initiation.pdf')
transcript_docs = transcript_loader.load()
slide_docs = slide_loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split documents and create Document objects with metadata
# Replace None with an empty string for slide_number in the metadata to avoid ValueError
transcript_splits = [
    Document(page_content=split.page_content, metadata={"source": "transcript", "lecture_number": 2, "slide_number": ""})
    for split in text_splitter.split_documents(transcript_docs)
]
slide_splits = [
    Document(page_content=split.page_content, metadata={"source": "slide", "lecture_number": 2, "slide_number": i + 1})
    for i, split in enumerate(text_splitter.split_documents(slide_docs))
]

# Combine transcripts and slides into one vector store with correct metadata
vectorstore = Chroma.from_documents(
    documents=transcript_splits + slide_splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

print(vectorstore)
# Set up the base retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create MultiQueryRetriever usi
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
)


# Post-processing function to include metadata in the output
def format_docs(docs):
    formatted = []
    for doc in docs:
        if doc.metadata["source"] == "slide":
            source_info = f"(Lecture {doc.metadata['lecture_number']}, Slide {doc.metadata['slide_number']})"
        else:
            source_info = f"(Lecture {doc.metadata['lecture_number']}, Transcript)"
        formatted.append(f"{source_info}: {doc.page_content}")
    return "\n\n".join(formatted)

# Define the prompt template for query responses
template = """
You are a project management expert. The Human will ask you questions about project management.
Use the following piece of context to answer the question.
If you don't know the answer, simply state that you don't know.
ALWAYS keep your answer concise and within 2 sentences!!! At the end of ypur response,
please provide the source lecture number and slide number of the information you used to answer the question, collate these different sources from multiple retrievals.
If they ask for lecture notes for a specific lecture number, please make lecture notes covering the key information in that lecture, very clearly and in detail.
If the user asks you to go into more detail, go into more detail about the last response you outputted, unless they specify otherwise.
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# RAG Chain setup with Multi-Query Retriever and custom prompt
rag_chain = (
    RunnableLambda(
        lambda x: {
            "context": format_docs(multi_query_retriever.get_relevant_documents(x["question"])),
            "question": x["question"],
        }
    )
    | prompt
    | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Streamlit UI setup
st.set_page_config(page_title="CS352 Lecture Helper")
with st.sidebar:
    st.title("CS352 Lecture Helper")

st.write("Chat with your lecture data. Type 'exit' to stop.")
chat_history = []

# Input and response handling in Streamlit
user_input = st.text_input('Enter your question:')
if user_input and user_input.lower() != 'exit':
    # Use rag_chain to generate a response
    rag_input = {"question": user_input}
    rag_result = rag_chain.invoke(rag_input)

    # Update chat history and display response
    chat_history.append((user_input, rag_result))
    st.write(f'**User:** {user_input}')
    st.markdown(f"**Chatbot:**\n\n{rag_result}")
