import os
import langchain
import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.schema import Document, HumanMessage, PromptValue, AIMessage
from langchain.chains.conversation.base import ConversationChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from typing import List, Tuple, Dict
from pydantic import Field
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(page_title="CS352 Lecture Helper", layout="wide")
store = {}

class DocumentLoader:
    def __init__(self, transcript_path, slide_path):
        self.transcript_loader = TextLoader(transcript_path)
        self.slide_loader = PyMuPDFLoader(slide_path)

    def load_documents(self):
        transcript_docs = self.transcript_loader.load()
        slide_docs = self.slide_loader.load()
        return transcript_docs, slide_docs


class DocumentSplitter:
    def __init__(self, transcript_docs, slide_docs):
        self.transcript_docs = transcript_docs
        self.slide_docs = slide_docs
        self.transcript_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"), breakpoint_threshold_type="percentile",
                                                   min_chunk_size=300
                                                   )
        #self.slide_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.slide_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"), breakpoint_threshold_type="standard_deviation",
                                                min_chunk_size=700
                                                )
        
    def split_documents(self):
        transcript_splits = [
            Document(page_content=split.page_content, metadata={"source": "transcript", "lecture_number": 2, "slide_number": ""})
            for split in self.transcript_splitter.split_documents(self.transcript_docs)
        ]
        # print ("Transcript splits: ", len(transcript_splits))
        
        slide_splits = [
            Document(page_content=split.page_content, metadata={"source": "slide", "lecture_number": 2, "slide_number": i + 1})
            for i, split in enumerate(self.slide_splitter.split_documents(self.slide_docs))
        ]
        # Print transcript and slide splits
        #print ("Transcript splits: ", len(transcript_splits))
        #print ("Slide splits: ", len(slide_splits))
        # Print clear transcript and slide splits
        '''        for split in transcript_splits:
            print(split.page_content)
            print("\n")
        
        for split in slide_splits:
            print(split.page_content)
            print("\n")
        '''
        return transcript_splits, slide_splits
        

class QuestionVectorStore:
    def __init__(self, questions=None, persist_directory="C:\\Users\\ojei1\\Documents\\ML\\RAG\\Chatbot\\chromadb"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.cache_file = os.path.join(persist_directory, 'question_cache.json')
        
        # Caching logic to save/load generated questions
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                cached_questions = json.load(f)
        else:
            cached_questions = []

        # Merging in newly provided questions (if any)
        if questions:
            cached_questions.extend(questions)

        # Writing the merged list back to cache 
        with open(self.cache_file, 'w') as f:
            json.dump(cached_questions, f, indent=4)

        # Loading or creating the Chroma store 
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            # Create from all known questions
            self.vectorstore = Chroma.from_documents(
                documents=cached_questions,   # cached_questions, not questions
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

        # If new questions exist and the store already existed, we add them now
        if os.path.exists(self.persist_directory) and questions:
            self.vectorstore.add_documents(questions)
    
    def as_retriever(self):
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6}
        )
    
# Class which implements QB-RAG
class QueryGenerator:
    def __init__(self, model_name="gpt-4o", temperature=0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
    def generate_questions(self, content):    
        prompt = f"""
        You are a precise quiz generator. Generate 5-7 high-quality, specific, and diverse questions that:
        - Are directly related to the content
        - Require more than a yes/no answer
        - Cover different aspects of the text
        - Are likely to be asked by a university student corresponding to the content
        
        Content: {content}
        
        Output each question on a new line.
        """
        response = self.llm.invoke(prompt)
        
        # Basic filtering
        questions = [
            q.strip() for q in response.content.split('\n') 
            if q.strip() and len(q) > 10 and '?' in q
        ]
        
        return questions[:5]  # Limit to 5 high-quality questions
        
def generate_and_store_queries(docs : List[Document]):
    query_gen = QueryGenerator()
    queries = []
    for doc in docs: 
        # Generate questions for each chunk
        questions = query_gen.generate_questions(doc.page_content)
        for question in questions:
            queries.append({"question" : question, "source": doc.metadata})
    return queries
    
class QBQueryRetriever():
    def __init__(self, question_retriever : VectorStoreRetriever):
        self.retriever = question_retriever
        
    def retrieve_related_content(self, query):
        # Retriever top-k questions that match the user query
        related_questions = self.retriever.get_relevant_documents(query)
        #print(related_questions)
        # Extract associated content metadata
        return related_questions

class ContentRetreiver:
    def __init__(self, content_vectorstore : Chroma):
        self.vectorstore = content_vectorstore
        
    def get_content(self, doc_list : list):
        # Retrieve documents associated with metadata
        docs = []
        for doc in doc_list:
            docs.extend(self.vectorstore.similarity_search_with_score(doc.page_content))
        return docs

class WeightedRetriever():
    
    def __init__(self, contents : Chroma, questions : list[Document]):
        self.questions = questions
        self.contents = contents
        
    
    # Dynamic weighting mechanism 
    def adjust_weights_based_on_query(self) -> list[Tuple[Document, float]]:
        
        slide_keywords = {
            "slide", "diagram", "figure", "chart", "table", "graph",
            "image", "visual", "presentation", "bullet points", "summary slide",
            "flowchart", "schematic", "layout", "infographic", "key takeaways",
            "graphical representation", "overview slide", "powerpoint slide",
            "illustration", "on the slide", "shown in the slide", "in the diagram",
            "the graph shows", "on the chart", "what is", "what are", "define"
        }
        
        transcript_keywords = {
            "explain", "definition", "detailed", "example", "discussion",
            "spoken", "lecture", "elaborate", "conversation", "context",
            "word-for-word", "verbatim", "narrative", "dialogue", 
            "clarify", "expand", "notes", "step-by-step", "explanation",
            "argument", "concept breakdown", "in the lecture", "the professor said",
            "during the talk", "explained in detail", "covered in the discussion",
            "from the transcript", "as mentioned by the professor", "the lecturer"
        }
        
        # Conduct similarity search between each document and all questions
        epsilon = 1.1
        new_order = []
        
        # for each question, get similar documents
        for question in self.questions:
            related_content = self.contents._similarity_search_with_relevance_scores(question.page_content,
                                                                                     k=5)
            #print(content for content in related_content)
            
            # Based on the question, now change the slide and transcript weightsof the related content
            for doc, score in related_content:
                if score < 0.25:
                    # If the score is low, then the content is not relevant to the question
                    continue
                # Tokenize and compute intersections for transcript/slide keyword sets
                q_list = question.page_content.split()
                q_set = set(q_list)
                slide_intersection = q_set.intersection(slide_keywords)
                transcript_intersection = q_set.intersection(transcript_keywords)
                transcript_intersection_score = len(transcript_intersection)
                slide_intersection_score = len(slide_intersection)

                # Update score based on intersections
                if slide_intersection_score > transcript_intersection_score:
                    print("-slide-")
                    if doc.metadata["source"] == "slide":
                        score *= ((slide_intersection_score * epsilon) 
                                / transcript_intersection_score)
                    else:
                        score *= ((transcript_intersection_score) 
                                / slide_intersection_score * epsilon)
                elif transcript_intersection_score > slide_intersection_score:
                    print("-transcript-")
                    if doc.metadata["source"] == "transcript":
                        score *= ((transcript_intersection_score * epsilon) 
                                / transcript_intersection_score)
                    else:
                        score *= ((slide_intersection_score) 
                                / transcript_intersection_score * epsilon)
                        
            
                new_order.append((doc, score))
        new_order.sort(key=lambda tup : tup[1], reverse=True)
        #print("----------------A")
        #print(new_order)
        #print("----------------A")
        return new_order
            
                            
                
                 
            
        
        

        

class QB_RAG_Chain:
    def __init__(self, question_retriever : QBQueryRetriever, content_retriever : VectorStoreRetriever,
                 content_store : Chroma):
        self.question_retriever = question_retriever
        self.content_retriever = content_retriever
        self.content_store = content_store
        self.session_id = "user_session_1"
        
        if self.session_id not in store:
            store[self.session_id] = InMemoryChatMessageHistory()
            
        self.memory = ConversationBufferMemory(
            chat_memory=store[self.session_id],
            return_messages=True,
        )
        


        # NEED TO CREATE SOMETHING HERE TO SELECT DIFFERENT TEMPLATES: PERHAPS A VERY ADVANCED SWITCH-CASE
        # May have to use voting and aggregations
        # E.g. slide reference (Chain A), transcript ref (Chain B),follow-up query of type Y (Chain C), follow-up-query of Type Y (Chain D), detailed lecture notes (Chain E), etc.
         
        self.template = """
        You are a project management expert. The Human will ask you questions about project management. Follow these updated guidelines:

        1. **Chat History and Context**:
        - Always analyze the conversation history first to determine if the user's question refers to earlier topics.
        - If the user's question is vague, infer its intent by connecting it to recent or related conversation history.
        - When in doubt, summarize the relevant part of the conversation history and ask for clarification if necessary.
        - If both memory and context are relevant, synthesize them for a cohesive and well-rounded response.

        2. **Answer Structure**:
        - Use clear, concise formatting (e.g., bullet points or numbered lists) for readability.
        - Provide real-world examples that are specific, realistic, and detailed enough to clarify how the concept applies practically.

        3. **Follow-Up Question Handling**:
        - For vague or ambiguous follow-ups:
            - Link back explicitly to the prior question/answer that seems most relevant.
            - Provide a brief summary of the related content from memory before answering.
            - State assumptions explicitly if the question's intent is unclear.
        - Build upon prior answers unless the user specifies otherwise.
        - Maintain continuity by explicitly linking follow-up answers to prior content and clarifying their connections.

        4. **Source Referencing**:
        - Reference all sources of retrieved information in a clear, collated format: [Source: Lecture X: Slide(s) A, B, Transcript].
        - If no source is available, state that clearly.

        5. **Uncertainty and Clarification**:
        - If unsure about the user's intent, respond with a summary of related context and a clarifying question.
        - If you don't know the answer, simply state, "I don't know."

        Here is the conversation history: {history}

        Use the following necessary context to answer the question: Context: {context}

        User Question: {question}
        Answer:

        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question", "history"])

    def process_query(self, query):
        context = self.format_docs(self.question_retriever.retrieve_related_content(query))
        history = store[self.session_id].messages
        return {
            "context": context,
            "question": query,
            "history": history
        }
    
    def format_docs(self, docs : list | list[Document] | list[Tuple[Document, float]]):
        formatted = []
        docs_standard = []
        #print("------------------------------")
        #print(docs)
        #print("------------------------------")
        # Standardise documents to be in the format [Docs, 0.0]
        if isinstance(docs, list) and all(isinstance(doc, tuple) and len(doc) == 2 for doc in docs):
            docs_standard = docs
        else:
            for doc in docs:
                docs_standard.append((doc, 0.0))  
        # Standardise other formats to be in the form [Doc, 0.0]

        for doc, _ in docs_standard:
            if doc.metadata["source"] == "slide":
                source_info = f"(Lecture {doc.metadata['lecture_number']}, Slide {doc.metadata['slide_number']})"
            else:
                source_info = f"(Lecture {doc.metadata['lecture_number']}, Transcript)"
            formatted.append(f"{source_info}: {doc.page_content}")
        return "\n\n".join(formatted)
    
    # Implementing message history for chat
    def get_session_history(self) -> InMemoryChatMessageHistory:
        if self.session_id not in store:
            store[self.session_id] = InMemoryChatMessageHistory()
            return store[self.session_id]

        assert len(self.memory.memory_variables) == 1
        key = self.memory.memory_variables[0]
        messages = self.memory.load_memory_variables({})[key]
        store[self.session_id] = InMemoryChatMessageHistory(messages=messages)
        return store[self.session_id]
    
            
    def create_rag_chain(self):
        return (
            RunnableLambda(
                lambda x: self.process_query(x["question"])
            )
            | self.prompt
            | ChatOpenAI(model_name="gpt-4o", temperature=0)
            | StrOutputParser()
        )
        
    # Generate LLM response    
    def generate(self, user_query : dict):
        question_docs = self.question_retriever.retrieve_related_content(user_query["question"])
        content_docs = []
        for qdoc in question_docs:
            content_docs.extend(self.content_retriever.get_relevant_documents(qdoc.page_content))
        weighting = WeightedRetriever(contents=self.content_store, questions=question_docs)
        weighted_docs = weighting.adjust_weights_based_on_query() 
        # Retrieve question docs and format them        
        context_str = self.format_docs(weighted_docs)
        #print("**********")
        #print(context_str)
        #print("**********")
        data = {"context" : context_str, "question": user_query["question"],
                "history": st.session_state.messages}
        
        print("HISTORY:")
        print(data["history"])
        
        chain = self.create_rag_chain()
        
        # Adding message history- Wrapping 'chain' with history
        with_message_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        
        # Get the result of the query with QB-RAG implemented. 
        # Format the human and AI messages and add them to the store
        result = with_message_history.stream(data, config={"configurable": {"session_id": self.session_id}},
                                             )
        #result_other = with_message_history.invoke(data, config={"configurable": {"session_id": self.session_id}})    
        
        
        # Now add the question and AI response to the store
        
        self.get_session_history().add_user_message(user_query["question"])
        #self.get_session_history().add_ai_message(result_other)
        
        #print("IN STORE:")
        #print(store[self.session_id])
        
        #history_list = self.get_session_history()
        #history_list.append(user_question)
        #history_list.append(result_for_store)
        #store[self.session_id] = history_list
        
        #print("Memory so far:", self.get_session_history().messages)
        print("Streamlit messages:", st.session_state["messages"])
        #print("STORE: " + (item for item in store.items()))
        return result
        



class StreamlitUI:
    def __init__(self, rag_chain: QB_RAG_Chain):
        self.rag_chain = rag_chain
        
    def initialize_session_state(self):
        """Initialize session state variables if they don't exist"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_id" not in st.session_state:
            st.session_state.chat_id = "unique_id_for_this_chat"

    def display_message(self, message: Dict, is_user: bool):
        """Display a single message with appropriate styling"""
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(message["content"])

    def display_chat_history(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            self.display_message(message, message["role"] == "user")

    def setup_ui(self):
        # Initialize session state
        self.initialize_session_state()
        
        # Sidebar
        with st.sidebar:
            st.title("CS352 Lecture Helper")
            
            # Add a clear chat button
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        # Main chat interface
        st.title("Chat with your Lecture Data")
        
        # Display chat history
        self.display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your lectures..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            self.display_message({"role": "user", "content": prompt}, True)
            
            # Create a placeholder for the assistant's response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # Generate streaming response
                rag_input = {"question": prompt}
                response = self.rag_chain.generate(user_query=rag_input)
                for chunk in response:
                    # Update the response in real-time
                    full_response += str(chunk)
                    response_placeholder.markdown(full_response + "▌")
                
                # Show final response without cursor
                response_placeholder.markdown(full_response)
                print(full_response)
            
            # Add complete response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    def apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
            <style>
            .stChatMessage {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .stChatMessage[data-testid="chat-message-user"] {
                background-color: #f0f2f6;
            }
            .stChatMessage[data-testid="chat-message-assistant"] {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
            }
            </style>
        """, unsafe_allow_html=True)



def main():
    # Load environment variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = st.secrets.api_keys.OPENAI_API_KEY
    os.environ['LANGCHAIN_API_KEY'] = st.secrets.api_keys.LANGCHAIN_API_KEY
    os.environ['LANGCHAIN_TRACING_V2'] = st.secrets.other_secrets.LANGCHAIN_TRACING_V2
    
    # 1) Load existing content Chroma store from disk
    content_vectorstore = Chroma(
        persist_directory="C:\\Users\\ojei1\\Documents\\ML\\RAG\\Chatbot\\content_chromadb",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Store (new) questions in the vector store
    question_store = QuestionVectorStore(questions=None)
    q_retriever = question_store.as_retriever()
    
    query_retriever = QBQueryRetriever(question_retriever=q_retriever)
    
    
    # 2) Build Weighted + QB retriever
    qb_retriever = content_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    if "qb_rag_chain" not in st.session_state:
        st.session_state["qb_rag_chain"] = QB_RAG_Chain(
            question_retriever=query_retriever,
            content_retriever=qb_retriever,
            content_store=content_vectorstore
                                   )

    # 5) Setup Streamlit UI
    streamlit_ui = StreamlitUI(st.session_state["qb_rag_chain"])
    streamlit_ui.apply_custom_css()
    streamlit_ui.setup_ui()

if __name__ == "__main__":
    main()
