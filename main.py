import os
import streamlit as st
import json
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from langchain.schema import Document, HumanMessage, AIMessage
from typing import List, Tuple
from functools import lru_cache
#__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

st.set_page_config(page_title="CS352 Lecture Helper", layout="wide")



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
    def __init__(self, questions=None, persist_directory="chromadb"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.cache_file = os.path.join(persist_directory, 'question_cache.json')
        
        if questions is None and self.persist_directory is None:
            self.questions = []
        
        else:
        
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
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
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
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
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

# Class which decomposes a query into smaller sub-queries for higher accuracy
class DecompQueryGenerator():
    def __init__(self, query, model_name="gpt-4o-mini", temperature=0):
        self.query = query
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
    def generate_decomp_queries(self):
        prompt= f"""
        You are an expert in breaking down the user question into sub-questions.
        Perform query decomposition. Given a user question, \
            IF SUITABLE, break down the following question into DISTINCT sub-questions that you need to answer in order to answer \
                the original question. Generate up to 4 sub-questions for the query.
                
        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        Question: {self.query}
        
        Output each sub-question on a new line.
                """
        response = self.llm.invoke(prompt)
        
        # Filtering
        sub_questions = [
            q.strip() for q in response.content.split('\n')
            if q.strip() and len(q) > 10 and '?' in q
        ]
        
        
        sub_questions = list(set(sub_questions))
        
        return sub_questions # return all, as we would get an arbitrary no. of Qs
        
class QBQueryRetriever():
    def __init__(self, question_retriever : VectorStoreRetriever):
        self.retriever = question_retriever
        
    ###########################    
    @lru_cache(maxsize=200)
    def get_cached_retrieval(self, query):
        return self.retriever.get_relevant_documents(query)
        
    def retrieve_related_content(self, queries : list):
        # Retriever top-k questions that match the user query
        related_sub_query_questions = []
        for query in queries:
            related_questions = self.get_cached_retrieval(query)
            #print(related_questions)
            # Extract associated content metadata
            for q in related_questions:
                related_sub_query_questions.append(q)
        return related_sub_query_questions

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
            "the graph shows", "on the chart", "what is", "what are", "define", "definition"
        }
        
        transcript_keywords = {
            "explain", "detailed", "example", "discussion",
            "spoken", "lecture", "elaborate", "conversation", "context",
            "word-for-word", "verbatim", "narrative", "dialogue", 
            "clarify", "expand", "notes", "step-by-step", "explanation",
            "argument", "concept breakdown", "in the lecture", "the professor said",
            "during the talk", "explained in detail", "covered in the discussion",
            "from the transcript", "as mentioned by the professor", "the lecturer"
        }
        
        # Conduct similarity search between each document and all questions
        epsilon = 1.3
        new_order = []
        
        # dictionary to store content 
        content_included = {}
        
        # for each question, get similar documents
        for question in self.questions:
            related_content = self.contents._similarity_search_with_relevance_scores(question.page_content,
                                                                                     k=5)
            #print(content for content in related_content)
            
            # Based on the question, now change the slide and transcript weightsof the related content
            for doc, score in related_content:
                        
                if score < 0.05:
                    # If the score is low, then the content is not relevant to the question
                    continue
                # Tokenize and compute intersections for transcript/slide keyword sets
                q_list = question.page_content.split()
                q_set = set(q_list)
                slide_intersection = q_set.intersection(slide_keywords)
                transcript_intersection = q_set.intersection(transcript_keywords)
                transcript_intersection_score = len(transcript_intersection)
                slide_intersection_score = len(slide_intersection)
                
                if slide_intersection_score == 0:
                    slide_intersection_score = 1
                if transcript_intersection_score == 0:
                    transcript_intersection_score = 1
                
                # Update score based on intersections
                if slide_intersection_score > transcript_intersection_score:
                    if doc.metadata["source"] == "slide":
                        score *= ((slide_intersection_score * epsilon) 
                                / transcript_intersection_score)
                    else:
                        score *= ((transcript_intersection_score) 
                                / slide_intersection_score * epsilon)
                elif transcript_intersection_score > slide_intersection_score:
                    if doc.metadata["source"] == "transcript":
                        score *= ((transcript_intersection_score * epsilon) 
                                / transcript_intersection_score)
                    else:
                        score *= ((slide_intersection_score) 
                                / transcript_intersection_score * epsilon)
                        
                # Check if a piece of content has already been processed
                # If it has a higher score, we update the score in the dict and 'new_order' list
                if hash(doc.page_content) in content_included:
                    if score > content_included[hash(doc.page_content)]:
                        content_included[hash(doc.page_content)] = score
                        for document, old_score in new_order:
                            if hash(document.page_content) == hash(doc.page_content):
                                new_order.remove((document, old_score))
                                new_order.append((doc, score))
                    else:
                        continue
                else:
                    content_included[hash(doc.page_content)] = score
                    new_order.append((doc, score))
        new_order.sort(key=lambda tup : tup[1], reverse=True)

        return new_order
            
                            
                
                 
            

        

class QB_RAG_Chain:
    def __init__(self, question_retriever : QBQueryRetriever, content_retriever : VectorStoreRetriever,
                 content_store : Chroma):
        self.question_retriever = question_retriever
        self.content_retriever = content_retriever
        self.content_store = content_store
        self.session_id = "unique_id_for_this_chat"
        self.llm = ChatOpenAI(model="gpt-4o-mini", 
                                       temperature=0)
        global sub_queries
        sub_queries = []
        
                    


        # NEED TO CREATE SOMETHING HERE TO SELECT DIFFERENT TEMPLATES: PERHAPS A VERY ADVANCED SWITCH-CASE
        # May have to use voting and aggregations
        # E.g. slide reference (Chain A), transcript ref (Chain B),follow-up query of type Y (Chain C), follow-up-query of Type Y (Chain D), detailed lecture notes (Chain E), etc.
         
        self.template = """
        You are a project management expert. The Human will ask you questions about project management. Follow these updated guidelines:

        1. **Chat History and Context**:
        - Always analyze the conversation history first to determine if the user's question refers to earlier topics AND content present there.
        - If the user's question is vague, infer its intent by connecting it to recent or related conversation history.
        - When in doubt, summarize the relevant part of the conversation history and ask for clarification if necessary.
        - If both memory and context are relevant, synthesize them for a concise, succinct, cohesive ,and well-rounded response.
        - Each peice of CONTEXT should have a number next to it. Prioritise CONTEXTS that have the highest number as being the most useful. 

        2. **Answer Structure**:
        - Use clear, concise formatting.
        - Answer in a succinct manner- GET TO THE POINT and answer the question directly.
        

        3. **Follow-Up Question Handling**:
        - For vague or ambiguous follow-ups:
            - Link back explicitly to the prior question/answer in the conversation HISTORY that seems most relevant.
            - Provide a brief summary of the related content from memory before answering.
            - State assumptions explicitly if the question's intent is unclear.
        - Build upon prior answers unless the user specifies otherwise.
        - Maintain continuity by explicitly linking follow-up answers to prior content and clarifying their connections.

        4. **Source Referencing**:
        - ALWAYS Rememebr to Reference ALL sources of retrieved information in a clear, collated format: [Source: Lecture X: Slide(s) A-B/ A, B, C, etc., Transcript].
        - If no source is available, state that clearly.

        5. **Uncertainty and Clarification**:
        - If unsure about the user's intent, respond with a summary of related context and a clarifying question.
        - If you don't know the answer, simply state, "I don't know."

        ALWAYS Rememebr to Reference ABSOLUTELY ALL sources of retrieved slide and transcript information context  in a clear, collated format throughout, so the user can see where you have sourced your response from: [Source: Lecture X: Slide(s) A, B,/A-C Transcript]. ALWAYS DO THIS!!! It is vital the user knows where you have got all your sources from.
        
        Make sure your lecture references are PRECISE and useful. 
        
        
        Here is the conversation history: {history}

        Use the following necessary context to answer the question: Context: {context}

        At the end, IF transcript content was used in your answer, put: [Lecturer says:] then all the relevant synthesized transcript content at the end that answers ALL of the question. ALWAYS DO THIS!!! 
        
        DO NOT MAKE up any information. If you do not understand or it is not referenced in the context, say "I don't know".
        User Question: {question}
        Answer:

        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question", "history"])

    def process_query(self, query):
        decomp_generator = DecompQueryGenerator(query=query)
        sub_queries = decomp_generator.generate_decomp_queries()
        
        question_docs = self.question_retriever.retrieve_related_content(sub_queries)
        content_docs = []
        
        # Retrieve question docs and format them        
        for qdoc in question_docs:
            content_docs.extend(self.content_retriever.get_relevant_documents(qdoc.page_content))
        weighting = WeightedRetriever(contents=self.content_store, questions=question_docs)
        weighted_docs = weighting.adjust_weights_based_on_query() 
        
        context = self.format_docs(weighted_docs)
        with open("context.txt", "w", encoding="utf-8") as f:
            f.write(context)
        f.close()
        
        # We NEED to get the CONTEXT, not the QUESTIONS relating to the context 
        history = st.session_state.messages.load_memory_variables({})["history"]


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
        return InMemoryChatMessageHistory(messages=st.session_state.messages.load_memory_variables({})["history"])
    
            
    def create_rag_chain(self):
        return (
            RunnableLambda(
                lambda x: self.process_query(x["question"])
            )
            | self.prompt
            | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )
    
    def classify_prompt(self, user_query : str) -> str:
        classifier_prompt = f"""
        Read the conversation history very carefully:
        {st.session_state.messages.load_memory_variables({})["history"]}
        This is the user's question: {user_query}
        Based on the history, classify this question as one of the following:
        1) FOLLOW_UP
        2) FULL_QUESTION

        Criteria:
        - If the user is referencing something immediately previously said 
        or is only asking for more detail, label it FOLLOW_UP.
        - If there is NO HISTORY and the question COULD be seen as a full question, label it FULL_QUESTION.
        - Otherwise, label it FULL_QUESTION.
        Answer with ONLY the label. NOTHING ELSE.
        """
        
        classifier_response = self.llm.invoke(classifier_prompt).content.strip()
        return classifier_response
        
    def format_conversation_history(self):
        history = st.session_state.messages.load_memory_variables({})["history"]
        formatted_history = ""
        for message in history:
            if isinstance(message, HumanMessage):
                formatted_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI: {message.content}\n"
        return formatted_history

    def follow_up(self, user_query: str):
        
        formatted_history = self.format_conversation_history()
        
        short_chain_prompt = f"""
        Read the conversation history very carefully:
        {st.session_state.messages.load_memory_variables({})["history"]}
        Now the user is asking: {user_query}
        Provide a follow-up response that builds on the previous answer.
        """
        response = self.llm.stream(short_chain_prompt)
        
        # Process streamed chunks
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'content'):  # Adjust based on structure
                full_response += chunk.content
            else:
                full_response += str(chunk)  # Fallback for raw text
                
        return full_response
        
        
    # Generate LLM response    
    def generate(self, user_query : dict):
        response = self.classify_prompt(user_query=user_query["question"])
        if "FOLLOW_UP" in response:
            return self.follow_up(user_query=user_query["question"])
        
        question_docs = self.question_retriever.retrieve_related_content(sub_queries)
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
                "history": st.session_state.messages.load_memory_variables({})["history"]}
        
        
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
        result = with_message_history.stream(data, config={"configurable": {"session_id": st.session_state.chat_id}},
                                             )
        #result_other = with_message_history.invoke(data, config={"configurable": {"session_id": self.session_id}})    
        
    
        return result
        
        # TO-DO: SUMMARY chat window of 5 maximum



class StreamlitUI:
    def __init__(self, rag_chain: QB_RAG_Chain):
        self.rag_chain = rag_chain
        
    def initialize_session_state(self):
        """Initialize session state variables if they don't exist"""
        if "messages" not in st.session_state:
            st.session_state.messages = ConversationBufferWindowMemory(
                k=3,
                return_messages=True,
                chat_memory= InMemoryChatMessageHistory(),
                human_prefix="Human Said",
                ai_prefix="AI Responded"
            )
            
        if "chat_id" not in st.session_state:
            st.session_state.chat_id = "unique_id_for_this_chat"
    


    def setup_ui(self):
        # Initialize session state
        # st.session_state.messages["chat_memory"] = InMemoryChatMessageHistory()
        self.initialize_session_state()
        
        # Sidebar
        with st.sidebar:
            st.title("CS352 Lecture Helper")
            
            # Add a clear chat button
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = ConversationBufferWindowMemory(
                    k=3,
                    return_messages=True,
                    chat_memory= InMemoryChatMessageHistory(),
                    human_prefix="Human Said",
                    ai_prefix="AI Responded"
                )
                #st.rerun()
                
        
        # Main chat interface
        st.title("Chat with your Lecture Data")
        
        
        history = st.session_state.messages.load_memory_variables({})["history"]
        for message in history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    human_text = message.content.replace('Human Said ', '')
                    st.write(human_text)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        
        # User input processing

        if prompt := st.chat_input("Ask a question about your lectures..."):
            # Display user input as a chat message
            with st.chat_message("user"):
                st.write(prompt)
                                
            # Create a placeholder for the assistant's response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Generate streaming response
                    rag_input = {"question": prompt}
                    response = self.rag_chain.generate(user_query=rag_input)
                    for chunk in response:
                        # Update the response in real-time
                        if 'content' in chunk:
                            full_response += chunk['content']
                            response_placeholder.write(full_response + "▌")
                        else:
                            full_response += str(chunk)
                            response_placeholder.write(full_response + "▌")
                    
                    # Show final response without cursor
                    response_placeholder.write(full_response)
            
            # Add complete response to chat history
            st.session_state.messages.save_context({"input": ("Human Said " + prompt)}, {"output ": full_response})

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

    content_vectorstore = Chroma()
    question_store = QuestionVectorStore(questions=None, persist_directory=None)
    q_retriever = question_store.as_retriever()
    
    if "content" not in st.session_state:
        
        # Load existing content Chroma store from disk
        content_vectorstore = Chroma(
            persist_directory="content_chromadb",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        st.session_state["content"] = content_vectorstore
    else:
        content_vectorstore = st.session_state["content"]



    if "question_content" not in st.session_state:
        question_store = QuestionVectorStore(questions=None)
        st.session_state["question_content"] = question_store 
        q_retriever = question_store.as_retriever()
    else:
        question_store = st.session_state["question_content"]
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
