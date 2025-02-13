import os
import streamlit as st
import json
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
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
from langchain.retrievers import MultiVectorRetriever, EnsembleRetriever, ParentDocumentRetriever
from typing import List, Tuple
from functools import lru_cache
from async_lru import alru_cache
from uuid import uuid4
__import__('pysqlite3')
import sys
import faiss
import asyncio
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from contextlib import contextmanager
import time
import concurrent.futures
import nest_asyncio
import re



sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

st.set_page_config(page_title="NovaCS", layout="wide")
nest_asyncio.apply()


@contextmanager
def timer(stage_name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{stage_name} took {end - start:.4f} seconds")

class DocumentLoader:
    def __init__(self, transcript_path, slide_path):
        self.transcript_loader = TextLoader(transcript_path)
        self.slide_loader = PyMuPDFLoader(slide_path)

    def load_documents(self):
        transcript_docs = self.transcript_loader.load()
        slide_docs = self.slide_loader.load()
        return transcript_docs, slide_docs


class DocumentSplitter:
    def __init__(self, transcript_docs, slide_docs, lecture_number):
        self.transcript_docs = transcript_docs
        self.slide_docs = slide_docs
        self.lecture_number = lecture_number
        self.transcript_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"), breakpoint_threshold_type="percentile",
                                                   min_chunk_size=300
                                                   )
        #self.slide_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.slide_splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"), breakpoint_threshold_type="standard_deviation",
                                                min_chunk_size=700
                                                )
        
    def split_documents(self):
        transcript_splits = [
            Document(page_content=split.page_content, metadata={"source": "transcript", "lecture_number": self.lecture_number, "slide_number": ""})
            for split in self.transcript_splitter.split_documents(self.transcript_docs)
        ]
        # print ("Transcript splits: ", len(transcript_splits))
        
        slide_splits = [
            Document(page_content=split.page_content, metadata={"source": "slide", "lecture_number": self.lecture_number, "slide_number": i + 1})
            for i, split in enumerate(self.slide_splitter.split_documents(self.slide_docs))
        ]

        return transcript_splits, slide_splits
        

class QuestionVectorStore:
    def __init__(self, questions=None, persist_directory="faiss_index"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory
        self.questions = questions
        
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(1536)  # 1536 = embedding dimension of OpenAI embeddings
        self.docstore = InMemoryDocstore()
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=self.index, 
            docstore=self.docstore,
            index_to_docstore_id={},
            distance_strategy= DistanceStrategy.COSINE)
        
        if questions is None and self.persist_directory is None:
            self.questions = []
        
        else:
            # Load from disk if exists
            if os.path.exists(self.persist_directory):
                print("Question DB Exists")
                self.vectorstore = FAISS.load_local(
                    folder_path=persist_directory, 
                    embeddings=self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            
            # Add questions if provided
            if self.questions:
                self.add_documents(self.questions)
        


    def add_documents(self, docs):
        uuids = [str(uuid4()) for _ in range(len(docs))]
        self.vectorstore.add_documents(documents=docs, ids=uuids)
        self.save_index()

    def save_index(self):
        docs_to_add = FAISS.from_documents(documents=self.questions, embedding=self.embeddings)
        docs_to_add.save_local(folder_path=self.persist_directory)
        self.vectorstore = FAISS.load_local(folder_path=self.persist_directory, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        
    def get_index(self):
        return FAISS.load_local(
            folder_path=self.persist_directory,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
                                )


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
        
        return questions[:3]  # Limit to 5 high-quality questions
        
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
        self.content_store = FAISS.load_local(folder_path="content_index", 
                                              embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
                                              allow_dangerous_deserialization=True)
        self.content = self.content_store.as_retriever()
        self.context = self.content.get_relevant_documents(query=self.query)
    def generate_decomp_queries(self):
        prompt= f"""
        You are an expert in breaking down the user question into sub-questions.
        Perform query decomposition. Given a user question, \
            IF SUITABLE, break down the following question into DISTINCT minimal sub-questions that you need to answer in order to answer \
                the original question. This could include asking for the features of an entity or definition if necessary. Generate up to 4 sub-questions for the query.
        If the query is a simple straightforward question, DO NOT break it down. Just return the same query.        
        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        CAREFULLY READ AND USE THE RELATED CONTEXT TO HELP YOU FORMULATE THE QUESTIONS, SO THE QUESTIONS ARE DOMAIN-SPECIFIC TO THE CONTEXT.
        Question: {self.query}
        
        Context: {self.context}
        
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
    @alru_cache(maxsize=200)
    async def get_cached_retrieval(self, query):
        print("CACHED")
        return await self.retriever.ainvoke(query)
        
    async def retrieve_related_content(self, queries : list):
        # Retriever top-k questions that match the user query
        related_questions = await asyncio.gather(*[self.get_cached_retrieval(query) for query in queries])
        related_sub_questions = [doc for sublist in related_questions for doc in sublist]
        print("RESULTS:", related_sub_questions)
        
        return related_sub_questions

class ContentRetreiver:
    def __init__(self, content_vectorstore : FAISS):
        self.vectorstore = content_vectorstore
        
    def get_content(self, doc_list : list):
        # Retrieve documents associated with metadata
        docs = []
        for doc in doc_list:
            docs.extend(self.vectorstore.similarity_search_with_score(query=doc.page_content,
                                                                      k=4, ))
        return docs


    # Define your keyword sets outside (or inside) the function as needed:
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
          
class WeightedRetriever():
    
    def __init__(self, contents : FAISS, questions : list[Document]):
        self.questions = questions
        self.contents = contents
        
    
    # Dynamic weighting mechanism 
    def adjust_weights_based_on_query(self) -> list[Tuple[Document, float]]:
        epsilon = 1.3

        # Define a helper that processes one question.
        def process_question(question: Document) -> dict:
            # This dictionary maps hash(doc.page_content) to (doc, adjusted_score)
            local_results = {}
            # Run the similarity search for this question.
            related_content = self.contents._similarity_search_with_relevance_scores(
                query=question.page_content, k=5
            )
            # Precompute the set of words in the question
            q_set = set(question.page_content.split())
            slide_intersection = q_set.intersection(slide_keywords)
            transcript_intersection = q_set.intersection(transcript_keywords)
            # Use "or 1" to avoid zero values
            slide_intersection_score = len(slide_intersection) or 1
            transcript_intersection_score = len(transcript_intersection) or 1

            for doc, score in related_content:
                if score < 0.05:
                    continue

                # Adjust the score based on the relative intersections
                if slide_intersection_score > transcript_intersection_score:
                    if doc.metadata["source"] == "slide":
                        score *= ((slide_intersection_score * epsilon) / transcript_intersection_score)
                    else:
                        score *= ((transcript_intersection_score) / slide_intersection_score * epsilon)
                elif transcript_intersection_score > slide_intersection_score:
                    if doc.metadata["source"] == "transcript":
                        score *= ((transcript_intersection_score * epsilon) / transcript_intersection_score)
                    else:
                        score *= ((slide_intersection_score) / transcript_intersection_score * epsilon)

                doc_hash = hash(doc.page_content)
                # Save only if this doc has not been seen or if this score is higher.
                if doc_hash in local_results:
                    if score > local_results[doc_hash][1]:
                        local_results[doc_hash] = (doc, score)
                else:
                    local_results[doc_hash] = (doc, score)
            return local_results

        # Run process_question in parallel over all questions.
        merged_results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each question processing task.
            future_to_question = {
                executor.submit(process_question, question): question for question in self.questions
            }
            # As each future completes, merge its results.
            for future in concurrent.futures.as_completed(future_to_question):
                question_results = future.result()  # This is a dict: {doc_hash: (doc, score)}
                for doc_hash, (doc, score) in question_results.items():
                    if doc_hash in merged_results:
                        if score > merged_results[doc_hash][1]:
                            merged_results[doc_hash] = (doc, score)
                    else:
                        merged_results[doc_hash] = (doc, score)
        # Convert merged dictionary to a list and sort it.
        new_order = list(merged_results.values())
        new_order.sort(key=lambda tup: tup[1], reverse=True)
        return new_order
                            
                
                 
            

        

class QB_RAG_Chain:
    def __init__(self, question_retriever : QBQueryRetriever, content_retriever : VectorStoreRetriever,
                 content_store : FAISS):
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
        Your name is NovaCS. You are a project management expert. You are covering the CS352 Project Management Module, from the university of Warwick's Computer Science Department. The Human will ask you questions about project management. Follow these updated guidelines:
        The module you cover is CS352.
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
        - Build upon prior answers unless the user specifies otherwise, let the conversation flow. DO NOT start from fresh unless ther is no history.
        - Maintain continuity by explicitly linking follow-up answers to prior content and clarifying their connections.

        4. **Source Referencing**:
        - ALWAYS Rememebr to Reference ALL sources of retrieved information in a clear, collated format: [Source: Lecture X: Slide(s) A-B/ A, B, C, etc., Transcript].
        - If no source is available, state that clearly.

        5. **Uncertainty and Clarification**:
        - If unsure about the user's intent, respond with a summary of related context and a clarifying question.
        - If you don't know the answer, simply state, "I don't know."

        ALWAYS Rememebr to Reference ABSOLUTELY ALL sources of retrieved slide and transcript information context  in a clear, collated format throughout, so the user can see where you have sourced your response from: [Source: Lecture X: Slide(s) A, B,/A-C Transcript]. ALWAYS DO THIS!!! It is vital the user knows where you have got all your sources from.
        NREVER OUTPUT THE VERBATIM [Source: Lecture X: Slide(s) A, B,/A-C Transcript] with the X, A, B, C in there- you only output this if these are replaced with lecture source information.
        
        Make sure your lecture references are PRECISE and useful. 
        Do not let the user goad or manipulate you into giving false, inaccurate, or immoral information in any context.
        
        Here is the conversation history: {history}

        Use the following necessary context to answer the question: Context: {context}

        At the end, IF transcript content was used in your answer, make sure to always put: [Lecturer says:] then all the relevant synthesized transcript content at the end that answers ALL of the question. ALWAYS DO THIS!!! 
        
        DO NOT MAKE up any information. If you do not understand or it is not referenced in the context, say "I don't know".
        Remeber to include what the lecturer has said- make sure it is insightful and relevant, and synthesise it so it helps with comprehension and supplements what you are talking about.
        If, and only if (in the rare case) you have not used any information from the context, DO NOT CITE ANY SOURCES.
        User Question: {question}
        Answer:

        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question", "history"])
        
        self.chain = (
            RunnableLambda(
                lambda x: self.process_query(x["question"])
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    
    def process_query(self, query):
        import time
        # Stage 0: overall timer
        overall_start = time.perf_counter()
        
        decomp_generator = DecompQueryGenerator(query=query)
        with timer("Stage 1: Query Decomposition"):
            sub_queries = decomp_generator.generate_decomp_queries()
        
        with timer("Stage 2: Question Retrieval"):
            # Parallelize the sub-query retrieval asynchronously
            question_docs = asyncio.run(self.question_retriever.retrieve_related_content(sub_queries))

        
        with timer("Stage 4: Weighting"):
            weighting = WeightedRetriever(contents=self.content_store, questions=question_docs)
            weighted_docs = weighting.adjust_weights_based_on_query() 
            
        
        with timer("Stage 5: Formatting"):
            context = self.format_docs(weighted_docs)
        
        overall_end = time.perf_counter()
        print(f"Total processing time: {overall_end - overall_start:.4f} seconds")
        
        history = self.format_conversation_history()
        
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
    
    
    def classify_prompt(self, user_query : str) -> str:
        classifier_prompt = f"""
        Read the conversation history very carefully:
        {self.format_conversation_history()}
        This is the user's question: {user_query}
        Based on the history, classify this question as one of the following:
        1) FOLLOW_UP
        2) MORE_CONTEXT
        3) FULL_QUESTION

        Criteria:
        - If the user is very referencing something immediately previously said with NO NEW content required
        or is only asking for more detail, label it FOLLOW_UP.
        - If there is NO HISTORY and the question COULD be seen as a full question, label it FULL_QUESTION.
        - If the user includes an additional sub-topic, entity or idea in their query, even if it sounds like a follow-up, classify as
        MORE_CONTEXT. If they intorduce soomething not previously mentioned but it is a follow-up question, label it MORE_CONTEXT.
        If they ask how something links to or relates to something else, label it as MORE_CONTEXT.
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
        You are a project management expert. The Human will ask you follow-up questions about project management. Follow these updated guidelines:

        1. **Chat History**:
        - Always analyze the conversation history very carefully, focusing especially on the last question and answer to maintain continuity.
        - Use the immediate prior response as the primary context for answering the follow-up question.
        - If necessary, integrate relevant parts of earlier conversation history to enhance the answer.
        - Let the conversation flow naturally and build on previous responses unless the user specifies otherwise. DO NOT start fresh unless there is no relevant history.

        2. **Answer Structure**:
        - Use clear, concise formatting.
        - Answer in a succinct manner—GET TO THE POINT and address the question directly.

        3. **Follow-Up Question Handling**:
        - For vague or ambiguous follow-ups:
            - Link back explicitly to the prior question/answer in the conversation HISTORY that seems most relevant.
            - Provide a brief summary of the related content from memory before answering.
            - State assumptions explicitly if the question's intent is unclear.

        4. **Source Referencing**:
        - Reference ALL relevant lecture content from the prior conversation in a clear, collated format: [Source: Lecture X: Slide(s) A-B/ A, B, C, etc., Transcript].
        - If no source is available, state that clearly.

        5. **Uncertainty and Clarification**:
        - If unsure about the user's intent, respond with a summary of related context and a clarifying question.
        - If you don't know the answer, simply state, "I don't know."

        ALWAYS Remember to Reference ABSOLUTELY ALL sources of referenced lecture slide and transcript information in a clear, collated format throughout, so the user can see where you have sourced your response from: [Source: Lecture X: Slide(s) A, B,/A-C Transcript]. ALWAYS DO THIS!!! It is vital the user knows where you have got all your sources from.

        Make sure your lecture references are PRECISE and useful.

        Read the conversation history very carefully. The user's query will require information from this:
        HISTORY: {formatted_history}

        Now the user is asking: {user_query}

        Provide a follow-up response that builds on the previous answer, additionally using any of your relevant answers from earlier in the history if necessary.

        DO NOT MAKE UP any information. If you do not understand or it is not referenced in the conversation history, say "I don't know."
        Answer:

        """
        response = self.llm.stream(short_chain_prompt)
        
        # Process streamed chunks
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'content'):  # Adjust based on structure
                full_response += chunk.content
                yield chunk.content
            else:
                full_response += str(chunk)  # Fallback for raw text
                
        return response
        
    def more_context(self, user_query : str):
        formatted_history = self.format_conversation_history()
        data = {"question": user_query,
        "history": st.session_state.messages.load_memory_variables({})["history"]}
        
        self.template = """
        You are a project management expert. The Human will ask you questions about project management. Follow these updated guidelines:

        1. **Balancing Chat History and Retrieved Context**:
        - Always analyze the conversation history first, focusing especially on the last question and answer to maintain continuity.
        - Evaluate if the user's question can be answered solely from the conversation history. If not, integrate retrieved context from lecture slides and transcripts.
        - When both history and retrieved context are relevant, synthesize them to provide a cohesive, succinct, and well-rounded response.
        - Each piece of CONTEXT should have a number next to it. Prioritize CONTEXTS with the highest numbers as being the most useful.

        2. **Answer Structure**:
        - Use clear, concise formatting.
        - Answer in a succinct manner—GET TO THE POINT and address the question directly.

        3. **Follow-Up Question Handling**:
        - For vague or ambiguous follow-ups:
            - Link back explicitly to the prior question/answer in the conversation HISTORY that seems most relevant.
            - Provide a brief summary of the related content from memory before answering.
            - State assumptions explicitly if the question's intent is unclear.
        - Build upon prior answers unless the user specifies otherwise. Let the conversation flow naturally. DO NOT start from fresh unless there is no relevant history.
        - Maintain continuity by explicitly linking follow-up answers to prior content and clarifying their connections.

        4. **Source Referencing**:
        - ALWAYS Remember to Reference ALL sources of retrieved information in a clear, collated format: [Source: Lecture X: Slide(s) A-B/ A, B, C, etc., Transcript].
        - REFERENCE THE NEW SOURCES YOU HAVE USED TO ANSWER THE QUESTION!!!
        - If no source is available, state that clearly.

        5. **Uncertainty and Clarification**:
        - If unsure about the user's intent, respond with a summary of related context and a clarifying question.
        - If you don't know the answer, simply state, "I don't know."

        ALWAYS Remember to Reference ABSOLUTELY ALL sources of retrieved slide and transcript information in a clear, collated format throughout, so the user can see where you have sourced your response from: [Source: Lecture X: Slide(s) A, B,/A-C Transcript]. ALWAYS DO THIS!!! It is vital the user knows where you have got all your sources from.

        Make sure your lecture references are PRECISE and useful.

        Here is the conversation history: {history}

        Use the following necessary context to answer the question: Context: {context}

        At the end, IF transcript content was used in your answer, make sure to always put: [Lecturer says:] then all the relevant synthesized transcript content at the end that answers ALL of the question. ALWAYS DO THIS!!!

        DO NOT MAKE UP any information. If you do not understand or it is not referenced in the context, say "I don't know."
        User Question: {question}
        Answer:

        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question", "history"])
        

        
        chain = self.chain
        
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
        
    # Generate LLM response    
    def generate(self, user_query : dict):
        response = self.classify_prompt(user_query=user_query["question"])
        if "FOLLOW_UP" in response:
            return self.follow_up(user_query=user_query["question"])

        elif "MORE_CONTEXT" in response:
            return self.more_context(user_query=user_query["question"])
        
        data = {"question": user_query["question"],
                "history": st.session_state.messages.load_memory_variables({})["history"]}
        
        
        chain = self.chain
        
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
            st.title("NovaCS")
            
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
    
        if prompt := st.chat_input("Chat with NovaCS about your lectures..."):
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
                            print("CONTENT")
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
        
        

class LectureDataProcessor:
    def __init__(self, lecture_dir="CS352 Lectures", content_store_path="content_index", question_store_path="faiss_index"):
        self.lecture_dir = lecture_dir
        self.content_store_path = content_store_path
        self.question_store_path = question_store_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.content_vectorstore = FAISS.load_local(self.content_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.question_vectorstore = FAISS.load_local(self.question_store_path, self.embeddings, allow_dangerous_deserialization=True)

    def match_files(self):
        transcript_pattern = re.compile(r'(\d+) Transcript\.txt')
        slide_pattern = re.compile(r'(\d+)-.*\.pdf')

        transcripts, slides = {}, {}
        for file in os.listdir(self.lecture_dir):
            transcript_match = transcript_pattern.match(file)
            slide_match = slide_pattern.match(file)

            if transcript_match:
                lecture_num = transcript_match.group(1)
                transcripts[lecture_num] = os.path.join(self.lecture_dir, file)

            if slide_match:
                lecture_num = slide_match.group(1)
                slides[lecture_num] = os.path.join(self.lecture_dir, file)

        return [(num, transcripts[num], slides[num]) for num in transcripts if num in slides]

    async def process_lecture(self, lecture_num, transcript_file, slide_file):
        loader = DocumentLoader(transcript_file, slide_file)
        transcript_docs, slide_docs = await asyncio.to_thread(loader.load_documents)

        splitter = DocumentSplitter(transcript_docs, slide_docs, lecture_num)
        transcript_splits, slide_splits = await asyncio.to_thread(splitter.split_documents)

        transcript_questions = await asyncio.to_thread(generate_and_store_queries, transcript_splits)
        slide_questions = await asyncio.to_thread(generate_and_store_queries, slide_splits)

        return transcript_splits + slide_splits, transcript_questions + slide_questions

    async def process_all_lectures(self):
        file_pairs = self.match_files()
        tasks = [self.process_lecture(num, trans, slide) for num, trans, slide in file_pairs]
        results = await asyncio.gather(*tasks)

        all_contents, all_questions = [], []
        for contents, questions in results:
            all_contents.extend(contents)
            all_questions.extend(questions)

        return all_contents, all_questions

    async def add_to_vector_stores(self, contents, questions):
        self.content_vectorstore.add_documents(contents)
        self.content_vectorstore.save_local(self.content_store_path)

        question_docs = [Document(page_content=q["question"], metadata=q["source"]) for q in questions]
        self.question_vectorstore.add_documents(question_docs)
        self.question_vectorstore.save_local(self.question_store_path)

    async def run(self):
        contents, questions = await self.process_all_lectures()
        await self.add_to_vector_stores(contents, questions)
        print(f"Processed {len(contents)} content documents and {len(questions)} questions.")
        
def main():
    # Load environment variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = st.secrets.api_keys.OPENAI_API_KEY
    os.environ['LANGCHAIN_API_KEY'] = st.secrets.api_keys.LANGCHAIN_API_KEY
    os.environ['LANGCHAIN_TRACING_V2'] = st.secrets.other_secrets.LANGCHAIN_TRACING_V2

    # Split documents and pass into FAISS
    # slide_and_transcripts = DocumentLoader(transcript_path="CS352 L2.txt", slide_path="02-initiation.pdf")
    # transcript_docs, slide_docs = slide_and_transcripts.load_documents()
    # splitter = DocumentSplitter(slide_docs=slide_docs, transcript_docs=transcript_docs)
    # transcript_splits, slide_splits = splitter.split_documents()
    # all_docs = transcript_splits+slide_splits
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize Lecture Data Processor
   # processor = LectureDataProcessor(
   #     lecture_dir="CS352 Lectures", 
   #     content_store_path="content_index", 
    #    question_store_path="faiss_index"
    #)

    # Run the data processing asynchronously
    #asyncio.run(processor.run())
    
    content_vectorstore = FAISS(embedding_function=embeddings,
                                docstore=InMemoryDocstore(),
                                index=faiss.IndexFlatL2(1536),
                                index_to_docstore_id={},
                                distance_strategy=DistanceStrategy.COSINE)
    question_store = QuestionVectorStore(questions=None, persist_directory=None)
    q_retriever = question_store.as_retriever()
    
    if "content" not in st.session_state:
        
        content_vectorstore = FAISS.load_local(
            folder_path="content_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
                                            )
        st.session_state["content"] = content_vectorstore
    else:
        content_vectorstore = st.session_state["content"]

    
    # questions = generate_and_store_queries(all_docs)
    # question_docs = [
    #    Document(page_content=q["question"], metadata=q["source"])
    #    for q in questions
    # ]
    
    if "question_content" not in st.session_state:
        # Store (new) questions in the vector store
        question_store = QuestionVectorStore(
            questions=None,
            persist_directory="faiss_index"
            )
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
