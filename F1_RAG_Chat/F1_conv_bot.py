import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


#! Reference link - https://python.langchain.com/docs/how_to/chatbots_retrieval/

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

def create_conv_retrieval_chain(k=5):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="YOUR_OPENAI_API_KEY")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # SYSTEM_TEMPLATE = """
    # Answer the user's questions based on the below context. 
    # If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know".
    
    # Important: Please provide sources for your answer based on the URLs in the context.

    # Context:
    # {context}
    # """

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context has relevant information to the question, stick to the information which is present in the context and don't add your own information.
    If the context doesn't contain any relevant information to the question, only answer if you already have information with sources OR just say "I don't know".
    
    Important: Please provide sources for your answer based on the URLs in the context.

    Context:
    {context}
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    return retriever, document_chain

def refine_query(query, conversation_history):

    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("""
    You are tasked with refining a user query for a Formula 1 racing database. The database contains race results, driver standings, team standings, and fastest laps. Consider the conversation history and the current query to generate a comprehensive search query.

    Conversation history:
    {conversation_history}

    Current query: {query}

    Your task:
    1. Analyze the conversation history and the current query.
    2. Identify key elements like specific Grand Prix, years, driver names, or result types (e.g., podium, winner).
    3. If the current query is a follow-up question, incorporate relevant context from previous queries.
    4. Generate a detailed search query that includes all relevant information, even if not explicitly stated in the current query.

    Use the following format for different query types:
    - Grand Prix results: "FORMULA 1 [GP Name] [Year] Race Results"
    - Driver standings: "[Year] DRIVER STANDINGS: [Driver Name]" (if specific driver)
    - Constructor standings: "[Year] CONSTRUCTOR STANDINGS: [Team Name]" (if specific team)
    - Fastest laps: "[Year] DHL FASTEST LAP AWARD" (if after 2006) or "[Year] FASTEST LAP" (if 2006 or earlier)

    Expand date ranges into individual years and interpret vague terms (e.g., "podium" as "top 3 positions").

    Provide only the refined query, nothing else.

    For example:

    If the user query is: "Who finished on the podium in Azerbaijan GP from 2020 to 2023?"

    The output should be a structured query:

    'FORMULA 1 Azerbaijan Grand Prix 2020 Race Results, FORMULA 1 Azerbaijan Grand Prix 2021 Race Results, FORMULA 1 Azerbaijan Grand Prix 2022 Race Results,FORMULA 1 Azerbaijan Grand Prix 2023 Race Results, top 3 positions in race results.

    Now, please provide the refined query based on the given conversation history and current query:
    """)
    
    chain = prompt | model | StrOutputParser()
    refined_query = chain.invoke({
        "conversation_history": "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in conversation_history]),
        "query": query
    })
    print("refined query =", refined_query)
    return refined_query

message_history = []
retriever, document_chain = create_conv_retrieval_chain()

def ask_question(query):
    global message_history

    refined_query = refine_query(query, message_history)
    docs = retriever.invoke(refined_query)
    # print("Retrieved documents - ",docs)
    # Add source URLs to document metadata
   # Format documents with sources, but keep them as Document objects
    docs_with_sources = [
        Document(
            page_content=f"{doc.page_content}\n\nSource: {doc.metadata['source_url']}",
            metadata=doc.metadata
        )
        for doc in docs
    ]

    message_history.append(HumanMessage(content=query))
    print("messages:", message_history)

    result = document_chain.invoke({
        "context": docs_with_sources,
        "messages": message_history
    })

    message_history.append(AIMessage(content=result))

    return result

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        answer = ask_question(user_query)
        print(answer)
        print("\n")