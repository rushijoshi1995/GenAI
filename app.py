import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Import your existing functions
from F1_conv_bot import create_conv_retrieval_chain, refine_query

# Page config
st.set_page_config(page_title="Formula 1 RAG Chat", page_icon=":racing_car:", layout="wide")

st.title("Formula 1 RAG Chat App")

# Initialize session state for messages and message_history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Create retriever and document_chain (only once, not in session state)
retriever, document_chain = create_conv_retrieval_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process user input and generate response
def process_input(user_input):
    refined_query = refine_query(user_input, st.session_state.message_history)
    docs = retriever.invoke(refined_query)
    
    docs_with_sources = [
        Document(
            page_content=f"{doc.page_content}\n\nSource: {doc.metadata['source_url']}",
            metadata=doc.metadata
        )
        for doc in docs
    ]

    result = document_chain.invoke({
        "context": docs_with_sources,
        "messages": st.session_state.message_history
    })

    return result

# Chat input
if prompt := st.chat_input("Ask about Formula 1"):
    # Add user message to both messages and message_history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_history.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = process_input(prompt)
        
        # Simulate streaming
        for i in range(len(full_response)):
            message_placeholder.markdown(full_response[:i+1] + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant message to both messages and message_history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.message_history.append(AIMessage(content=full_response))

# Clear chat button
# if st.button("Clear Chat"):
#     st.session_state.messages = []
#     st.session_state.message_history = []
#     st.experimental_rerun()

# About section
# with st.expander("About this app"):
#     st.write("This is a Formula 1 RAG Chat App powered by AI. Ask questions about Formula 1!")