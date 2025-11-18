import streamlit as st
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from agent import rag_bot


@dataclass
class Message:
    actor: str
    payload: str
    
USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

# Setting up the UI page title and config 
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

# Initialize chat history
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

if "sources" not in st.session_state:
    st.session_state.sources = []


# msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)
    
agent_config = {"configurable": {"thread_id": 1}}

# Input box for user message
if prompt := st.chat_input("Type your message..."):
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)

    result = rag_bot.invoke({"messages": [HumanMessage(content=prompt)]}, config=agent_config)
    response: str = result["messages"][-1].content
    
    st.session_state.sources = result["sources"]
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))#, context=context))
    
    with st.chat_message(ASSISTANT):
        st.write(response)

    # Source documents
    with st.expander("Source documents"):
        st.write(st.session_state.sources)