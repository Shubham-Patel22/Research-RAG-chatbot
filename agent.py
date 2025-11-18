from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
#from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage #, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import AgentState
from langgraph.graph import END, StateGraph
#from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# loading the environment variables
load_dotenv()

# setting the llm
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# setting the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# getting connection to the FAISS vector store
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# defining state variables in a class
class State(AgentState):
    messages: list[BaseMessage]
    sources: list[str]

'''@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join((f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs)
    return {"content": serialized, "artifact": retrieved_docs}'''

# Step 1: Generate an tool Message that includes a tool-call to be sent.
def query_fetch(query: str)->dict:
    """Always retrieve before answering."""
    if not query:   
        return {"content": "", "sources": []}
    
    retrieved_docs = vector_store.similarity_search(query, k=2)
    retrieved_content = ''
    sources_consulted = []
    if retrieved_docs:
        retrieved_content = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        for doc in retrieved_docs:
            source = doc.metadata['title'] + ' - ' + ', '.join(doc.metadata['author'])
            if source not in sources_consulted:
                sources_consulted.append(source)
    return {"content": retrieved_content, "sources": sources_consulted}

# Step 2: Execute the retrieval.
# tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: State):
    # Get generated ToolMessages
    '''recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]'''

    # Extract latest human message
    query = None
    for m in reversed(state["messages"]):
        if m.type == "human":
            query = m.content
            break

    if not query:
        return {"messages": []}

    fetched_context = query_fetch(query)
    # Format into prompt
    #docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = ("You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        f"{fetched_context['content']}")
    conversation_messages = [message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response], "sources": fetched_context['sources']}

# Build graph
graph_builder = StateGraph(State)
#graph_builder.add_node(query_fetch)
#graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("generate")
'''graph_builder.add_conditional_edges(
    "query_fetch",
    tools_condition,
    {END: END, "tools": "tools"})'''
#graph_builder.add_edge("query_fetch", "generate")
graph_builder.add_edge("generate", END)
rag_bot = graph_builder.compile(checkpointer=MemorySaver())