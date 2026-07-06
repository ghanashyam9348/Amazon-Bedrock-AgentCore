import os
from typing_extensions import TypedDict

from langchain_core.tools import tool

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# Import AgentCore runtime and memory integrations
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from dotenv import load_dotenv

_ = load_dotenv()

app = BedrockAgentCoreApp()
# AgentCore Memory Configuration
REGION = "ap-south-1"
MEMORY_ID = "gp_agent_memory-plxccz7mY3"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize short-term memory checkpointer
checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID, region_name=REGION)


faq_emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
faq_store = FAISS.load_local("./faiss_index", faq_emb, allow_dangerous_deserialization=True)


@tool
def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for relevant information.
    Use this tool when the user asks questions about products, services, or policies.
    
    Args:
        query: The search query to find relevant FAQ entries
        
    Returns:
        Relevant FAQ entries that might answer the question
    """
    results = faq_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Search the FAQ knowledge base with more results for complex queries.
    Use this when the initial search doesn't provide enough information.
    
    Args:
        query: The search query
        num_results: Number of results to retrieve (default: 5)
        
    Returns:
        More comprehensive FAQ entries
    """
    results = faq_store.similarity_search(query, k=num_results)
    
    if not results:
        return "No relevant FAQ entries found."
    
    context = "\n\n---\n\n".join([
        f"FAQ Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Reformulate the query to focus on a specific aspect.
    Use this when you need to search for a different angle of the question.
    
    Args:
        original_query: The original user question
        focus_aspect: The specific aspect to focus on (e.g., "pricing", "activation", "troubleshooting")
        
    Returns:
        A reformulated query focused on the specified aspect
    """
    reformulated = f"{focus_aspect} related to {original_query}"
    results = faq_store.similarity_search(reformulated, k=3)
    
    if not results:
        return f"No results found for aspect: {focus_aspect}"
    
    context = "\n\n---\n\n".join([
        f"Entry {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(results)
    ])
    
    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


tools = [search_faq, search_detailed_faq, reformulate_query]


# Initialize the LLM
llm = init_chat_model(
    model="openai/gpt-oss-20b", 
    model_provider="groq",
    api_key=GROQ_API_KEY
)

system_prompt = """You are a helpful FAQ assistant with access to a knowledge base and conversation memory.

Your goal is to answer user questions accurately using the available tools while remembering the current conversation.

Guidelines:
1. Use the current conversation context when it is relevant
2. Use the search_faq tool to find relevant information from the knowledge base
3. If the query is complex, use reformulate_query to search different aspects
4. Keep answers consistent with information already provided in this thread
5. Always provide a clear, concise answer based on the retrieved information
6. If you cannot find relevant information, clearly state that

Think step-by-step and use tools strategically to provide the best answer."""

# Create the agent with memory configurations
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt=system_prompt,
)


# AgentCore Entrypoint
@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime with memory support"""
    print("Received payload:", payload)
    print("Context:", context)
    
    # Extract query from payload
    query = payload.get("prompt", "No prompt found in input")
    
    # Extract thread_id for short-term memory scope
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))
    
    # Configure memory context
    config = {
        "configurable": {
            "thread_id": thread_id,  # Maps to AgentCore session_id
            "actor_id": actor_id     # Maps to AgentCore actor_id
        }
    }
    
    # Invoke the agent with memory
    result = agent.invoke(
        {"messages": [("human", query)]},
        config=config
    )
    
    print("Result:", result)
    
    # Extract the final answer from the result
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No response generated"
    
    # Return the answer
    return {
        "result": answer,
        "actor_id": actor_id,
        "thread_id": thread_id
    }


if __name__ == "__main__":
    app.run()