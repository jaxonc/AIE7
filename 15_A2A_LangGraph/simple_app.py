import asyncio
import logging
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from a2a.client import A2ACardResolver, A2AClient
from A2A_tool import create_a2a_tool_with_client

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the chat model for processing A2A responses
model = init_chat_model("openai:gpt-4o-mini")

async def main():
    # A2A server configuration
    base_url = 'http://localhost:10000'
    
    # Initialize A2A client
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        # Fetch the agent card
        try:
            logger.info(f'Fetching agent card from: {base_url}')
            agent_card = await resolver.get_agent_card()
            logger.info('Successfully fetched agent card')
            
            # Initialize A2A client
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            logger.info('A2AClient initialized successfully')
            
            # Create A2A tool bound to the client
            a2a_tool = create_a2a_tool_with_client(client)
            
            # Create tools list for the agent
            tools = [a2a_tool]
            
            def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
                """Call the model with tools and return the response."""
                response = model.bind_tools(tools).invoke(state["messages"])
                return {"messages": [response]}
            
            # Create the graph with agentic architecture
            builder = StateGraph(MessagesState)
            builder.add_node("call_model", call_model)
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "call_model")
            builder.add_conditional_edges(
                "call_model",
                tools_condition,
            )
            builder.add_edge("tools", "call_model")
            
            graph = builder.compile()
            
            # Test the agentic A2A interaction
            logger.info("Testing agentic A2A interaction through LangGraph...")
            
            # Test 1: Query that should trigger A2A tool
            test_query = "What are the latest developments in artificial intelligence?"
            logger.info(f"Sending query: {test_query}")
            
            initial_state = MessagesState(
                messages=[HumanMessage(content=test_query)]
            )
            
            result = await graph.ainvoke(initial_state)
            print(f"Agentic A2A Response: {result['messages'][-1].content}")
            
            # Test 2: Query that might not need A2A tool
            test_query2 = "Hello, how are you?"
            logger.info(f"Sending query: {test_query2}")
            
            initial_state2 = MessagesState(
                messages=[HumanMessage(content=test_query2)]
            )
            
            result2 = await graph.ainvoke(initial_state2)
            print(f"Agentic Response 2: {result2['messages'][-1].content}")
            
        except Exception as e:
            logger.error(f"Error initializing A2A client: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())