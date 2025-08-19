import logging
from typing import Any, Dict
from uuid import uuid4

from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
)
from langchain_core.tools import tool
import json

# Configure logging
logger = logging.getLogger(__name__)

@tool
async def a2a_agent_tool(message: str, a2a_client: A2AClient) -> str:
    """Send a message to the A2A server and return the response.
    
    Use this tool when you need to interact with the A2A server to get information
    from web search, academic papers, or document retrieval.
    """
    try:
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'message_id': uuid4().hex,
            },
        }
        
        request = SendStreamingMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**send_message_payload)
        )
        
        response_stream = a2a_client.send_message_streaming(request)
        
        logger.info(f"Starting streaming request for message: {message}")
        
        # Collect all JSON dumps from the streaming response
        json_dumps = []
        
        async for chunk in response_stream:
            chunk_json = chunk.model_dump(mode='json', exclude_none=True)
            json_dumps.append(chunk_json)
            logger.debug(f"Received chunk: {chunk_json}")
        
        logger.info(f"Received {len(json_dumps)} chunks from A2A server")
        
        if not json_dumps:
            return "No response received from A2A server"
        
        # Return all JSON dumps as a formatted string
        return f"Received {len(json_dumps)} chunks:\n\n" + "\n\n".join(
            f"Chunk {i+1}:\n{json.dumps(chunk, indent=2)}" 
            for i, chunk in enumerate(json_dumps)
        )
        
    except Exception as e:
        logger.error(f"Error communicating with A2A server: {e}")
        return f"Error: {str(e)}"

def create_a2a_tool_with_client(a2a_client: A2AClient):
    """Create an A2A tool bound to the specific client."""
    async def a2a_tool_wrapper(message: str) -> str:
        """Send a message to the A2A server and return the response.
        
        Use this tool when you need to interact with the A2A server to get information
        from web search, academic papers, or document retrieval.
        
        Args:
            message: The message to send to the A2A server
        """
        return await a2a_agent_tool.ainvoke({"message": message, "a2a_client": a2a_client})
    
    return a2a_tool_wrapper
