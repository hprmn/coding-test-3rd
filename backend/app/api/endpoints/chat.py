"""
Chat API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import uuid
import json
from datetime import datetime
import redis
from app.db.session import get_db
from app.core.config import settings
from app.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationCreate,
    Conversation,
    ChatMessage
)
from app.services.query_engine import QueryEngine

router = APIRouter()

# Redis client for conversation storage
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    db: Session = Depends(get_db)
):
    """Process a chat query using RAG with Redis-backed conversation history"""

    # Get conversation history from Redis if conversation_id provided
    conversation_history = []
    if request.conversation_id:
        conv_key = f"conversation:{request.conversation_id}"
        conv_data = redis_client.get(conv_key)

        if conv_data:
            conversation = json.loads(conv_data)
            conversation_history = conversation.get("messages", [])

    # Process query
    query_engine = QueryEngine(db)
    response = await query_engine.process_query(
        query=request.query,
        fund_id=request.fund_id,
        conversation_history=conversation_history
    )

    # Update conversation history in Redis
    if request.conversation_id:
        conv_key = f"conversation:{request.conversation_id}"
        conv_data = redis_client.get(conv_key)

        if not conv_data:
            # Create new conversation
            conversation = {
                "conversation_id": request.conversation_id,
                "fund_id": request.fund_id,
                "messages": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        else:
            conversation = json.loads(conv_data)

        # Add new messages
        conversation["messages"].extend([
            {
                "role": "user",
                "content": request.query,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "role": "assistant",
                "content": response["answer"],
                "timestamp": datetime.utcnow().isoformat()
            }
        ])
        conversation["updated_at"] = datetime.utcnow().isoformat()

        # Store in Redis with 24 hour TTL
        redis_client.setex(
            conv_key,
            86400,  # 24 hours
            json.dumps(conversation)
        )

    return ChatQueryResponse(**response)


@router.post("/conversations", response_model=Conversation)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation in Redis"""
    conversation_id = str(uuid.uuid4())

    conversation = {
        "conversation_id": conversation_id,
        "fund_id": request.fund_id,
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    # Store in Redis with 24 hour TTL
    conv_key = f"conversation:{conversation_id}"
    redis_client.setex(
        conv_key,
        86400,  # 24 hours
        json.dumps(conversation)
    )

    return Conversation(
        conversation_id=conversation_id,
        fund_id=request.fund_id,
        messages=[],
        created_at=datetime.fromisoformat(conversation["created_at"]),
        updated_at=datetime.fromisoformat(conversation["updated_at"])
    )


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get conversation history from Redis"""
    conv_key = f"conversation:{conversation_id}"
    conv_data = redis_client.get(conv_key)

    if not conv_data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = json.loads(conv_data)

    return Conversation(
        conversation_id=conversation_id,
        fund_id=conv["fund_id"],
        messages=[ChatMessage(**msg) for msg in conv["messages"]],
        created_at=datetime.fromisoformat(conv["created_at"]),
        updated_at=datetime.fromisoformat(conv["updated_at"])
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation from Redis"""
    conv_key = f"conversation:{conversation_id}"

    if not redis_client.exists(conv_key):
        raise HTTPException(status_code=404, detail="Conversation not found")

    redis_client.delete(conv_key)

    return {"message": "Conversation deleted successfully"}
