"""MongoDB client for storing conversation data and session transcripts."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo

from ..models.config import Settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for conversation and session data storage."""
    
    def __init__(self, mongodb_uri: str, database_name: str):
        self.uri = mongodb_uri
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for performance."""
        try:
            # Session indexes
            await self.db.sessions.create_index("session_id", unique=True)
            await self.db.sessions.create_index("start_time")
            await self.db.sessions.create_index("topic")
            
            # Message indexes
            await self.db.messages.create_index("session_id")
            await self.db.messages.create_index("timestamp")
            
            # Conversation indexes
            await self.db.conversations.create_index("session_id")
            await self.db.conversations.create_index("exchange_id")
            
            logger.info("Created MongoDB indexes")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
    
    async def create_session(self, session_data: Dict[str, Any]) -> bool:
        """Create a session document in MongoDB."""
        try:
            # Store full session data in MongoDB
            session_doc = {
                "session_id": session_data["id"],
                "title": session_data["title"],
                "topic": session_data["topic"],
                "start_time": session_data["start_time"],
                "end_time": session_data.get("end_time"),
                "status": session_data["status"],
                "messages": session_data.get("messages", []),
                "conversation_metadata": {
                    "total_exchanges": 0,
                    "concepts_discussed": [],
                    "metaphors_used": [],
                    "persona_adjustments": []
                },
                "analytics": {
                    "session_quality": None,
                    "learning_metrics": {},
                    "final_analysis": {}
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db.sessions.insert_one(session_doc)
            logger.info(f"Created session in MongoDB: {session_data['id']}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create session in MongoDB: {e}")
            return False
    
    async def add_message_exchange(self, session_id: str, user_input: str, assistant_response: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message exchange to the session."""
        try:
            exchange_doc = {
                "session_id": session_id,
                "exchange_id": f"{session_id}_{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow(),
                "user_input": user_input,
                "assistant_response": assistant_response,
                "metadata": metadata or {},
                "analysis": {
                    "concepts_extracted": [],
                    "metaphors_identified": [],
                    "ma_insights": []
                }
            }
            
            # Insert the exchange
            await self.db.conversations.insert_one(exchange_doc)
            
            # Update session with the new exchange
            await self.db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "messages": {
                            "timestamp": datetime.utcnow().isoformat(),
                            "user": user_input,
                            "assistant": assistant_response
                        }
                    },
                    "$inc": {"conversation_metadata.total_exchanges": 1},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message exchange: {e}")
            return False
    
    async def update_session_analysis(self, session_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Update session with analysis data from MA."""
        try:
            await self.db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "analytics": analysis_data,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update session analysis: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from MongoDB."""
        try:
            session_doc = await self.db.sessions.find_one({"session_id": session_id})
            return session_doc
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all message exchanges for a session."""
        try:
            cursor = self.db.conversations.find(
                {"session_id": session_id}
            ).sort("timestamp", 1)
            
            return await cursor.to_list(None)
        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")
            return []
    
    async def end_session(self, session_id: str, final_analysis: Dict[str, Any]) -> bool:
        """Mark session as completed with final analysis."""
        try:
            await self.db.sessions.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "end_time": datetime.utcnow(),
                        "status": "completed",
                        "analytics.final_analysis": final_analysis,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False
    
    async def clear_all_data(self) -> Dict[str, Any]:
        """Clear all data from MongoDB collections."""
        stats = {"sessions_deleted": 0, "conversations_deleted": 0, "collections_dropped": 0}
        
        try:
            # Get counts before deletion
            sessions_count = await self.db.sessions.count_documents({})
            conversations_count = await self.db.conversations.count_documents({})
            stats["sessions_deleted"] = sessions_count
            stats["conversations_deleted"] = conversations_count
            
            # Drop collections completely (faster than deleting documents)
            await self.db.sessions.drop()
            await self.db.conversations.drop()
            stats["collections_dropped"] = 2
            
            # Recreate collections with any required indexes
            await self.db.create_collection("sessions")
            await self.db.create_collection("conversations")
            
            # Create indexes for performance
            await self.db.sessions.create_index("session_id", unique=True)
            await self.db.conversations.create_index([("session_id", 1), ("timestamp", 1)])
            
            logger.info(f"Cleared MongoDB: {sessions_count} sessions, {conversations_count} conversations")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to clear MongoDB: {e}")
            raise
    
    async def search_sessions(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search sessions with query."""
        try:
            cursor = self.db.sessions.find(query).limit(limit).sort("start_time", -1)
            return await cursor.to_list(None)
        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []
    
    async def get_learning_trajectory(self, days_back: int = 30) -> Dict[str, Any]:
        """Get learning trajectory data over time."""
        try:
            from datetime import timedelta
            start_date = datetime.utcnow() - timedelta(days=days_back)
            
            pipeline = [
                {"$match": {"start_time": {"$gte": start_date}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$start_time"}},
                    "session_count": {"$sum": 1},
                    "total_exchanges": {"$sum": "$conversation_metadata.total_exchanges"},
                    "topics": {"$addToSet": "$topic"}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            cursor = self.db.sessions.aggregate(pipeline)
            results = await cursor.to_list(None)
            
            return {
                "period_days": days_back,
                "daily_stats": results,
                "total_sessions": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning trajectory: {e}")
            return {"error": str(e)}