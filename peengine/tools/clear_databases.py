"""Clear all data from Neo4j and MongoDB databases with strong confirmation."""

import asyncio
import logging
import secrets
import string
from typing import Dict, Any

from ..models.config import Settings
from ..database.neo4j_client import Neo4jClient
from ..database.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


def generate_confirmation_string() -> str:
    """Generate a random confirmation string."""
    # Mix of uppercase, lowercase, and numbers for readability
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    # 8 characters is enough to be secure but not too annoying
    return ''.join(secrets.choice(chars) for _ in range(8))


class DatabaseCleaner:
    """Clear all data from both databases with strong confirmation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.neo4j_client = Neo4jClient(settings.database_config)
        # MongoDB client needs URI and database name separately
        self.mongodb_client = MongoDBClient(
            settings.mongodb_uri, 
            settings.mongodb_database
        )
    
    async def clear_all_databases(self) -> Dict[str, Any]:
        """Clear all data from both Neo4j and MongoDB after confirmation."""
        
        # Generate random confirmation string
        confirmation_code = generate_confirmation_string()
        
        print("⚠️  WARNING: DATABASE DESTRUCTION IMMINENT ⚠️")
        print()
        print("This will PERMANENTLY DELETE:")
        print("  • All knowledge graph nodes and relationships (Neo4j)")
        print("  • All session conversations and transcripts (MongoDB)")  
        print("  • All imported knowledge graphs")
        print("  • All learning session history")
        print()
        print("🚨 THIS CANNOT BE UNDONE! 🚨")
        print()
        print(f"To confirm, type exactly: {confirmation_code}")
        
        user_input = input("Confirmation code: ").strip()
        
        if user_input != confirmation_code:
            print("❌ Confirmation failed. Databases are safe.")
            return {"cancelled": True, "reason": "confirmation_failed", "success": False}
        
        print()
        print("🔥 STARTING DATABASE DESTRUCTION...")
        print()
        
        results = {
            "neo4j_stats": {},
            "mongodb_stats": {},
            "success": True,
            "errors": []
        }
        
        try:
            # Clear Neo4j
            print("Clearing Neo4j knowledge graph...")
            self.neo4j_client.connect()
            neo4j_stats = self.neo4j_client.clear_all_data()
            results["neo4j_stats"] = neo4j_stats
            self.neo4j_client.close()
            print(f"✓ Neo4j cleared: {neo4j_stats['nodes_deleted']} nodes, {neo4j_stats['relationships_deleted']} relationships")
            
        except Exception as e:
            error_msg = f"Failed to clear Neo4j: {e}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        try:
            # Clear MongoDB
            print("Clearing MongoDB conversations...")
            await self.mongodb_client.connect()
            mongodb_stats = await self.mongodb_client.clear_all_data()
            results["mongodb_stats"] = mongodb_stats
            await self.mongodb_client.close()
            print(f"✓ MongoDB cleared: {mongodb_stats['sessions_deleted']} sessions, {mongodb_stats['conversations_deleted']} conversations")
            
        except Exception as e:
            error_msg = f"Failed to clear MongoDB: {e}"
            results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        if results["errors"]:
            results["success"] = False
            print()
            print("⚠️  Some errors occurred during cleanup:")
            for error in results["errors"]:
                print(f"   • {error}")
        else:
            print()
            print("🧹 COMPLETE DATABASE DESTRUCTION SUCCESSFUL")
            print("   All knowledge and conversations have been permanently deleted.")
            print("   You can now start fresh with 'zerdisha import-graphs' and 'zerdisha start <topic>'")
        
        return results


async def main():
    """Main clearing function for CLI usage."""
    settings = Settings()
    cleaner = DatabaseCleaner(settings)
    
    result = await cleaner.clear_all_databases()
    
    if result.get("cancelled"):
        print("Operation cancelled.")
        exit(0)
    elif not result.get("success"):
        print("Database clearing completed with errors.")
        exit(1)
    else:
        print("Database clearing completed successfully.")
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())