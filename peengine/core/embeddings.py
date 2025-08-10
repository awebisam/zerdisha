"""Vector embedding service for u-vectors and c-vectors."""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..models.graph import Vector

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing vector embeddings."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=llm_config.azure_openai_key,
            azure_endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version
        )
        self.model = "text-embedding-ada-002"  # Standard embedding model
        self.dimension = 1536  # Ada-002 dimension
    
    async def create_u_vector(self, concept: str, user_explanation: str, metaphors: List[str]) -> Vector:
        """Create user's metaphorical understanding vector (u-vector)."""
        
        # Combine user's explanation with metaphors for u-vector
        u_text = f"Concept: {concept}\n"
        u_text += f"User explanation: {user_explanation}\n"
        u_text += f"Metaphors used: {', '.join(metaphors)}\n"
        u_text += "This represents the user's personal, metaphorical understanding."
        
        embedding = await self._get_embedding(u_text)
        
        return Vector(
            values=embedding,
            model=self.model,
            dimension=len(embedding)
        )
    
    async def create_c_vector(self, concept: str, canonical_definition: str, domain: str) -> Vector:
        """Create canonical academic vector (c-vector)."""
        
        # Use canonical definition for c-vector
        c_text = f"Concept: {concept}\n"
        c_text += f"Domain: {domain}\n" 
        c_text += f"Canonical definition: {canonical_definition}\n"
        c_text += "This represents the formal, academic understanding."
        
        embedding = await self._get_embedding(c_text)
        
        return Vector(
            values=embedding,
            model=self.model,
            dimension=len(embedding)
        )
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimension
    
    def calculate_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vector1.values or not vector2.values:
            return 0.0
        
        # Ensure same dimensions
        if len(vector1.values) != len(vector2.values):
            logger.warning("Vector dimension mismatch")
            return 0.0
        
        # Convert to numpy arrays
        v1 = np.array(vector1.values)
        v2 = np.array(vector2.values)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def calculate_gap_score(self, u_vector: Vector, c_vector: Vector) -> Dict[str, float]:
        """Calculate gap between user understanding and canonical knowledge."""
        similarity = self.calculate_similarity(u_vector, c_vector)
        
        # Gap score is inverse of similarity
        gap = 1.0 - similarity
        
        # Classify gap severity
        if gap < 0.2:
            severity = "low"
        elif gap < 0.5:
            severity = "medium"
        else:
            severity = "high"
        
        return {
            "similarity": similarity,
            "gap_score": gap,
            "severity": severity
        }
    
    async def generate_canonical_definition(self, concept: str, domain: str) -> str:
        """Generate canonical definition for creating c-vector."""
        
        prompt = f"""
Provide a concise, canonical definition of the concept "{concept}" in the domain of {domain}.

Requirements:
- Use formal academic terminology
- Include key properties and characteristics
- Be precise and unambiguous
- Limit to 2-3 sentences
- Focus on the essential, widely-accepted understanding

Concept: {concept}
Domain: {domain}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_config.azure_openai_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for canonical accuracy
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate canonical definition: {e}")
            return f"Standard definition of {concept} in {domain}"
    
    async def find_similar_concepts(
        self, 
        target_vector: Vector, 
        concept_vectors: Dict[str, Vector],
        top_k: int = 5
    ) -> List[tuple]:
        """Find concepts with similar vectors."""
        
        similarities = []
        
        for concept_name, vector in concept_vectors.items():
            similarity = self.calculate_similarity(target_vector, vector)
            similarities.append((concept_name, similarity))
        
        # Sort by similarity (highest first) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analyze_vector_clusters(self, vectors: Dict[str, Vector]) -> Dict[str, Any]:
        """Analyze vector clusters to identify conceptual groupings."""
        
        if len(vectors) < 2:
            return {"clusters": [], "analysis": "Insufficient data for clustering"}
        
        # Calculate pairwise similarities
        concepts = list(vectors.keys())
        n = len(concepts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_similarity(vectors[concepts[i]], vectors[concepts[j]])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        # Simple clustering: find highly similar pairs (similarity > 0.8)
        clusters = []
        used_concepts = set()
        
        for i in range(n):
            if concepts[i] in used_concepts:
                continue
                
            cluster = [concepts[i]]
            used_concepts.add(concepts[i])
            
            for j in range(i + 1, n):
                if concepts[j] not in used_concepts and similarity_matrix[i][j] > 0.8:
                    cluster.append(concepts[j])
                    used_concepts.add(concepts[j])
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return {
            "clusters": clusters,
            "similarity_matrix": similarity_matrix.tolist(),
            "analysis": f"Found {len(clusters)} concept clusters"
        }
    
    async def create_metaphor_bridge_vector(
        self, 
        source_concept: str, 
        target_concept: str, 
        metaphor: str
    ) -> Vector:
        """Create vector representing metaphorical connection between concepts."""
        
        bridge_text = f"""
Metaphorical bridge: {metaphor}
Source concept: {source_concept}
Target concept: {target_concept}
This represents the conceptual bridge that connects {source_concept} to {target_concept} through the metaphor "{metaphor}".
"""
        
        embedding = await self._get_embedding(bridge_text)
        
        return Vector(
            values=embedding,
            model=self.model,
            dimension=len(embedding)
        )