"""Vector embedding service for u-vectors and c-vectors."""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..models.graph import Vector, Node, NodeType

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
    
    async def create_c_vector(self, concept: str, domain: str) -> Vector:
        """Create canonical academic vector (c-vector) with stored definition."""
        
        # Generate canonical definition first
        canonical_definition = await self.generate_canonical_definition(concept, domain)
        
        # Create vector from definition
        c_text = f"Concept: {concept}\n"
        c_text += f"Domain: {domain}\n" 
        c_text += f"Canonical definition: {canonical_definition}\n"
        c_text += "This represents the formal, academic understanding."
        
        embedding = await self._get_embedding(c_text)
        
        return Vector(
            values=embedding,
            model=self.model,
            dimension=len(embedding),
            metadata={
                "canonical_definition": canonical_definition,
                "concept": concept,
                "domain": domain,
                "vector_type": "c_vector"
            }
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
    
    def calculate_gap_score(self, u_vector: Vector, c_vector: Vector) -> Dict[str, Any]:
        """Calculate gap between user understanding and canonical knowledge with detailed analysis."""
        
        # Calculate cosine similarity
        similarity = self.calculate_similarity(u_vector, c_vector)
        
        # Gap score is inverse of similarity
        gap_score = 1.0 - similarity
        
        # Classify gap severity with more nuanced categories
        if similarity >= 0.85:
            severity = "minimal"
            severity_description = "Very close alignment with canonical understanding"
        elif similarity >= 0.7:
            severity = "low"
            severity_description = "Good alignment with minor differences"
        elif similarity >= 0.5:
            severity = "moderate"
            severity_description = "Noticeable differences from canonical understanding"
        elif similarity >= 0.3:
            severity = "high"
            severity_description = "Significant gaps in understanding"
        else:
            severity = "critical"
            severity_description = "Major misalignment with canonical knowledge"
        
        # Calculate additional metrics
        u_magnitude = np.linalg.norm(u_vector.values) if u_vector.values else 0.0
        c_magnitude = np.linalg.norm(c_vector.values) if c_vector.values else 0.0
        
        # Determine conceptual emphasis differences
        emphasis_difference = abs(u_magnitude - c_magnitude) / max(c_magnitude, 1e-8)
        
        return {
            "similarity": float(similarity),
            "gap_score": float(gap_score),
            "severity": severity,
            "severity_description": severity_description,
            "u_vector_magnitude": float(u_magnitude),
            "c_vector_magnitude": float(c_magnitude),
            "emphasis_difference": float(emphasis_difference),
            "canonical_definition": c_vector.metadata.get("canonical_definition", ""),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def generate_canonical_definition(self, concept: str, domain: str) -> str:
        """Generate canonical definition for creating c-vector with domain-specific prompting."""
        
        # Domain-specific prompting templates
        domain_templates = {
            "physics": "Focus on fundamental laws, mathematical relationships, and measurable properties. Include units and quantitative aspects where applicable.",
            "chemistry": "Emphasize molecular structure, chemical properties, reaction mechanisms, and thermodynamic considerations.",
            "biology": "Highlight biological processes, evolutionary context, cellular mechanisms, and physiological functions.",
            "mathematics": "Define in terms of formal mathematical properties, axioms, theorems, and logical relationships.",
            "philosophy": "Present the concept through logical analysis, key philosophical positions, and conceptual distinctions.",
            "computer_science": "Focus on algorithmic properties, computational complexity, data structures, and implementation considerations.",
            "psychology": "Emphasize cognitive processes, behavioral patterns, empirical findings, and theoretical frameworks.",
            "economics": "Include market mechanisms, quantitative models, economic principles, and empirical relationships.",
            "general": "Provide a comprehensive academic definition with essential properties and characteristics."
        }
        
        domain_guidance = domain_templates.get(domain.lower(), domain_templates["general"])
        
        prompt = f"""
Provide a precise, canonical definition of "{concept}" in the {domain} domain.

Domain-specific requirements:
{domain_guidance}

General requirements:
- Use formal academic terminology appropriate to {domain}
- Include essential properties and characteristics
- Be concise but comprehensive (2-3 sentences)
- Focus on widely-accepted understanding in {domain}
- Avoid metaphorical language
- Include key relationships to other concepts in the domain

Domain context: {domain}
Concept: {concept}

Canonical definition:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_config.azure_openai_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for canonical accuracy
                max_tokens=250  # Slightly increased for domain-specific detail
            )
            
            definition = response.choices[0].message.content.strip()
            logger.info(f"Generated canonical definition for {concept} in {domain}")
            return definition
            
        except Exception as e:
            logger.error(f"Failed to generate canonical definition: {e}")
            return f"Standard definition of {concept} in {domain} domain"
    
    def find_similar_concepts(
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
    
    async def save_c_vector_to_neo4j(
        self, 
        concept: str, 
        domain: str, 
        c_vector: Vector, 
        neo4j_client
    ) -> Optional[Node]:
        """Save canonical vector to Neo4j with proper node structure."""
        
        try:
            # Create a unique node ID for the canonical vector
            node_id = f"c_vector_{concept.lower().replace(' ', '_')}_{domain.lower()}"
            
            # Create node with canonical vector
            node = Node(
                id=node_id,
                label=f"{concept} (Canonical)",
                node_type=NodeType.CONCEPT,
                properties={
                    "concept": concept,
                    "domain": domain,
                    "vector_type": "canonical",
                    "canonical_definition": c_vector.metadata.get("canonical_definition", ""),
                    "created_by": "embedding_service"
                },
                c_vector=c_vector,
                u_vector=None  # Only canonical vector for this node
            )
            
            # Save to Neo4j
            success = neo4j_client.create_node(node)
            
            if success:
                logger.info(f"Saved canonical vector for {concept} in {domain} to Neo4j")
                return node
            else:
                logger.error(f"Failed to save canonical vector for {concept} to Neo4j")
                return None
                
        except Exception as e:
            logger.error(f"Error saving canonical vector to Neo4j: {e}")
            return None
    
    async def get_or_create_c_vector(
        self, 
        concept: str, 
        domain: str, 
        neo4j_client
    ) -> Optional[Vector]:
        """Get existing c_vector from Neo4j or create and save a new one."""
        
        try:
            # Try to find existing canonical vector node
            node_id = f"c_vector_{concept.lower().replace(' ', '_')}_{domain.lower()}"
            existing_node_data = neo4j_client.get_node(node_id)
            
            if existing_node_data and existing_node_data.get("c_vector"):
                # Return existing c_vector
                c_vector_data = existing_node_data["c_vector"]
                logger.info(f"Found existing canonical vector for {concept} in {domain}")
                return Vector(**c_vector_data)
            
            # Create new canonical vector
            logger.info(f"Creating new canonical vector for {concept} in {domain}")
            c_vector = await self.create_c_vector(concept, domain)
            
            # Save to Neo4j
            saved_node = await self.save_c_vector_to_neo4j(concept, domain, c_vector, neo4j_client)
            
            if saved_node:
                return c_vector
            else:
                logger.warning(f"Created c_vector but failed to save to Neo4j for {concept}")
                return c_vector  # Return the vector even if saving failed
                
        except Exception as e:
            logger.error(f"Error in get_or_create_c_vector: {e}")
            return None