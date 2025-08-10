"""Vector embedding service for u-vectors and c-vectors."""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..models.graph import Vector, Node, NodeType

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing vector embeddings."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        
        # Initialize Azure OpenAI client with timeout
        self.client = AsyncAzureOpenAI(
            api_key=llm_config.azure_openai_key,
            azure_endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version,
            timeout=30.0  # 30 second timeout for all API calls
        )
        self.model = "text-embedding-ada-002"  # Standard embedding model
        self.dimension = 1536  # Ada-002 dimension
        
        # Performance optimizations: caching
        self._c_vector_cache: Dict[str, Dict[str, Any]] = {}  # Cache for canonical vectors
        self._canonical_definition_cache: Dict[str, Dict[str, Any]] = {}  # Cache for definitions
        self._cache_ttl_hours = 24  # Cache TTL in hours
    
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
        """Create canonical academic vector (c-vector) with stored definition and caching."""
        
        # Check cache first
        cache_key = f"{concept.lower()}_{domain.lower()}"
        cached_vector = self._get_cached_c_vector(cache_key)
        if cached_vector:
            logger.debug(f"Using cached c_vector for {concept} in {domain}")
            return cached_vector
        
        # Generate canonical definition first (with caching)
        canonical_definition = await self.generate_canonical_definition(concept, domain)
        
        # Create vector from definition
        c_text = f"Concept: {concept}\n"
        c_text += f"Domain: {domain}\n" 
        c_text += f"Canonical definition: {canonical_definition}\n"
        c_text += "This represents the formal, academic understanding."
        
        embedding = await self._get_embedding(c_text)
        
        c_vector = Vector(
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
        
        # Cache the vector
        self._cache_c_vector(cache_key, c_vector)
        logger.debug(f"Cached new c_vector for {concept} in {domain}")
        
        return c_vector
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API with comprehensive error handling, fallbacks, and timeout."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        # Truncate text if too long (OpenAI has token limits)
        if len(text) > 8000:  # Conservative limit
            logger.warning(f"Text too long for embedding ({len(text)} chars), truncating")
            text = text[:8000] + "..."
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use asyncio.wait_for for additional timeout protection
                response = await asyncio.wait_for(
                    self.client.embeddings.create(
                        model=self.model,
                        input=text.strip()
                    ),
                    timeout=30.0  # 30 second timeout
                )
                
                embedding = response.data[0].embedding
                
                # Validate embedding
                if not embedding or len(embedding) != self.dimension:
                    logger.error(f"Invalid embedding received: length {len(embedding) if embedding else 0}, expected {self.dimension}")
                    if attempt == max_retries - 1:
                        return [0.0] * self.dimension
                    continue
                
                # Check for all-zero embedding (shouldn't happen but let's be safe)
                if all(x == 0.0 for x in embedding):
                    logger.warning("Received all-zero embedding from API")
                    if attempt == max_retries - 1:
                        return self._generate_fallback_embedding(text)
                    continue
                
                logger.debug(f"Successfully generated embedding for text ({len(text)} chars)")
                return embedding
                
            except asyncio.TimeoutError:
                logger.warning(f"Embedding API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))
                    continue
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_msg or "429" in error_msg:
                    delay = base_delay * (2 ** attempt) + (attempt * 0.5)  # Exponential backoff with jitter
                    logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                
                elif "timeout" in error_msg or "connection" in error_msg:
                    delay = base_delay * (attempt + 1)
                    logger.warning(f"Connection issue, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                
                elif "invalid" in error_msg or "400" in error_msg:
                    logger.error(f"Invalid request for embedding: {e}")
                    # Don't retry for invalid requests
                    break
                
                elif "unauthorized" in error_msg or "401" in error_msg or "403" in error_msg:
                    logger.error(f"Authentication error for embedding API: {e}")
                    # Don't retry for auth errors
                    break
                
                else:
                    logger.error(f"Embedding API error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (attempt + 1))
                        continue
        
        # All retries failed, return fallback
        logger.error(f"Failed to get embedding after {max_retries} attempts, using fallback")
        return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate a simple fallback embedding when API is unavailable."""
        import hashlib
        import struct
        
        # Create a deterministic but varied embedding based on text content
        # This won't be semantically meaningful but will be consistent
        
        # Hash the text to get deterministic values
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()
        
        # Convert hash bytes to floats
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 4), 4):
            if i + 4 <= len(text_hash):
                # Convert 4 bytes to float
                float_val = struct.unpack('f', text_hash[i:i+4])[0]
                # Normalize to reasonable range
                normalized_val = max(-1.0, min(1.0, float_val / 1000.0))
                embedding.append(normalized_val)
        
        # Pad with zeros if needed
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        # Truncate if too long
        embedding = embedding[:self.dimension]
        
        logger.info(f"Generated fallback embedding with {len(embedding)} dimensions")
        return embedding
    
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
        """Generate canonical definition for creating c-vector with comprehensive error handling and caching."""
        
        if not concept or not concept.strip():
            logger.warning("Empty concept provided for canonical definition")
            return f"No concept specified for {domain} domain"
        
        if not domain or not domain.strip():
            domain = "general"
            logger.info(f"No domain specified for {concept}, using 'general'")
        
        # Check cache first
        cache_key = f"{concept.lower()}_{domain.lower()}"
        cached_definition = self._get_cached_canonical_definition(cache_key)
        if cached_definition:
            logger.debug(f"Using cached canonical definition for {concept} in {domain}")
            return cached_definition
        
        # Dynamically generate domain-specific guidance
        domain_guidance = f"You are an expert in {domain}. Your task is to provide a canonical definition of the concept '{concept}' as it is understood in this domain."

        prompt = f"""
{domain_guidance}

Please adhere to the following requirements:
- The definition must be precise, formal, and use academic terminology appropriate for the {domain} domain.
- It should be comprehensive yet concise, ideally 2-3 sentences.
- The definition must focus on the widely-accepted, canonical understanding of the concept within {domain}.
- Avoid using any metaphorical or figurative language.
- Where applicable, include key relationships to other fundamental concepts in the domain.
- The output should be only the definition itself.

Concept: "{concept}"
Domain: {domain}

Canonical Definition:"""
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use asyncio.wait_for for additional timeout protection
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.llm_config.azure_openai_deployment_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,  # Low temperature for canonical accuracy
                        max_tokens=250,  # Slightly increased for domain-specific detail
                    ),
                    timeout=30.0  # 30 second timeout
                )
                
                definition = response.choices[0].message.content.strip()
                
                # Validate definition quality
                if not definition or len(definition) < 10:
                    logger.warning(f"Generated definition too short for {concept}: '{definition}'")
                    if attempt == max_retries - 1:
                        return self._generate_fallback_definition(concept, domain)
                    continue
                
                # Check for obvious errors or non-definitions
                if any(phrase in definition.lower() for phrase in ['i cannot', 'i am unable', 'sorry', 'error']):
                    logger.warning(f"Generated definition contains error indicators for {concept}")
                    if attempt == max_retries - 1:
                        return self._generate_fallback_definition(concept, domain)
                    continue
                
                # Cache the definition
                self._cache_canonical_definition(cache_key, definition)
                logger.info(f"Generated and cached canonical definition for {concept} in {domain}")
                return definition
                
            except asyncio.TimeoutError:
                logger.warning(f"Canonical definition API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))
                    continue
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_msg or "429" in error_msg:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit for canonical definition, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                
                elif "timeout" in error_msg or "connection" in error_msg:
                    delay = base_delay * (attempt + 1)
                    logger.warning(f"Connection timeout for canonical definition, retrying in {delay:.1f}s: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                
                elif "unauthorized" in error_msg or "401" in error_msg or "403" in error_msg:
                    logger.error(f"Authentication error for canonical definition: {e}")
                    break  # Don't retry auth errors
                
                else:
                    logger.error(f"Error generating canonical definition (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (attempt + 1))
                        continue
        
        # All attempts failed, use fallback
        logger.error(f"Failed to generate canonical definition for {concept} after {max_retries} attempts")
        return self._generate_fallback_definition(concept, domain)
    
    def _generate_fallback_definition(self, concept: str, domain: str) -> str:
        """Generate a basic fallback definition when LLM is unavailable."""
        logger.info(f"Using fallback definition for {concept} in {domain}")
        return f"The concept of \"{concept}\" as understood in the academic domain of {domain}."
    
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
    
    def _get_cached_c_vector(self, cache_key: str) -> Optional[Vector]:
        """Get cached canonical vector if still valid."""
        if cache_key not in self._c_vector_cache:
            return None
        
        cached_data = self._c_vector_cache[cache_key]
        cache_time = cached_data.get("timestamp")
        
        if not cache_time:
            # Invalid cache entry, remove it
            del self._c_vector_cache[cache_key]
            return None
        
        # Check if cache is still valid (within TTL)
        if datetime.now() - cache_time > timedelta(hours=self._cache_ttl_hours):
            logger.debug(f"Cache expired for c_vector: {cache_key}")
            del self._c_vector_cache[cache_key]
            return None
        
        return cached_data.get("vector")
    
    def _cache_c_vector(self, cache_key: str, vector: Vector) -> None:
        """Cache canonical vector with timestamp."""
        self._c_vector_cache[cache_key] = {
            "vector": vector,
            "timestamp": datetime.now()
        }
        
        # Clean up old cache entries if cache gets too large
        if len(self._c_vector_cache) > 100:  # Max 100 cached vectors
            self._cleanup_c_vector_cache()
    
    def _get_cached_canonical_definition(self, cache_key: str) -> Optional[str]:
        """Get cached canonical definition if still valid."""
        if cache_key not in self._canonical_definition_cache:
            return None
        
        cached_data = self._canonical_definition_cache[cache_key]
        cache_time = cached_data.get("timestamp")
        
        if not cache_time:
            # Invalid cache entry, remove it
            del self._canonical_definition_cache[cache_key]
            return None
        
        # Check if cache is still valid (within TTL)
        if datetime.now() - cache_time > timedelta(hours=self._cache_ttl_hours):
            logger.debug(f"Cache expired for canonical definition: {cache_key}")
            del self._canonical_definition_cache[cache_key]
            return None
        
        return cached_data.get("definition")
    
    def _cache_canonical_definition(self, cache_key: str, definition: str) -> None:
        """Cache canonical definition with timestamp."""
        self._canonical_definition_cache[cache_key] = {
            "definition": definition,
            "timestamp": datetime.now()
        }
        
        # Clean up old cache entries if cache gets too large
        if len(self._canonical_definition_cache) > 200:  # Max 200 cached definitions
            self._cleanup_canonical_definition_cache()
    
    def _cleanup_c_vector_cache(self) -> None:
        """Remove oldest cache entries to keep cache size manageable."""
        if len(self._c_vector_cache) <= 50:  # Keep at most 50 entries
            return
        
        # Sort by timestamp and keep only the newest 50
        sorted_items = sorted(
            self._c_vector_cache.items(),
            key=lambda x: x[1].get("timestamp", datetime.min),
            reverse=True
        )
        
        self._c_vector_cache = dict(sorted_items[:50])
        logger.debug(f"Cleaned up c_vector cache, kept {len(self._c_vector_cache)} entries")
    
    def _cleanup_canonical_definition_cache(self) -> None:
        """Remove oldest cache entries to keep cache size manageable."""
        if len(self._canonical_definition_cache) <= 100:  # Keep at most 100 entries
            return
        
        # Sort by timestamp and keep only the newest 100
        sorted_items = sorted(
            self._canonical_definition_cache.items(),
            key=lambda x: x[1].get("timestamp", datetime.min),
            reverse=True
        )
        
        self._canonical_definition_cache = dict(sorted_items[:100])
        logger.debug(f"Cleaned up canonical definition cache, kept {len(self._canonical_definition_cache)} entries")
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all caches and return statistics."""
        c_vector_count = len(self._c_vector_cache)
        definition_count = len(self._canonical_definition_cache)
        
        self._c_vector_cache.clear()
        self._canonical_definition_cache.clear()
        
        logger.info(f"Cleared embedding service caches: {c_vector_count} c_vectors, {definition_count} definitions")
        return {
            "c_vectors_cleared": c_vector_count,
            "definitions_cleared": definition_count
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "c_vector_cache_size": len(self._c_vector_cache),
            "definition_cache_size": len(self._canonical_definition_cache),
            "cache_ttl_hours": self._cache_ttl_hours,
            "max_c_vector_cache": 100,
            "max_definition_cache": 200
        }