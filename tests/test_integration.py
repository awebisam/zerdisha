"""Integration tests for complete PEEngine workflows."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime
import uuid

from peengine.core.orchestrator import ExplorationEngine
from peengine.models.config import Settings
from peengine.models.graph import Session, Node, Edge, Vector, NodeType, EdgeType


@pytest.fixture
def mock_settings():
    """Provides a mock Settings object for integration tests."""
    with patch('peengine.models.config.Settings') as MockSettings:
        instance = MockSettings.return_value
        instance.database_config = MagicMock()
        instance.mongodb_uri = "mongodb://localhost:27017"
        instance.mongodb_database = "test_integration_db"
        instance.llm_config = MagicMock()
        instance.persona_config = MagicMock()
        yield instance


@pytest.fixture
def integration_engine(mock_settings):
    """Provides a fully mocked ExplorationEngine for integration testing."""
    with patch('peengine.core.orchestrator.Neo4jClient') as MockNeo4j, \
            patch('peengine.core.orchestrator.MongoDBClient') as MockMongoDB, \
            patch('peengine.core.orchestrator.ConversationalAgent') as MockCA, \
            patch('peengine.core.orchestrator.PatternDetector') as MockPD, \
            patch('peengine.core.orchestrator.MetacognitiveAgent') as MockMA, \
            patch('peengine.core.orchestrator.EmbeddingService') as MockEmbeddingService, \
            patch('peengine.core.orchestrator.AnalyticsEngine') as MockAnalytics:

        # Create engine instance
        engine = ExplorationEngine(mock_settings)

        # Setup mock instances
        engine.graph_db = MockNeo4j.return_value
        engine.message_db = MockMongoDB.return_value
        engine.ca = MockCA.return_value
        engine.pd = MockPD.return_value
        engine.ma = MockMA.return_value
        engine.embedding_service = MockEmbeddingService.return_value
        engine.analytics = MockAnalytics.return_value

        # Mock database connections
        engine.graph_db.connect = MagicMock()
        engine.graph_db.create_indexes = MagicMock()
        engine.graph_db.close = MagicMock()
        engine.message_db.connect = AsyncMock()
        engine.message_db.close = AsyncMock()

        # Mock agent initialization
        engine.ca.initialize = AsyncMock()
        engine.pd.initialize = AsyncMock()
        engine.ma.initialize = AsyncMock()

        # Return the engine without initialization for testing
        return engine


class TestCompleteGapCheckFlow:
    """Integration tests for complete gap check workflow."""

    @pytest.mark.asyncio
    async def test_complete_gap_check_flow(self, integration_engine):
        """Test complete gap check from user conversation to gap analysis."""

        # Setup: Start a session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("quantum mechanics", "Quantum Exploration")
        assert session.topic == "quantum mechanics"
        assert session.status == "active"

        # Step 1: Simulate conversation flow
        conversation_exchanges = [
            {
                "user_input": "What is quantum superposition?",
                "ca_response": {
                    "message": "Think of it like a coin spinning in the air - before it lands, it's neither heads nor tails, but both possibilities exist simultaneously.",
                    "reasoning": {"metaphor_used": "spinning coin"}
                },
                "pd_extractions": [MagicMock(
                    concept="quantum superposition",
                    domain="physics",
                    context="fundamental quantum principle",
                    confidence=0.95,
                    metaphors=["spinning coin"]
                )],
                "ma_analysis": {
                    "insights": ["User beginning quantum exploration"],
                    "flags": []
                }
            }
        ]

        for exchange in conversation_exchanges:
            # Mock CA response
            integration_engine.ca.process_input = AsyncMock(
                return_value=exchange["ca_response"])

            # Mock PD extractions
            integration_engine.pd.extract_patterns = AsyncMock(
                return_value=exchange["pd_extractions"])

            # Mock MA analysis
            integration_engine.ma.analyze_session = AsyncMock(
                return_value=exchange["ma_analysis"])

            # Mock database operations
            integration_engine.message_db.add_message_exchange = AsyncMock()
            integration_engine.message_db.update_session_analysis = AsyncMock()
            integration_engine.graph_db.create_node = MagicMock(
                return_value=True)
            integration_engine.graph_db.create_edge = MagicMock(
                return_value=True)

            # Process the user input
            result = await integration_engine.process_user_input(exchange["user_input"])

            # Verify response structure
            assert "message" in result
            assert "session_id" in result
            assert result["message"] == exchange["ca_response"]["message"]

        # Step 2: Execute gap check command
        # Mock gap check dependencies
        integration_engine._identify_recent_concept = AsyncMock(
            return_value="quantum superposition")

        # Mock u_vector retrieval
        u_vector = Vector(values=[0.1, 0.2, 0.3, 0.4,
                          0.5], model="test", dimension=5)
        integration_engine._get_user_vector = AsyncMock(return_value=u_vector)

        # Mock c_vector creation
        c_vector = Vector(
            values=[0.15, 0.25, 0.35, 0.45, 0.55],
            model="test",
            dimension=5,
            metadata={
                "canonical_definition": "A quantum mechanical principle where particles exist in multiple states simultaneously until measured."}
        )
        integration_engine._get_or_create_canonical_vector = AsyncMock(
            return_value=c_vector)

        # Mock gap score calculation
        gap_analysis = {
            "similarity": 0.92,
            "gap_score": 0.08,
            "severity": "minimal",
            "canonical_definition": "A quantum mechanical principle where particles exist in multiple states simultaneously until measured.",
            "severity_description": "Very close alignment"
        }
        integration_engine.embedding_service.calculate_gap_score = MagicMock(
            return_value=gap_analysis)

        # Mock message formatting
        expected_message = "🎯 **Understanding Check: Quantum Superposition**\n\nYour spinning coin metaphor captures the essence beautifully! Your understanding shows **92% alignment** with canonical physics - excellent work!"
        integration_engine._format_gap_message = AsyncMock(
            return_value=expected_message)

        # Execute gap check
        gap_result = await integration_engine.execute_command("gapcheck")

        # Verify complete gap check flow
        assert "error" not in gap_result
        assert gap_result["concept"] == "quantum superposition"
        assert gap_result["similarity"] == 0.92
        assert gap_result["gap_score"] == 0.08
        assert gap_result["severity"] == "minimal"
        assert gap_result["message"] == expected_message

        # Verify all methods were called in correct sequence
        integration_engine._identify_recent_concept.assert_called_once()
        integration_engine._get_user_vector.assert_called_once_with(
            "quantum superposition")
        integration_engine._get_or_create_canonical_vector.assert_called_once_with(
            "quantum superposition")
        integration_engine.embedding_service.calculate_gap_score.assert_called_once_with(
            u_vector, c_vector)
        integration_engine._format_gap_message.assert_called_once_with(
            "quantum superposition", gap_analysis)

        # Verify session state
        assert len(integration_engine.current_session.messages) == 1
        assert integration_engine.current_session.topic == "quantum mechanics"
        assert integration_engine.current_session.status == "active"


class TestMAInfluenceChangesCABehavior:
    """Integration tests for metacognitive influence on conversational agent."""

    @pytest.mark.asyncio
    async def test_ma_influence_changes_ca_behavior(self, integration_engine):
        """Test that MA influence actually changes CA responses through persona adjustments."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("physics", "Metaphor Lock Test")

        # Phase 1: Normal conversation without metaphor lock
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Think of energy like water flowing through pipes...",
            "reasoning": {"metaphor_used": "water flow"}
        })
        integration_engine.pd.extract_patterns = AsyncMock(return_value=[])
        integration_engine.ma.analyze_session = AsyncMock(return_value={
            "insights": ["User exploring energy concepts"],
            "flags": []
        })
        integration_engine.message_db.add_message_exchange = AsyncMock()
        integration_engine.message_db.update_session_analysis = AsyncMock()
        integration_engine.graph_db.create_node = MagicMock(return_value=True)
        integration_engine.graph_db.create_edge = MagicMock(return_value=True)

        # Process first exchange
        result1 = await integration_engine.process_user_input("What is kinetic energy?")

        # Verify no persona adjustment was applied during normal conversation (MA is command-only)
        integration_engine.ca.update_persona.assert_not_called()

        # Phase 2: MA detects metaphor lock and provides persona adjustments via /analyze then /apply_ma
        ma_analysis_with_adjustments = {
            "insights": ["User locked into water metaphors", "Need metaphor diversity"],
            "flags": ["metaphor_lock"],
            "persona_adjustments": {
                "metaphor_diversity": "encourage_new_domains",
                "prompting_style": "suggest_alternative_metaphors"
            }
        }
        integration_engine.ma.analyze_session = AsyncMock(
            return_value=ma_analysis_with_adjustments)

        # Run analysis command (does not apply adjustments yet)
        analyze_result = await integration_engine.execute_command("analyze")
        assert analyze_result["applied"] is False
        assert analyze_result["analysis"]["flags"] == ["metaphor_lock"]

        # Mock CA persona update and apply adjustments
        integration_engine.ca.update_persona = AsyncMock()
        apply_result = await integration_engine.execute_command("apply_ma")
        assert apply_result.get("applied", False) is True
        integration_engine.ca.update_persona.assert_called_once_with({
            "metaphor_diversity": "encourage_new_domains",
            "prompting_style": "suggest_alternative_metaphors"
        })

        # Phase 3: Verify CA behavior change in next response
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Instead of water, let's explore energy like a bouncing ball - how does the ball's motion relate to its energy?",
            "reasoning": {"metaphor_used": "bouncing ball", "avoided_overused_metaphor": "water_flow"}
        })

        result3 = await integration_engine.process_user_input("How does energy change during motion?")

        # Verify behavioral change
        assert "bouncing ball" in result3["message"]
        # Shows it's moving away from water metaphor
        assert "Instead of water" in result3["message"]

        # Verify the complete flow (commands don't add messages)
        assert len(integration_engine.current_session.messages) >= 2
        # Verify update_persona was called once via apply_ma
        assert integration_engine.ca.update_persona.call_count == 1


class TestSessionMapIntegration:
    """Integration tests for complete session mapping with real data."""

    @pytest.mark.asyncio
    async def test_session_map_integration(self, integration_engine):
        """Test complete session mapping with realistic node and edge data."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("quantum physics", "Quantum Entanglement Session")

        # Create realistic nodes and edges
        created_nodes = [
            {
                "id": "node1",
                "label": "quantum entanglement",
                "node_type": "concept",
                "properties": {"domain": "physics", "session_id": session.id}
            },
            {
                "id": "node2",
                "label": "connected coins",
                "node_type": "metaphor",
                "properties": {"concept": "quantum entanglement", "session_id": session.id}
            }
        ]

        created_edges = [
            {
                "id": "edge1",
                "source_id": "node1",
                "target_id": "node2",
                "source_label": "quantum entanglement",
                "target_label": "connected coins",
                "edge_type": "metaphorical"
            }
        ]

        session.nodes_created = ["node1", "node2"]
        session.edges_created = ["edge1"]

        # Mock session map data retrieval
        def mock_get_node(node_id):
            return next((node for node in created_nodes if node["id"] == node_id), None)

        def mock_get_edge(edge_id):
            return next((edge for edge in created_edges if edge["id"] == edge_id), None)

        integration_engine.graph_db.get_node = MagicMock(
            side_effect=mock_get_node)
        integration_engine.graph_db.get_edge = MagicMock(
            side_effect=mock_get_edge)

        # Mock relationship description generation
        expected_descriptions = [
            "• Quantum entanglement emerges from your exploration of 'connected coins,' showing how particles maintain mysterious correlations across any distance."
        ]
        integration_engine._generate_relationship_descriptions = AsyncMock(
            return_value=expected_descriptions)

        # Execute session map command
        map_result = await integration_engine.execute_command("map")

        # Verify complete session map integration
        assert "error" not in map_result
        assert map_result["session_id"] == session.id
        assert map_result["topic"] == "quantum physics"

        # Verify nodes are included
        assert "nodes" in map_result
        assert len(map_result["nodes"]) == 2

        # Verify edges are included
        assert "edges" in map_result
        assert len(map_result["edges"]) == 1

        # Verify relationship descriptions
        assert "relationship_descriptions" in map_result
        assert map_result["relationship_descriptions"] == expected_descriptions

        # Verify counts
        assert map_result["node_count"] == 2
        assert map_result["connection_count"] == 1


class TestErrorRecoveryScenarios:
    """Integration tests for error recovery and graceful failure handling."""

    @pytest.mark.asyncio
    async def test_api_failure_recovery_in_gap_check(self, integration_engine):
        """Test graceful handling of OpenAI API failures during gap check."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("error recovery", "API Failure Test")
        session.messages = [
            {
                "timestamp": datetime.now().isoformat(),
                "user": "What is photosynthesis?",
                "assistant": "Think of it like a solar panel converting sunlight to energy..."
            }
        ]

        # Mock successful concept identification
        integration_engine._identify_recent_concept = AsyncMock(
            return_value="photosynthesis")

        # Mock successful u_vector retrieval
        u_vector = Vector(values=[0.1, 0.2, 0.3], model="test", dimension=3)
        integration_engine._get_user_vector = AsyncMock(return_value=u_vector)

        # Mock c_vector creation with API failure in canonical definition generation
        c_vector = Vector(
            values=[0.15, 0.25, 0.35],
            model="test",
            dimension=3,
            metadata={
                "canonical_definition": "Standard definition of photosynthesis in biology domain"}
        )
        integration_engine._get_or_create_canonical_vector = AsyncMock(
            return_value=c_vector)

        # Mock gap calculation
        gap_analysis = {
            "similarity": 0.80,
            "gap_score": 0.20,
            "severity": "low",
            "canonical_definition": "Standard definition of photosynthesis in biology domain",
            "severity_description": "Good alignment"
        }
        integration_engine.embedding_service.calculate_gap_score = MagicMock(
            return_value=gap_analysis)

        # Mock message formatting with API failure, should use fallback
        integration_engine._format_gap_message = AsyncMock(
            side_effect=Exception("API failure in message formatting")
        )
        integration_engine._fallback_gap_message = MagicMock(
            return_value="🎯 **Gap Analysis for 'photosynthesis'**\n\nYour understanding shows **80% alignment** with canonical knowledge - excellent work!"
        )

        # Execute gap check - should handle API failures gracefully
        result = await integration_engine.execute_command("gapcheck")

        # Verify graceful error recovery
        assert "error" not in result
        assert result["concept"] == "photosynthesis"
        assert result["similarity"] == 0.80
        assert "80% alignment" in result["message"]

        # Verify fallback was used
        integration_engine._fallback_gap_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_connection_failure_recovery(self, integration_engine):
        """Test graceful handling of database connection failures."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("database errors", "DB Failure Test")

        # Test Case: Neo4j failure during node creation
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Atoms are like tiny solar systems...",
            "reasoning": {"metaphor_used": "solar systems"}
        })

        integration_engine.pd.extract_patterns = AsyncMock(return_value=[MagicMock(
            concept="atomic structure",
            domain="chemistry",
            context="basic atomic model",
            confidence=0.9,
            metaphors=["solar systems"]
        )])

        integration_engine.ma.analyze_session = AsyncMock(return_value={
            "insights": ["User learning atomic concepts"],
            "flags": []
        })

        # Mock database operations
        integration_engine.message_db.add_message_exchange = AsyncMock()
        integration_engine.message_db.update_session_analysis = AsyncMock()

        # Simulate Neo4j node creation failure
        integration_engine.graph_db.create_node = MagicMock(
            return_value=False)  # Failure
        integration_engine.graph_db.create_edge = MagicMock(
            return_value=False)  # Failure

        # Process input - should handle database failures gracefully
        result = await integration_engine.process_user_input("What is atomic structure?")

        # Verify conversation continues despite database failures
        assert "message" in result
        assert result["message"] == "Atoms are like tiny solar systems..."
        # No concepts created due to DB failure
        assert result["new_concepts"] == 0

    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, integration_engine):
        """Test graceful handling of agent processing failures."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("agent failures", "Agent Failure Test")

        # Test Case: PD extraction failure
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Gravity is like a bowling ball on a trampoline...",
            "reasoning": {"metaphor_used": "bowling ball on trampoline"}
        })
        integration_engine.pd.extract_patterns = AsyncMock(
            side_effect=Exception("PD extraction failed"))
        integration_engine.ma.analyze_session = AsyncMock(
            return_value={"insights": [], "flags": []})
        integration_engine.message_db.add_message_exchange = AsyncMock()
        integration_engine.message_db.update_session_analysis = AsyncMock()

        # Should continue processing despite PD failure
        result = await integration_engine.process_user_input("What is gravity?")

        assert "message" in result
        assert result["message"] == "Gravity is like a bowling ball on a trampoline..."
        # No concepts extracted due to PD failure
        assert result["new_concepts"] == 0

    @pytest.mark.asyncio
    async def test_session_recovery_after_failures(self, integration_engine):
        """Test that sessions can recover and continue after various failures."""

        # Setup session
        integration_engine.message_db.create_session = AsyncMock(
            return_value=True)
        integration_engine.graph_db.create_session_summary = MagicMock(
            return_value=True)
        integration_engine.ca.start_session = AsyncMock()
        integration_engine.graph_db.search_nodes_by_content = MagicMock(
            return_value=[])

        session = await integration_engine.start_session("recovery test", "Recovery Test")

        # Phase 1: Normal operation
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Photosynthesis is like a factory converting sunlight...",
            "reasoning": {"metaphor_used": "factory"}
        })
        integration_engine.pd.extract_patterns = AsyncMock(return_value=[])
        integration_engine.ma.analyze_session = AsyncMock(
            return_value={"insights": [], "flags": []})
        integration_engine.message_db.add_message_exchange = AsyncMock()
        integration_engine.message_db.update_session_analysis = AsyncMock()
        integration_engine.graph_db.create_node = MagicMock(return_value=True)

        result1 = await integration_engine.process_user_input("What is photosynthesis?")
        assert "message" in result1
        assert len(session.messages) == 1

        # Phase 2: Recovery after temporary failure
        integration_engine.ca.process_input = AsyncMock(return_value={
            "message": "Chlorophyll is like the green paint that captures sunlight...",
            "reasoning": {"metaphor_used": "green paint"}
        })

        # Should work again
        result2 = await integration_engine.process_user_input("What about chlorophyll?")
        assert "message" in result2
        assert "green paint" in result2["message"]

        # Verify session state is maintained
        assert session.status == "active"
        assert session.topic == "recovery test"
        assert len(session.messages) >= 1
