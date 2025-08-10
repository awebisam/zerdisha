"""Tests for CLI display functions."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from io import StringIO

from peengine.cli import (
    display_session_map,
    display_gap_check,
    display_seed,
    display_session_summary,
    generate_adaptive_error_message,
    generate_adaptive_conversation_error
)


@pytest.fixture
def mock_console():
    """Mock Rich console for testing output."""
    with patch('peengine.cli.console') as mock_console:
        yield mock_console


@pytest.fixture
def sample_session_map_data():
    """Sample session map data with nodes and edges."""
    return {
        "topic": "Quantum Mechanics",
        "nodes": [
            {
                "id": "node1",
                "label": "quantum superposition",
                "node_type": "concept",
                "properties": {"domain": "physics"}
            },
            {
                "id": "node2", 
                "label": "wave function",
                "node_type": "concept",
                "properties": {"domain": "physics"}
            },
            {
                "id": "node3",
                "label": "measurement",
                "node_type": "process",
                "properties": {"domain": "physics"}
            }
        ],
        "edges": [
            {
                "id": "edge1",
                "source_id": "node1",
                "target_id": "node2", 
                "source_label": "quantum superposition",
                "target_label": "wave function",
                "edge_type": "describes"
            },
            {
                "id": "edge2",
                "source_id": "node2",
                "target_id": "node3",
                "source_label": "wave function", 
                "target_label": "measurement",
                "edge_type": "collapses_during"
            }
        ],
        "node_count": 3,
        "connection_count": 2
    }


@pytest.fixture
def sample_session_map_with_descriptions():
    """Sample session map data with LLM-generated relationship descriptions."""
    return {
        "topic": "Quantum Mechanics",
        "nodes": [
            {
                "id": "node1",
                "label": "quantum superposition",
                "node_type": "concept",
                "properties": {"domain": "physics"}
            },
            {
                "id": "node2",
                "label": "wave function", 
                "node_type": "concept",
                "properties": {"domain": "physics"}
            }
        ],
        "edges": [
            {
                "id": "edge1",
                "source_id": "node1",
                "target_id": "node2",
                "source_label": "quantum superposition",
                "target_label": "wave function", 
                "edge_type": "describes"
            }
        ],
        "relationship_descriptions": [
            "Quantum superposition emerges from the mathematical structure of the wave function",
            "The wave function provides the probabilistic framework for understanding superposition states"
        ],
        "node_count": 2,
        "connection_count": 1
    }


@pytest.fixture
def sample_gap_check_data():
    """Sample gap check analysis data."""
    return {
        "concept": "quantum tunneling",
        "similarity": 0.85,
        "gap_score": 0.15,
        "severity": "low",
        "message": "🎯 **Understanding Check: Quantum Tunneling**\n\nYour grasp of quantum tunneling shows strong intuition! Your metaphors about particles 'sneaking through walls' capture the essence beautifully.\n\n**Alignment: 85%** - Excellent work!\n\nYour understanding emphasizes the probabilistic nature and barrier penetration, which aligns well with the canonical physics perspective. Keep exploring these quantum mysteries!"
    }


class TestDisplaySessionMap:
    """Test cases for display_session_map function."""

    def test_display_session_map_with_relationships(self, mock_console, sample_session_map_data):
        """Test that session map displays both nodes and edges correctly."""
        # Act
        display_session_map(sample_session_map_data)

        # Assert
        # Verify console.print was called multiple times
        assert mock_console.print.call_count >= 3  # Table + relationships + summary

        # Get all print calls
        print_calls = mock_console.print.call_args_list

        # Check that nodes table was printed (first call should be the table)
        table_call = print_calls[0]
        table_obj = table_call[0][0]
        # Check that it's a Table object with the correct title
        assert hasattr(table_obj, 'title')
        assert "Quantum Mechanics" in str(table_obj.title)

        # Check that relationships section was printed
        relationship_calls = [call for call in print_calls if "Conceptual Connections" in str(call)]
        assert len(relationship_calls) > 0

        # Check that summary statistics were printed
        summary_calls = [call for call in print_calls if "Summary: 3 concepts, 2 connections" in str(call)]
        assert len(summary_calls) > 0

    def test_display_session_map_with_llm_descriptions(self, mock_console, sample_session_map_with_descriptions):
        """Test session map display with LLM-generated relationship descriptions."""
        # Act
        display_session_map(sample_session_map_with_descriptions)

        # Assert
        print_calls = mock_console.print.call_args_list

        # Check that LLM descriptions were used instead of simple table format
        description_calls = [call for call in print_calls 
                           if "Quantum superposition emerges from" in str(call)]
        assert len(description_calls) > 0

        # Verify summary shows correct counts
        summary_calls = [call for call in print_calls if "Summary: 2 concepts, 1 connections" in str(call)]
        assert len(summary_calls) > 0

    def test_display_session_map_no_relationships(self, mock_console):
        """Test session map display when no relationships exist."""
        # Arrange
        map_data = {
            "topic": "Solo Concept",
            "nodes": [
                {
                    "id": "node1",
                    "label": "isolated concept",
                    "node_type": "concept", 
                    "properties": {"domain": "general"}
                }
            ],
            "edges": [],
            "node_count": 1,
            "connection_count": 0
        }

        # Act
        display_session_map(map_data)

        # Assert
        print_calls = mock_console.print.call_args_list

        # Should still print table and summary, but no relationships section
        assert mock_console.print.call_count >= 2

        # Check no "Conceptual Connections" section
        relationship_calls = [call for call in print_calls if "Conceptual Connections" in str(call)]
        assert len(relationship_calls) == 0

        # Check summary shows 0 connections
        summary_calls = [call for call in print_calls if "Summary: 1 concepts, 0 connections" in str(call)]
        assert len(summary_calls) > 0

    def test_display_session_map_error_handling(self, mock_console):
        """Test session map display handles error responses gracefully."""
        # Arrange
        error_data = {"error": "No active session"}

        # Act
        display_session_map(error_data)

        # Assert
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "No active session" in str(call_args)
        assert "[red]" in str(call_args)  # Error should be in red

    def test_display_session_map_missing_node_data(self, mock_console):
        """Test session map handles missing node data gracefully."""
        # Arrange
        incomplete_data = {
            "topic": "Incomplete Session",
            "nodes": [
                {
                    "id": "node1",
                    # Missing label, node_type, properties
                }
            ],
            "edges": [],
            "node_count": 1,
            "connection_count": 0
        }

        # Act
        display_session_map(incomplete_data)

        # Assert - should not crash, should use defaults
        assert mock_console.print.call_count >= 2

        # Check that defaults were used (Unknown, unknown, general)
        print_calls = mock_console.print.call_args_list
        table_call = str(print_calls[0])
        # The exact assertion depends on Rich's table rendering, but it should not crash

    def test_display_session_map_relationship_formatting(self, mock_console, sample_session_map_data):
        """Test proper formatting of relationships in fallback mode."""
        # Act
        display_session_map(sample_session_map_data)

        # Assert
        print_calls = mock_console.print.call_args_list

        # Should have at least 3 calls: nodes table + relationships header + relationships table + summary
        assert len(print_calls) >= 3
        
        # Check that relationships section header was printed
        relationship_header_found = False
        for call in print_calls:
            call_str = str(call)
            if "Conceptual Connections" in call_str:
                relationship_header_found = True
                break
        assert relationship_header_found
        
        # Check that a relationships table was created (look for Table objects after the first one)
        table_objects = [call for call in print_calls if hasattr(call[0][0], 'add_column')]
        assert len(table_objects) >= 2  # At least nodes table and relationships table


class TestDisplayGapCheck:
    """Test cases for display_gap_check function."""

    def test_display_gap_check_results(self, mock_console, sample_gap_check_data):
        """Test gap analysis message display."""
        # Act
        display_gap_check(sample_gap_check_data)

        # Assert
        mock_console.print.assert_called_once()
        
        # Check that Panel was created with correct content
        call_args = mock_console.print.call_args[0][0]
        
        # Should be a Panel object with the gap check message
        assert hasattr(call_args, 'renderable')  # Panel has renderable attribute
        
        # The message should be wrapped in Markdown - check the original message
        markdown_obj = call_args.renderable
        assert hasattr(markdown_obj, 'markup')
        # The markup should contain our message content
        markup_content = str(markdown_obj.markup)
        assert "quantum tunneling" in markup_content
        assert "85%" in markup_content
        assert "Excellent work" in markup_content

    def test_display_gap_check_no_message(self, mock_console):
        """Test gap check display when no message is provided."""
        # Arrange
        empty_data = {}

        # Act
        display_gap_check(empty_data)

        # Assert
        mock_console.print.assert_called_once()
        
        call_args = mock_console.print.call_args[0][0]
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "Gap check analysis not available" in markup_content

    def test_display_gap_check_error_response(self, mock_console):
        """Test gap check display with error response."""
        # Arrange
        error_data = {"error": "No recent concept identified"}

        # Act
        display_gap_check(error_data)

        # Assert
        mock_console.print.assert_called_once()
        
        # Should still display in panel format, but with the error as message
        call_args = mock_console.print.call_args[0][0]
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "Gap check analysis not available" in markup_content


class TestSessionMapFormatting:
    """Test cases for session map table and relationship formatting."""

    def test_session_map_table_formatting(self, mock_console):
        """Test proper table formatting for session map nodes."""
        # Arrange
        map_data = {
            "topic": "Test Formatting",
            "nodes": [
                {
                    "id": "node1",
                    "label": "test concept",
                    "node_type": "concept",
                    "properties": {"domain": "testing"}
                },
                {
                    "id": "node2", 
                    "label": "another concept",
                    "node_type": "process",
                    "properties": {"domain": "validation"}
                }
            ],
            "edges": [],
            "node_count": 2,
            "connection_count": 0
        }

        # Act
        display_session_map(map_data)

        # Assert
        print_calls = mock_console.print.call_args_list
        
        # First call should be the table
        table_call = print_calls[0]
        table_obj = table_call[0][0]
        
        # Verify it's a Table object with correct title
        assert hasattr(table_obj, 'title')
        assert "Test Formatting" in str(table_obj.title)
        
        # Should have called print at least twice (table + summary)
        assert len(print_calls) >= 2

    def test_relationship_table_formatting(self, mock_console):
        """Test relationship table formatting in fallback mode."""
        # Arrange
        map_data = {
            "topic": "Relationship Test",
            "nodes": [
                {"id": "n1", "label": "concept A", "node_type": "concept", "properties": {"domain": "test"}},
                {"id": "n2", "label": "concept B", "node_type": "concept", "properties": {"domain": "test"}}
            ],
            "edges": [
                {
                    "id": "e1",
                    "source_id": "n1",
                    "target_id": "n2",
                    "source_label": "concept A",
                    "target_label": "concept B", 
                    "edge_type": "relates_to"
                }
            ],
            "node_count": 2,
            "connection_count": 1
        }

        # Act
        display_session_map(map_data)

        # Assert
        print_calls = mock_console.print.call_args_list
        
        # Should have multiple print calls including relationships
        assert len(print_calls) >= 3  # nodes table + relationships header + relationship table + summary
        
        # Check for relationships section
        relationship_header_found = False
        for call in print_calls:
            if "Conceptual Connections" in str(call):
                relationship_header_found = True
                break
        assert relationship_header_found

    def test_empty_session_map_formatting(self, mock_console):
        """Test formatting when session map is empty."""
        # Arrange
        empty_map = {
            "topic": "Empty Session",
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "connection_count": 0
        }

        # Act
        display_session_map(empty_map)

        # Assert
        print_calls = mock_console.print.call_args_list
        
        # Should still print table (empty) and summary
        assert len(print_calls) >= 2
        
        # Summary should show 0 counts
        summary_found = False
        for call in print_calls:
            if "Summary: 0 concepts, 0 connections" in str(call):
                summary_found = True
                break
        assert summary_found


class TestCLIErrorHandling:
    """Test cases for CLI error handling and graceful error message display."""

    @pytest.mark.asyncio
    async def test_adaptive_error_message_generation(self):
        """Test generation of adaptive error messages."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.ca.client.chat.completions.create = AsyncMock()
        mock_engine.ca._get_deployment_name = MagicMock(return_value="gpt-4")
        mock_engine.current_session = MagicMock()
        mock_engine.current_session.messages = [
            {"user": "What is quantum entanglement?", "assistant": "Think of it like..."}
        ]
        
        # Mock successful LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I understand you were trying to check your understanding gaps. It looks like there was a temporary issue with the analysis system.\n\nHere's how we can get back on track:\n• Try the /gapcheck command again in a moment\n• We can explore the concept through direct conversation\n• Your exploration of quantum mechanics is going well - let's keep the momentum going!"
        
        mock_engine.ca.client.chat.completions.create.return_value = mock_response

        error_context = {
            "command": "gapcheck",
            "error": "Vector similarity calculation failed",
            "session_active": True,
            "session_topic": "quantum mechanics"
        }

        # Act
        result = await generate_adaptive_error_message(mock_engine, error_context)

        # Assert
        assert "gapcheck" in result or "understanding gaps" in result
        assert "quantum mechanics" in result
        assert "get back on track" in result or "try" in result.lower()
        
        # Verify LLM was called
        mock_engine.ca.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_error_message_llm_failure(self):
        """Test adaptive error message fallback when LLM fails."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.ca.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_engine.ca._get_deployment_name = MagicMock(return_value="gpt-4")
        mock_engine.current_session = None

        error_context = {
            "command": "map",
            "error": "Database connection failed", 
            "session_active": False,
            "session_topic": None
        }

        # Act
        result = await generate_adaptive_error_message(mock_engine, error_context)

        # Assert - should use fallback message
        assert "map" in result.lower()
        assert "quick fixes" in result.lower() or "try" in result.lower()
        assert "don't worry" in result.lower() or "exploration" in result.lower()

    @pytest.mark.asyncio
    async def test_adaptive_conversation_error_generation(self):
        """Test generation of adaptive conversation error messages."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.ca.client.chat.completions.create = AsyncMock()
        mock_engine.ca._get_deployment_name = MagicMock(return_value="gpt-4")
        
        # Mock successful LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I can see you're exploring fascinating ideas about quantum physics! I encountered a brief processing hiccup, but your curiosity about wave-particle duality is exactly the kind of deep thinking that leads to breakthroughs.\n\nLet's keep the exploration flowing:\n• Try rephrasing your question about photons\n• We could approach this from the historical perspective\n• Your intuition about light behavior is valuable - let's build on it!\n\nWhat aspect of quantum behavior intrigues you most right now?"
        
        mock_engine.ca.client.chat.completions.create.return_value = mock_response

        error_context = {
            "error_type": "conversation_processing",
            "error": "Pattern extraction timeout",
            "user_input": "How do photons behave like waves and particles at the same time?",
            "session_active": True,
            "session_topic": "quantum physics"
        }

        # Act
        result = await generate_adaptive_conversation_error(mock_engine, error_context)

        # Assert
        assert "quantum physics" in result
        assert "exploration" in result.lower()
        assert "keep" in result.lower() or "continue" in result.lower()
        assert "?" in result  # Should end with a question to continue engagement
        
        # Verify LLM was called
        mock_engine.ca.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_conversation_error_llm_failure(self):
        """Test conversation error message fallback when LLM fails."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.ca.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        error_context = {
            "error_type": "conversation_processing",
            "error": "Network timeout",
            "user_input": "Tell me about black holes",
            "session_active": True,
            "session_topic": "astrophysics"
        }

        # Act
        result = await generate_adaptive_conversation_error(mock_engine, error_context)

        # Assert - should use fallback message
        assert "astrophysics" in result
        assert "technical hiccup" in result.lower()
        assert "exploration going" in result.lower() or "continue" in result.lower()
        assert "?" in result  # Should end with engaging question

    def test_display_error_messages_gracefully(self, mock_console):
        """Test that error messages are displayed gracefully in CLI functions."""
        # Test session map error display
        display_session_map({"error": "Database connection failed"})
        
        # Should print error in red
        mock_console.print.assert_called_with("[red]Database connection failed[/red]")
        
        # Reset mock
        mock_console.reset_mock()
        
        # Test gap check with missing data
        display_gap_check({"error": "No concept identified"})
        
        # Should still create panel with fallback message
        assert mock_console.print.called
        call_args = mock_console.print.call_args[0][0]
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "Gap check analysis not available" in markup_content

    def test_display_functions_handle_missing_keys(self, mock_console):
        """Test that display functions handle missing dictionary keys gracefully."""
        # Test session map with minimal data
        minimal_map = {"topic": "Test"}  # Missing nodes, edges, counts
        
        display_session_map(minimal_map)
        
        # Should not crash, should use empty defaults
        assert mock_console.print.called
        
        # Reset and test gap check with empty dict
        mock_console.reset_mock()
        
        display_gap_check({})  # Empty dict
        
        # Should display fallback message
        assert mock_console.print.called
        call_args = mock_console.print.call_args[0][0]
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "Gap check analysis not available" in markup_content

    def test_display_functions_handle_none_values(self, mock_console):
        """Test that display functions handle None values gracefully."""
        # Test with None values in data
        map_with_nones = {
            "topic": None,
            "nodes": [
                {
                    "id": "node1",
                    "label": None,
                    "node_type": None,
                    "properties": None
                }
            ],
            "edges": [],
            "node_count": None,
            "connection_count": None
        }
        
        display_session_map(map_with_nones)
        
        # Should not crash and should handle None values
        assert mock_console.print.called
        
        # Check that it used fallback values
        print_calls = mock_console.print.call_args_list
        # Should have printed something without crashing
        assert len(print_calls) > 0


class TestDisplayOtherFunctions:
    """Test cases for other display functions for completeness."""

    def test_display_seed(self, mock_console):
        """Test seed display function."""
        seed_data = {
            "seed_concept": "quantum decoherence",
            "rationale": "This concept bridges quantum and classical worlds",
            "suggested_questions": [
                "How does decoherence explain the quantum-to-classical transition?",
                "What role does the environment play in decoherence?"
            ]
        }
        
        display_seed(seed_data)
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        
        # Should be a Panel with seed information
        assert hasattr(call_args, 'renderable')
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "quantum decoherence" in markup_content
        assert "bridges quantum and classical" in markup_content

    def test_display_seed_error(self, mock_console):
        """Test seed display with error."""
        error_data = {"error": "No session active"}
        
        display_seed(error_data)
        
        mock_console.print.assert_called_once_with("[red]No session active[/red]")

    def test_display_session_summary(self, mock_console):
        """Test session summary display."""
        summary_data = {
            "duration_minutes": 25.5,
            "total_exchanges": 12,
            "concepts_created": 5,
            "connections_made": 8,
            "session_id": "test_session_123"
        }
        
        display_session_summary(summary_data)
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        
        # Should be a Panel with summary information
        assert hasattr(call_args, 'renderable')
        markdown_obj = call_args.renderable
        markup_content = str(markdown_obj.markup)
        assert "25.5 minutes" in markup_content
        assert "12" in markup_content  # exchanges
        assert "5" in markup_content   # concepts
        assert "8" in markup_content   # connections
        assert "test_session_123" in markup_content