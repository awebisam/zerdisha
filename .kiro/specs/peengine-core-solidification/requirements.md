# Requirements Document

## Introduction

The Personal Exploration Engine (PEEngine) currently has a solid foundational architecture with a three-agent system (Conversational Agent, Pattern Detector, Metacognitive Agent) and Neo4j knowledge graph storage. However, several critical features remain as placeholders or incomplete implementations. This feature focuses on transitioning PEEngine from a prototype to a functionally complete core system by implementing the missing `/gapcheck` command functionality, activating metacognitive influence on the conversational agent, and enhancing the `/map` command with relationship visualization.

## Requirements

### Requirement 1

**User Story:** As a PEEngine user, I want the `/gapcheck` command to provide meaningful analysis of my understanding gaps, so that I can identify areas where my knowledge differs from canonical understanding.

#### Acceptance Criteria

1. WHEN a user executes `/gapcheck` THEN the system SHALL identify the most recently discussed concept from the current session context
2. WHEN the system identifies a target concept THEN it SHALL retrieve the user's understanding vector (u_vector) from Neo4j
3. WHEN retrieving vectors THEN the system SHALL fetch the canonical knowledge vector (c_vector) for the concept
4. IF a c_vector does not exist THEN the system SHALL generate and save one using EmbeddingService.create_c_vector()
5. WHEN generating a c_vector THEN the system SHALL first create a canonical definition using EmbeddingService.generate_canonical_definition()
6. WHEN both vectors are available THEN the system SHALL calculate the gap score using EmbeddingService.calculate_gap_score()
7. WHEN the gap score is calculated THEN the system SHALL generate a user-facing message explaining the similarity and gap
8. WHEN similarity is high (>0.8) THEN the message SHALL indicate close alignment with canonical knowledge
9. WHEN similarity is low (<0.6) THEN the message SHALL highlight significant gaps and explain canonical emphasis areas

### Requirement 2

**User Story:** As a PEEngine user, I want the Metacognitive Agent to actively influence the Conversational Agent's behavior during sessions, so that the dialogue adapts dynamically based on my learning patterns.

#### Acceptance Criteria

1. WHEN the Metacognitive Agent provides analysis THEN the system SHALL check for persona_adjustments in the ma_analysis dictionary
2. IF persona_adjustments are present THEN the system SHALL immediately call ca.update_persona() with the adjustments
3. WHEN persona adjustments are applied THEN they SHALL take effect for the very next turn in the conversation
4. WHEN a user repeatedly uses the same metaphor THEN the MA SHALL eventually trigger a persona adjustment
5. WHEN persona adjustment occurs THEN the CA's responses SHALL subtly change to prompt for new metaphors
6. WHEN the CA persona is updated THEN the system prompt SHALL be modified accordingly

### Requirement 3

**User Story:** As a PEEngine user, I want the `/map` command to show both concepts and their relationships, so that I can visualize the knowledge graph structure of my exploration session.

#### Acceptance Criteria

1. WHEN a user executes `/map` THEN the system SHALL fetch all nodes created in the current session
2. WHEN fetching session data THEN the system SHALL also retrieve the edges that connect the nodes
3. WHEN returning session map data THEN the response dictionary SHALL include a list of edges
4. WHEN displaying the session map THEN connections SHALL be rendered in human-readable format
5. WHEN showing relationships THEN the format SHALL display "[Concept A] --(Relationship Type)--> [Concept B]"
6. WHEN multiple relationships exist THEN all connections SHALL be displayed in a clear, organized manner

### Requirement 4

**User Story:** As a developer working on PEEngine, I want comprehensive test coverage for the core functionality, so that I can ensure the system is reliable and maintainable.

#### Acceptance Criteria

1. WHEN testing the orchestrator THEN unit tests SHALL verify _gap_check functionality with mocked services
2. WHEN testing gap check THEN tests SHALL assert correct gap score calculation and c-vector creation when missing
3. WHEN testing metacognitive influence THEN tests SHALL confirm ca.update_persona is called when MA provides adjustments
4. WHEN testing CLI functionality THEN tests SHALL verify display_session_map correctly formats nodes and edges
5. WHEN running integration tests THEN gap check flow SHALL return meaningful, non-placeholder responses
6. WHEN testing MA influence flow THEN tests SHALL capture CA persona changes before and after trigger points
7. WHEN executing manual testing protocol THEN all commands SHALL function without errors
8. WHEN running full test suite THEN all unit and integration tests SHALL pass

### Requirement 5

**User Story:** As a PEEngine user, I want the system to be stable and error-free during normal operation, so that I can focus on learning without technical interruptions.

#### Acceptance Criteria

1. WHEN starting a new exploration session THEN the system SHALL initialize without errors
2. WHEN engaging in conversation THEN all three agents SHALL process inputs and provide responses
3. WHEN executing special commands THEN they SHALL complete successfully and provide expected output
4. WHEN the system encounters missing data THEN it SHALL gracefully handle the situation by generating required data
5. WHEN database operations fail THEN the system SHALL provide meaningful error messages
6. WHEN running the complete manual testing protocol THEN the application SHALL remain stable throughout