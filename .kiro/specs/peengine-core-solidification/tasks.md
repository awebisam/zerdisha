# Implementation Plan

- [ ] 1. Implement core gap check functionality in orchestrator
  - Create `_identify_recent_concept()` method to extract the most recently discussed concept from session messages
  - Implement `_get_user_vector()` method to retrieve u_vector for a concept from Neo4j
  - Create `_get_or_create_canonical_vector()` method that fetches existing c_vector or generates new one
  - Implement `_format_gap_message()` method to create user-friendly gap analysis message
  - Replace placeholder `_gap_check()` method with full implementation that orchestrates the gap analysis flow
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

- [ ] 2. Enhance embedding service with canonical vector generation
  - Modify `generate_canonical_definition()` method to use domain-specific prompting for better canonical definitions
  - Update `create_c_vector()` method to store canonical definition in vector metadata
  - Add `calculate_gap_score()` method that computes similarity and categorizes gap severity
  - Create helper method to save c_vectors to Neo4j with proper node structure
  - _Requirements: 1.4, 1.5, 1.6_

- [ ] 3. Activate metacognitive influence on conversational agent
  - Modify `process_user_input()` method in orchestrator to check for persona_adjustments in MA analysis
  - Add immediate call to `ca.update_persona()` when persona_adjustments are present
  - Enhance `update_persona()` method in ConversationalAgent to properly merge adjustments with current persona
  - Add logging to track when persona adjustments are applied and what changes were made
  - _Requirements: 2.1, 2.2, 2.3, 2.6_

- [ ] 4. Implement metaphor lock detection in metacognitive agent
  - Enhance session analysis template to better detect repeated metaphor usage patterns
  - Add metaphor tracking logic in `analyze_session()` method to identify when same metaphors are overused
  - Create persona adjustment logic that triggers when metaphor lock-in is detected
  - Implement adjustment recommendations that encourage metaphor diversity
  - _Requirements: 2.4, 2.5_

- [ ] 5. Enhance session map with relationship visualization
  - Add `get_edge()` method to Neo4jClient to retrieve edge data by ID
  - Modify `_show_session_map()` method in orchestrator to fetch and include edges in response
  - Update `display_session_map()` function in CLI to render relationships in human-readable format
  - Implement relationship formatting that shows "[Concept A] --(Relationship Type)--> [Concept B]" pattern
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 6. Create comprehensive unit tests for orchestrator functionality
  - Write `test_gap_check_with_existing_vectors()` to test gap analysis when both vectors exist
  - Create `test_gap_check_creates_missing_c_vector()` to verify c_vector generation when missing
  - Implement `test_persona_adjustment_applied()` to confirm MA adjustments are applied to CA
  - Add `test_identify_recent_concept()` to verify concept extraction from conversation
  - Write `test_format_gap_message()` to ensure proper user message formatting
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 7. Create unit tests for embedding service enhancements
  - Write `test_generate_canonical_definition()` to verify canonical definition generation
  - Create `test_create_c_vector_with_metadata()` to test c_vector creation with stored definition
  - Implement `test_calculate_gap_score()` to verify similarity calculation and severity categorization
  - Add `test_gap_score_edge_cases()` to handle zero vectors and dimension mismatches
  - _Requirements: 4.1, 4.2_

- [ ] 8. Create unit tests for CLI display functions
  - Write `test_display_session_map_with_relationships()` to verify relationship rendering
  - Create `test_display_gap_check_results()` to test gap analysis message display
  - Implement `test_session_map_formatting()` to verify proper table and relationship formatting
  - Add `test_cli_error_handling()` to ensure graceful error message display
  - _Requirements: 4.4_

- [ ] 9. Create integration tests for complete workflows
  - Write `test_complete_gap_check_flow()` that simulates full user journey from conversation to gap analysis
  - Create `test_ma_influence_changes_ca_behavior()` to verify end-to-end metacognitive influence
  - Implement `test_session_map_integration()` to test complete session mapping with real data
  - Add `test_error_recovery_scenarios()` to verify graceful handling of API failures and database issues
  - _Requirements: 4.5, 4.6, 4.7_

- [ ] 10. Implement error handling and edge cases
  - Add comprehensive error handling to `_gap_check()` for missing sessions, concepts, and API failures
  - Implement graceful degradation in `_get_or_create_canonical_vector()` when OpenAI API is unavailable
  - Add validation to persona adjustment application to prevent invalid adjustments from corrupting CA
  - Create fallback mechanisms for session map display when nodes or edges are missing
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 11. Add performance optimizations and caching
  - Implement c_vector caching in embedding service to avoid redundant canonical definition generation
  - Add conversation history limiting in gap check to analyze only recent exchanges for performance
  - Optimize session map queries to fetch nodes and edges efficiently in single database operations
  - Add timeout handling for OpenAI API calls to prevent hanging operations
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 12. Create manual testing documentation and validation
  - Document complete manual testing protocol with step-by-step user journey
  - Create test scenarios for error conditions like network failures and empty sessions
  - Implement validation checklist for verifying all three core features work correctly
  - Add performance benchmarks for gap check and session map operations with large datasets
  - _Requirements: 4.8, 5.6_