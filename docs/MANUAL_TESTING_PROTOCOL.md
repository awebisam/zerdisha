# PEEngine Core Solidification - Manual Testing Protocol

## Overview

This document provides comprehensive manual testing procedures for validating the three core features implemented in the PEEngine core solidification:

1. **Gap Check Functionality** (`/gapcheck` command)
2. **Metacognitive Influence** (MA → CA persona adjustments)
3. **Enhanced Session Mapping** (`/map` command with relationships)

## Prerequisites

### Environment Setup
```bash
# Ensure environment is configured
cp .env.example .env
# Edit .env with proper Azure OpenAI credentials

# Start databases
docker-compose up -d neo4j mongodb

# Install PEEngine in development mode
pip install -e .

# Verify installation
peengine --help
```

### Database Health Check
```bash
# Check Neo4j is running
curl http://localhost:7474

# Check MongoDB is running
docker-compose ps
```

## Core Feature Testing Protocol

### Test Suite 1: Gap Check Functionality

#### Test 1.1: Basic Gap Check Flow
**Objective**: Verify complete gap check functionality from conversation to analysis

**Steps**:
1. Start new session:
   ```bash
   peengine start "quantum mechanics exploration"
   ```

2. Engage in focused conversation about a specific concept (4-5 exchanges):
   ```
   User: "I'm curious about quantum entanglement"
   [Continue conversation about entanglement for several turns]
   ```

3. Execute gap check:
   ```
   /gapcheck
   ```

**Expected Results**:
- System identifies "quantum entanglement" as recent concept
- Retrieves or creates canonical vector (c_vector)
- Calculates similarity score between user understanding and canonical knowledge
- Returns meaningful analysis message with:
  - Concept name
  - Similarity score (0.0-1.0)
  - Gap severity (low/medium/high)
  - Personalized explanation of gaps or alignment

**Success Criteria**:
- [ ] Command executes without errors
- [ ] Returns structured gap analysis (not placeholder text)
- [ ] Similarity score is reasonable (0.0-1.0 range)
- [ ] Message is personalized and meaningful
- [ ] C-vector is created and stored in Neo4j if missing

#### Test 1.2: Gap Check Error Scenarios

**Test 1.2a: Empty Session Gap Check**
```bash
peengine start "empty session test"
/gapcheck
```
**Expected**: Clear error message about no conversation to analyze

**Test 1.2b: Ambiguous Concept Gap Check**
```bash
peengine start "vague conversation test"
# Have very general conversation without clear concepts
/gapcheck
```
**Expected**: Error message about no clear concept identified

**Test 1.2c: Network Failure Simulation**
- Temporarily disable internet connection
- Execute `/gapcheck` on session with clear concept
**Expected**: Graceful error handling with meaningful message

### Test Suite 2: Metacognitive Influence

#### Test 2.1: Metaphor Lock Detection and Persona Adjustment
**Objective**: Verify MA detects metaphor overuse and triggers CA persona changes

**Steps**:
1. Start session focused on a complex topic:
   ```bash
   peengine start "understanding neural networks"
   ```

2. Deliberately use the same metaphor repeatedly (6-8 exchanges):
   ```
   User: "Neural networks are like a brain with neurons"
   [Continue using brain/neuron metaphors exclusively]
   ```

3. Monitor CA responses for behavioral changes:
   - Initial responses should accommodate brain metaphors
   - After several exchanges, CA should start prompting for new metaphors
   - Look for phrases like "What other metaphors might help?" or similar

**Expected Results**:
- MA detects metaphor lock-in pattern
- Persona adjustments are applied to CA
- CA behavior changes to encourage metaphor diversity
- System logs show persona adjustment application

**Success Criteria**:
- [ ] MA analysis includes persona_adjustments after repeated metaphor use
- [ ] CA responses change to prompt for new metaphors
- [ ] Persona adjustments are logged in system output
- [ ] Behavioral change is noticeable in conversation flow

#### Test 2.2: Persona Adjustment Persistence
**Objective**: Verify persona adjustments persist for subsequent turns

**Steps**:
1. Continue from Test 2.1 after persona adjustment triggered
2. Engage in 3-4 more exchanges
3. Observe if CA maintains adjusted behavior

**Expected Results**:
- Persona adjustments remain active for subsequent conversation turns
- CA continues encouraging metaphor diversity
- Adjustments don't revert immediately

### Test Suite 3: Enhanced Session Mapping

#### Test 3.1: Session Map with Relationships
**Objective**: Verify session map displays both concepts and their connections

**Steps**:
1. Start session and build rich concept network:
   ```bash
   peengine start "exploring thermodynamics"
   ```

2. Discuss multiple related concepts to create connections:
   ```
   User: "How does entropy relate to energy?"
   [Continue building connections between entropy, energy, heat, temperature, etc.]
   ```

3. Execute session map:
   ```
   /map
   ```

**Expected Results**:
- Table showing all concepts (nodes) created in session
- List of relationships between concepts
- Relationship format: "[Concept A] --(Relationship Type)--> [Concept B]"
- Connection count summary

**Success Criteria**:
- [ ] All session concepts are displayed in table format
- [ ] Relationships are shown in human-readable format
- [ ] Both nodes and edges are included in output
- [ ] Connection count matches actual relationships created
- [ ] No database errors or missing data warnings

#### Test 3.2: Large Session Map Performance
**Objective**: Test session map performance with many concepts

**Steps**:
1. Create session with 15+ concepts through extended conversation
2. Execute `/map` command
3. Measure response time and output quality

**Expected Results**:
- Command completes within reasonable time (< 5 seconds)
- All concepts and relationships are displayed
- Output remains readable and well-formatted

## Error Condition Testing

### Network Failure Scenarios

#### Test E1: OpenAI API Unavailable
**Setup**: Block access to OpenAI endpoints
**Commands to Test**:
- `/gapcheck` (canonical definition generation)
- Normal conversation (CA responses)
- Session analysis (MA processing)

**Expected Behavior**:
- Graceful error messages (not stack traces)
- System continues functioning where possible
- Clear indication of what functionality is impacted

#### Test E2: Database Connection Issues

**Test E2a: Neo4j Unavailable**
```bash
docker-compose stop neo4j
peengine start "database test"
```
**Expected**: Clear error about graph database connection

**Test E2b: MongoDB Unavailable**
```bash
docker-compose stop mongodb
peengine start "session test"
```
**Expected**: Clear error about session storage

### Data Consistency Testing

#### Test D1: Corrupted Session Data
**Objective**: Verify graceful handling of inconsistent data

**Steps**:
1. Create normal session with concepts and relationships
2. Manually corrupt some data in databases
3. Execute `/map` and `/gapcheck` commands

**Expected Results**:
- Commands handle missing nodes/edges gracefully
- Clear warnings about data inconsistencies
- Partial results displayed where possible

## Performance Benchmarks

### Gap Check Performance

#### Benchmark G1: Gap Check Response Time
**Test Conditions**:
- Session with 10 concepts
- Clear target concept for analysis
- Existing c_vector vs. new c_vector generation

**Measurements**:
- Time from `/gapcheck` command to response
- Target: < 3 seconds with existing c_vector
- Target: < 10 seconds with new c_vector generation

**Test Script**:
```bash
# Time the gap check command
time echo "/gapcheck" | peengine start "performance test"
```

#### Benchmark G2: Large Conversation Gap Check
**Test Conditions**:
- Session with 50+ message exchanges
- Multiple concepts discussed
- Gap check on recent concept

**Measurements**:
- Response time should remain reasonable (< 5 seconds)
- Memory usage should not spike excessively

### Session Map Performance

#### Benchmark M1: Large Session Map
**Test Conditions**:
- Session with 25+ concepts
- 40+ relationships between concepts
- Complex interconnected graph

**Measurements**:
- Map generation time: Target < 3 seconds
- Output formatting time: Target < 1 second
- Memory usage during rendering

**Test Script**:
```bash
# Create large session programmatically if needed
# Then measure map performance
time echo "/map" | peengine start "large session test"
```

#### Benchmark M2: Relationship Rendering Performance
**Test Conditions**:
- Session with 100+ relationships
- Complex multi-domain connections

**Measurements**:
- Relationship formatting time
- Console output rendering time
- Memory usage for large relationship lists

## Validation Checklist

### Pre-Test Validation
- [ ] Environment variables properly configured
- [ ] Docker containers running (Neo4j, MongoDB)
- [ ] PEEngine installed and accessible via CLI
- [ ] Database connections healthy
- [ ] Azure OpenAI API accessible

### Core Feature Validation

#### Gap Check Feature
- [ ] `/gapcheck` executes without errors on active session
- [ ] Returns meaningful analysis (not placeholder text)
- [ ] Handles empty sessions gracefully
- [ ] Creates c_vectors when missing
- [ ] Calculates reasonable similarity scores (0.0-1.0)
- [ ] Provides personalized gap analysis messages
- [ ] Handles API failures gracefully
- [ ] Performance meets benchmarks (< 10 seconds)

#### Metacognitive Influence Feature
- [ ] MA detects metaphor lock-in patterns
- [ ] Persona adjustments are generated and applied
- [ ] CA behavior changes after adjustments
- [ ] Adjustments persist across conversation turns
- [ ] System logs persona adjustment activity
- [ ] No errors during persona updates
- [ ] Graceful handling of invalid adjustments

#### Session Map Feature
- [ ] `/map` displays all session concepts
- [ ] Relationships are shown in readable format
- [ ] Both nodes and edges included in output
- [ ] Connection count is accurate
- [ ] Handles sessions with no relationships
- [ ] Performance acceptable for large sessions (< 5 seconds)
- [ ] Graceful handling of missing data
- [ ] Proper formatting for complex graphs

### Error Handling Validation
- [ ] Network failures produce clear error messages
- [ ] Database connection issues handled gracefully
- [ ] Corrupted data doesn't crash the system
- [ ] API rate limits handled appropriately
- [ ] Invalid user inputs handled safely
- [ ] System remains stable during error conditions

### Performance Validation
- [ ] Gap check completes within time targets
- [ ] Session map renders efficiently
- [ ] Memory usage remains reasonable
- [ ] No memory leaks during extended sessions
- [ ] Concurrent operations handled properly
- [ ] Large dataset operations scale appropriately

## Test Execution Log Template

### Session Information
- **Date**: ___________
- **Tester**: ___________
- **Environment**: ___________
- **PEEngine Version**: ___________

### Test Results Summary
- **Total Tests Executed**: ___/___
- **Tests Passed**: ___
- **Tests Failed**: ___
- **Critical Issues Found**: ___

### Detailed Results

#### Gap Check Tests
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 1.1 | Basic Gap Check Flow | ⬜ Pass ⬜ Fail | |
| 1.2a | Empty Session Error | ⬜ Pass ⬜ Fail | |
| 1.2b | Ambiguous Concept Error | ⬜ Pass ⬜ Fail | |
| 1.2c | Network Failure Handling | ⬜ Pass ⬜ Fail | |

#### Metacognitive Influence Tests
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 2.1 | Metaphor Lock Detection | ⬜ Pass ⬜ Fail | |
| 2.2 | Persona Adjustment Persistence | ⬜ Pass ⬜ Fail | |

#### Session Map Tests
| Test ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| 3.1 | Map with Relationships | ⬜ Pass ⬜ Fail | |
| 3.2 | Large Session Performance | ⬜ Pass ⬜ Fail | |

#### Performance Benchmarks
| Benchmark | Target | Actual | Status |
|-----------|--------|--------|--------|
| Gap Check Response Time | < 3s | ___s | ⬜ Pass ⬜ Fail |
| New C-Vector Generation | < 10s | ___s | ⬜ Pass ⬜ Fail |
| Session Map Generation | < 3s | ___s | ⬜ Pass ⬜ Fail |
| Large Session Map | < 5s | ___s | ⬜ Pass ⬜ Fail |

### Issues and Observations
_Record any bugs, unexpected behavior, or areas for improvement_

### Recommendations
_Suggest any changes or improvements based on testing results_

---

## Quick Test Commands

For rapid validation during development:

```bash
# Quick smoke test
peengine start "quick test" && echo "Hello quantum mechanics" && echo "/gapcheck" && echo "/map" && echo "/end"

# Performance test
time peengine start "performance test"

# Error handling test
peengine start "error test" # with databases stopped
```

## Automated Test Execution

While this is primarily a manual testing protocol, key tests can be automated:

```bash
# Run all unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=peengine --cov-report=html
```

This manual testing protocol ensures comprehensive validation of all core features while providing clear success criteria and performance benchmarks for the PEEngine core solidification.