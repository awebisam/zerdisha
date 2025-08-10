# PEEngine Core Solidification - Testing Suite

## Overview

This testing suite provides comprehensive validation for the three core features implemented in the PEEngine core solidification:

1. **Gap Check Functionality** - `/gapcheck` command with u-vector/c-vector analysis
2. **Metacognitive Influence** - MA persona adjustments affecting CA behavior  
3. **Enhanced Session Mapping** - `/map` command with relationship visualization

## Testing Documentation Structure

```
docs/
├── MANUAL_TESTING_PROTOCOL.md    # Comprehensive manual testing procedures
├── VALIDATION_CHECKLIST.md       # Quick validation checklist
├── TEST_SCENARIOS.md             # Detailed error and edge case scenarios
└── TESTING_SUITE_README.md       # This overview document

tests/
└── performance_benchmarks.py     # Automated performance testing
```

## Quick Start Testing

### 1. Environment Validation
```bash
# Check prerequisites
docker-compose ps
peengine --help

# Verify database connectivity
curl http://localhost:7474  # Neo4j
curl http://localhost:8081  # MongoDB Express
```

### 2. Smoke Test (5 minutes)
```bash
# Quick validation of all three features
peengine start "smoke test"

# Build some concepts through conversation
echo "I want to understand quantum entanglement"
# [Continue conversation for 3-4 exchanges]

# Test gap check
echo "/gapcheck"

# Test session map
echo "/map"

# Test metacognitive influence (use same metaphor repeatedly)
# [Use "quantum particles are like dancers" metaphor 5+ times]
# [Observe CA behavior change]

echo "/end"
```

### 3. Performance Validation (10 minutes)
```bash
# Run automated benchmarks
python tests/performance_benchmarks.py

# Quick performance check
time peengine start "performance test"
```

## Testing Levels

### Level 1: Basic Functionality ✅
**Time Required**: 15 minutes  
**Purpose**: Verify all three features work without errors

**Tests**:
- [ ] Gap check executes and returns analysis
- [ ] Session map shows concepts and relationships
- [ ] Metacognitive influence changes CA behavior
- [ ] No crashes or critical errors

**Documentation**: `VALIDATION_CHECKLIST.md`

### Level 2: Comprehensive Validation ✅
**Time Required**: 45 minutes  
**Purpose**: Thorough testing of all features and error conditions

**Tests**:
- [ ] Complete user journey scenarios
- [ ] Error handling validation
- [ ] Performance benchmark compliance
- [ ] Integration between features

**Documentation**: `MANUAL_TESTING_PROTOCOL.md`

### Level 3: Stress and Edge Cases ✅
**Time Required**: 2+ hours  
**Purpose**: Validate system behavior under extreme conditions

**Tests**:
- [ ] Network failure scenarios
- [ ] Database connectivity issues
- [ ] Large dataset performance
- [ ] Extended runtime stability

**Documentation**: `TEST_SCENARIOS.md`

## Performance Benchmarks

### Target Performance Metrics

| Feature | Scenario | Target Time | Measurement |
|---------|----------|-------------|-------------|
| Gap Check | Existing c-vector | < 3 seconds | Command completion |
| Gap Check | New c-vector | < 10 seconds | Including generation |
| Gap Check | Large session | < 5 seconds | 20+ concepts |
| Session Map | Normal session | < 3 seconds | 5-10 concepts |
| Session Map | Large session | < 5 seconds | 20+ concepts |
| MA Analysis | Extended history | < 8 seconds | 50+ messages |

### Running Benchmarks

```bash
# All benchmarks
python tests/performance_benchmarks.py

# Specific feature
python tests/performance_benchmarks.py --feature gapcheck
python tests/performance_benchmarks.py --feature sessionmap
python tests/performance_benchmarks.py --feature metacognitive

# Custom run count
python tests/performance_benchmarks.py --runs 10
```

## Critical Success Criteria

### Must Pass (Release Blocking)
- [ ] All three core features execute without errors
- [ ] Gap check returns meaningful analysis (not placeholders)
- [ ] Metacognitive influence demonstrably changes CA behavior
- [ ] Session map displays both nodes and relationships
- [ ] Performance meets minimum benchmarks
- [ ] Error handling prevents system crashes

### Should Pass (Quality Gates)
- [ ] User experience is intuitive and smooth
- [ ] Error messages are clear and actionable
- [ ] Performance exceeds target benchmarks
- [ ] Features integrate seamlessly
- [ ] System remains stable during extended use

### Nice to Have (Future Improvements)
- [ ] Advanced error recovery mechanisms
- [ ] Performance optimizations for large datasets
- [ ] Enhanced user feedback and guidance
- [ ] Improved visualization of complex relationships

## Test Execution Workflow

### Pre-Release Testing
1. **Environment Setup** - Verify all prerequisites
2. **Smoke Testing** - Quick validation of basic functionality
3. **Performance Testing** - Run automated benchmarks
4. **Manual Testing** - Execute comprehensive test protocol
5. **Error Scenario Testing** - Validate error handling
6. **Documentation Review** - Ensure all tests are documented

### Continuous Testing
- Run smoke tests after each code change
- Execute performance benchmarks weekly
- Full manual testing before releases
- Update test scenarios as new edge cases are discovered

### Test Result Documentation

#### Test Execution Log Template
```
Date: ___________
Tester: ___________
Environment: ___________
PEEngine Version: ___________

Results Summary:
- Total Tests: ___/___
- Passed: ___
- Failed: ___
- Critical Issues: ___

Performance Results:
- Gap Check: ___s (target: <3s)
- Session Map: ___s (target: <3s)
- MA Analysis: ___s (target: <8s)

Critical Issues:
1. ________________________________
2. ________________________________

Recommendations:
________________________________
```

## Troubleshooting Common Issues

### Environment Issues
```bash
# Database not running
docker-compose up -d neo4j mongodb

# Missing environment variables
cp .env.example .env
# Edit .env with proper credentials

# PEEngine not installed
pip install -e .
```

### Performance Issues
```bash
# Check system resources
top
df -h

# Monitor database performance
docker stats

# Check network connectivity
ping api.openai.com
```

### Test Failures
```bash
# Check logs
tail -f peengine.log

# Verify database state
# Neo4j: http://localhost:7474
# MongoDB: http://localhost:8081

# Reset environment
docker-compose down
docker-compose up -d
```

## Integration with Development Workflow

### Pre-Commit Testing
```bash
# Quick validation before commits
python tests/performance_benchmarks.py --feature gapcheck --runs 1
```

### CI/CD Integration
```bash
# Automated testing pipeline
pytest tests/ -v
python tests/performance_benchmarks.py --runs 3
```

### Release Validation
```bash
# Complete test suite execution
./run_full_test_suite.sh
```

## Test Data Management

### Test Database Setup
- Use dedicated test databases
- Clean state before each test run
- Reproducible test data sets
- Isolation between test executions

### Test Data Cleanup
```bash
# Clean test databases
docker-compose down
docker volume prune
docker-compose up -d
```

## Reporting and Metrics

### Test Coverage Metrics
- Feature coverage: 100% of core features tested
- Error scenario coverage: All major error conditions
- Performance coverage: All critical performance paths

### Quality Metrics
- Test pass rate: Target 100% for critical tests
- Performance compliance: Target 100% within benchmarks
- Error handling coverage: All error paths tested

### Continuous Improvement
- Regular review of test scenarios
- Addition of new edge cases as discovered
- Performance benchmark updates
- Test automation improvements

This testing suite ensures comprehensive validation of the PEEngine core solidification while providing clear guidance for different testing needs and scenarios.