#!/bin/bash

# Personal Exploration Engine - Docker Setup Script

echo "🧠 Personal Exploration Engine - Docker Setup"
echo "=============================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        echo "❌ Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    echo "✅ Docker Compose found: $COMPOSE_CMD"
}

# Function to create .env if it doesn't exist
setup_env() {
    if [ ! -f .env ]; then
        echo "📝 Creating .env file from example..."
        cp .env.example .env
        echo "⚠️  Please edit .env with your Azure OpenAI credentials"
        echo "   Required: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"
    else
        echo "✅ .env file exists"
    fi
}

# Function to start services
start_services() {
    echo "🚀 Starting database services..."
    $COMPOSE_CMD up -d neo4j mongodb
    
    echo "⏳ Waiting for services to be ready..."
    
    # Wait for Neo4j
    echo "Waiting for Neo4j..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if $COMPOSE_CMD exec -T neo4j cypher-shell -u neo4j -p knowledge123 "RETURN 1" >/dev/null 2>&1; then
            echo "✅ Neo4j is ready"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "❌ Neo4j failed to start within timeout"
        exit 1
    fi
    
    # Wait for MongoDB
    echo "Waiting for MongoDB..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if $COMPOSE_CMD exec -T mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
            echo "✅ MongoDB is ready"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "❌ MongoDB failed to start within timeout"
        exit 1
    fi
}

# Function to setup Neo4j indexes and constraints
setup_neo4j() {
    echo "🔧 Setting up Neo4j indexes and constraints..."
    
    $COMPOSE_CMD exec -T neo4j cypher-shell -u neo4j -p knowledge123 << 'EOF'
CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id);
CREATE INDEX node_type_index IF NOT EXISTS FOR (n:Node) ON (n.node_type);
CREATE INDEX session_id_index IF NOT EXISTS FOR (s:Session) ON (s.id);
CREATE INDEX edge_type_index IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.edge_type);
CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE;
EOF
    
    echo "✅ Neo4j setup complete"
}

# Function to start optional services
start_optional_services() {
    read -p "🤔 Start Neo4j Browser and MongoDB Express for database management? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🌐 Starting web interfaces..."
        $COMPOSE_CMD up -d neo4j-browser mongo-express
        echo "✅ Web interfaces started"
        echo "   Neo4j Browser: http://localhost:8080"
        echo "   MongoDB Express: http://localhost:8081 (admin/knowledge123)"
    fi
}

# Function to display connection info
show_connection_info() {
    echo ""
    echo "🔌 Database Connection Information:"
    echo "================================="
    echo "Neo4j:"
    echo "  URI: bolt://localhost:7687"
    echo "  Username: neo4j"
    echo "  Password: knowledge123"
    echo "  Browser: http://localhost:7474"
    echo ""
    echo "MongoDB:"
    echo "  URI: mongodb://localhost:27017"
    echo "  Database: socratic_lab"
    echo ""
}

# Function to install Python dependencies
install_python_deps() {
    if command -v python3 >/dev/null 2>&1; then
        echo "🐍 Installing Python dependencies..."
        python3 -m pip install -e .
        echo "✅ Python dependencies installed"
    else
        echo "⚠️  Python3 not found. Please install Python 3.8+ and run: pip install -e ."
    fi
}

# Function to test the installation
test_installation() {
    read -p "🧪 Test the installation? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Testing peengine installation..."
        if command -v peengine >/dev/null 2>&1; then
            peengine status
        else
            echo "⚠️  peengine command not found. Try: python -m peengine.cli status"
        fi
    fi
}

# Main execution
main() {
    check_docker
    check_docker_compose
    setup_env
    start_services
    setup_neo4j
    start_optional_services
    install_python_deps
    show_connection_info
    test_installation
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env with your Azure OpenAI credentials"
    echo "2. Run: peengine init"
    echo "3. Start exploring: peengine start 'your topic'"
    echo ""
    echo "To stop services: $COMPOSE_CMD down"
    echo "To view logs: $COMPOSE_CMD logs -f"
}

# Handle script arguments
case "${1:-}" in
    "start")
        check_docker
        check_docker_compose
        start_services
        ;;
    "stop")
        check_docker_compose
        $COMPOSE_CMD down
        echo "✅ Services stopped"
        ;;
    "restart")
        check_docker_compose
        $COMPOSE_CMD down
        start_services
        ;;
    "logs")
        check_docker_compose
        $COMPOSE_CMD logs -f "${2:-}"
        ;;
    "status")
        check_docker_compose
        $COMPOSE_CMD ps
        ;;
    *)
        main
        ;;
esac