#!/usr/bin/env python3
"""Installation and setup script for Zerdisha."""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Found:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_neo4j():
    """Check if Neo4j is available."""
    try:
        result = subprocess.run(['neo4j', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Neo4j detected")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  Neo4j not detected. You'll need to:")
    print("   • Install Neo4j Desktop or")
    print("   • Set up Neo4j cloud instance or")
    print("   • Run Docker: docker run -p 7474:7474 -p 7687:7687 neo4j")
    return False


def setup_environment():
    """Set up environment file."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from example")
            print("⚠️  Please edit .env with your API keys and database credentials")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def test_installation():
    """Test if installation works."""
    print("🧪 Testing installation...")
    try:
        result = subprocess.run([sys.executable, "-c", "import peengine; print('Import successful')"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Installation test passed")
            return True
        else:
            print(f"❌ Import test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Installation test timed out")
        return False


def main():
    """Main installation process."""
    print("🧠 Zerdisha - Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    check_neo4j()  # Warning only, not blocking
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Edit .env with your API keys and Neo4j credentials")
    print("2. Start Neo4j database")
    print("3. Run: zerdisha init")
    print("4. Start exploring: zerdisha start 'your topic'")
    print("\nFor help: zerdisha --help")


if __name__ == "__main__":
    main()