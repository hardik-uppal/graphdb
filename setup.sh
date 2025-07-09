#!/bin/bash

# Setup script for Graph-Powered Transaction Analytics

echo "🚀 Setting up Graph-Powered Transaction Analytics..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your OpenAI API key and database credentials"
fi

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "⚠️  PostgreSQL not found. Please install PostgreSQL first:"
    echo "   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "   macOS: brew install postgresql"
    echo "   Or use Docker: docker run --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. Start PostgreSQL service"
echo "3. Run: python scripts/init_db.py"
echo "4. Run: python scripts/load_data.py"
echo "5. Run: streamlit run app.py"
echo ""
echo "Or use Docker:"
echo "1. Set OPENAI_API_KEY environment variable"
echo "2. Run: docker-compose up"
