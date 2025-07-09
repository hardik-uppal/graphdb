# Usage Guide - Graph-Powered Transaction Analytics

## Quick Start

### 1. Environment Setup
```bash
# Clone or navigate to the project directory
cd /home/hardik/Projects/graphdb

# Set up your OpenAI API key in .env file
# Edit .env and replace "your_openai_api_key_here" with your actual API key
```

### 2. Database Setup

#### Option A: Using Docker (Recommended)
```bash
# Start PostgreSQL with Docker
docker run --name postgres-analytics -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=transaction_analytics -p 5432:5432 -d postgres:15

# Or use docker-compose (starts both PostgreSQL and the app)
export OPENAI_API_KEY="your-actual-api-key"
docker-compose up
```

#### Option B: Local PostgreSQL
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb transaction_analytics

# Initialize database schema
python scripts/init_db.py
```

### 3. Load Your Data
```bash
# Load and process the transaction CSV
python scripts/load_data.py
```

### 4. Start the Application
```bash
# Launch the Streamlit interface
streamlit run app.py
```

## Features Overview

### 💬 Chat Interface
Ask natural language questions about your transactions:

**Example Questions:**
- "What did I spend the most money on?"
- "Show me all transactions related to coffee or food"
- "What's my monthly spending summary for October?"
- "Find unusual or anomalous transactions"
- "What are my top 5 merchants by spending?"
- "Show me all investment transactions"
- "What transaction patterns can you detect?"

**How it works:**
1. Your question is sent to OpenAI GPT-4
2. GPT-4 determines which analysis function to call
3. The system queries your PostgreSQL database
4. Results are enhanced with graph insights
5. A natural language response is generated

### 📊 Dashboard
Visual overview of your financial data:
- **Summary Metrics**: Total transactions, income, expenses, net amount
- **Category Breakdown**: Pie chart of spending by transaction type
- **Monthly Trends**: Bar chart showing income vs expenses over time
- **Top Merchants**: Table of highest spending destinations

### 🕸️ Graph Explorer
Visualize transaction relationships:
- **Graph Statistics**: Nodes, edges, density, connectivity
- **Interactive Visualization**: See how transactions connect
- **Transaction Clusters**: Groups of similar transactions
- **Relationship Types**:
  - Similarity edges (based on semantic similarity)
  - Temporal edges (transactions close in time)
  - Merchant edges (same merchant connections)

### 🔍 Transaction Search
Semantic search powered by embeddings:
- Search using natural language descriptions
- Find transactions similar to "coffee", "grocery", "gas", etc.
- Results ranked by semantic similarity
- No need for exact keyword matches

### 📈 Analytics
Advanced analysis features:
- **Anomaly Detection**: Find unusual transaction patterns
- **Time Series Analysis**: Spending patterns over time
- **Category Trends**: Track spending changes by category
- **Graph-based Insights**: Leverage network analysis

## System Architecture

### 1. Data Layer
- **PostgreSQL**: Stores raw transaction data with accuracy guarantees
- **Embeddings**: OpenAI text-embedding-3-small for semantic vectors
- **Graph Storage**: NetworkX for in-memory graph operations

### 2. Processing Layer
- **Embedding Service**: Converts transaction descriptions to vectors
- **Graph Service**: Builds and analyzes transaction networks
- **GNN Model**: PyTorch Geometric for enhanced representations

### 3. Interface Layer
- **Query Interface**: Natural language to SQL/graph operations
- **Streamlit App**: Interactive web interface
- **Visualization**: Plotly for charts and graph displays

## Data Flow

```
CSV File → Pandas DataFrame → PostgreSQL
    ↓
Transaction Text → OpenAI API → Embedding Vectors
    ↓
Embeddings + Metadata → Graph Construction → NetworkX Graph
    ↓
Graph + Features → GNN Training → Enhanced Representations
    ↓
User Question → OpenAI GPT-4 → Function Selection → Query Execution
    ↓
Database Results + Graph Insights → Natural Language Response
```

## API Reference

### Core Classes

#### `DataLoader`
Handles CSV processing and database loading.
```python
from src.data_loader import DataLoader

loader = DataLoader()
transactions = loader.process_csv_file("scotiabank.csv", db_session)
stats = loader.get_summary_statistics(transactions)
```

#### `EmbeddingService`
Manages OpenAI embeddings and similarity calculations.
```python
from src.embedding_service import EmbeddingService

service = EmbeddingService()
embedding = service.create_embedding("coffee purchase")
similar = service.find_similar_transactions(embedding, transactions)
```

#### `GraphService`
Constructs and analyzes transaction graphs.
```python
from src.graph_service import GraphService

graph_service = GraphService()
graph_service.build_graph_from_transactions(transactions, db_session)
clusters = graph_service.detect_transaction_clusters(transactions, db_session)
```

#### `QueryInterface`
Processes natural language queries.
```python
from src.query_interface import QueryInterface

interface = QueryInterface()
result = interface.process_natural_language_query("How much did I spend on food?", db_session)
```

## Configuration

### Environment Variables (.env)
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=transaction_analytics
DB_USER=postgres
DB_PASSWORD=postgres

# Graph parameters
EMBEDDING_DIMENSION=1536
SIMILARITY_THRESHOLD=0.8
```

### Graph Configuration
- **Similarity Threshold**: Minimum cosine similarity for embedding edges (0.8)
- **Temporal Window**: Connect transactions within 3 days
- **Merchant Grouping**: Connect all transactions from same merchant

## Troubleshooting

### Common Issues

#### "No module named 'src'"
```bash
# Make sure you're in the project directory
cd /home/hardik/Projects/graphdb
python scripts/init_db.py
```

#### "Connection refused" (PostgreSQL)
```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Or use Docker
docker run --name postgres-analytics -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15
```

#### "Invalid API key" (OpenAI)
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY

# Make sure the key is valid and has credits
```

#### "Empty results" from queries
```bash
# Reload data to ensure embeddings are created
python scripts/load_data.py
```

### Performance Tips

1. **Large Datasets**: For >10,000 transactions, consider batching graph operations
2. **Memory Usage**: GNN training can be memory-intensive; reduce batch size if needed
3. **Query Speed**: Complex graph analysis may take time; use caching for repeated queries
4. **API Costs**: OpenAI embeddings cost ~$0.0001 per 1K tokens; ~$0.10 for 1000 transactions

## Extending the System

### Adding New Data Sources
1. Create a new parser in `DataLoader`
2. Map columns to the standard Transaction schema
3. Update the CSV processing pipeline

### Custom Analysis Functions
1. Add new functions to `QueryInterface.functions`
2. Implement the function in `QueryInterface`
3. Update the OpenAI function calling prompts

### Enhanced Visualizations
1. Add new chart types to the Streamlit app
2. Extend the graph visualization with additional node/edge attributes
3. Create specialized dashboards for different user types

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Database**: Use strong passwords and connection encryption
- **Data Privacy**: Transaction data contains sensitive financial information
- **Network**: Consider firewall rules for production deployments

## Deployment

### Production Deployment
1. Use environment-specific configuration
2. Set up proper database backups
3. Configure load balancing for high traffic
4. Monitor API usage and costs
5. Implement proper logging and error handling

### Scaling
- **Database**: Use PostgreSQL clustering or cloud solutions
- **Embeddings**: Cache frequently accessed embeddings
- **Graph**: Consider graph databases (Neo4j) for very large datasets
- **Compute**: Use GPU acceleration for large GNN models

## Recent Updates (July 2025)

### OpenAI API Compatibility
- ✅ **Updated to latest OpenAI API**: The system now uses the modern `tools` parameter instead of the deprecated `functions` parameter
- ✅ **Tool calling**: Properly handles `tool_calls` in the response format
- ✅ **Automatic fallback**: Falls back to keyword-based queries when OpenAI quota is exceeded
- ✅ **Error handling**: Robust error handling for API quota limits and connection issues
