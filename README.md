# Graph-Powered Transaction Analytics Engine

Turn a month's worth of transaction CSVs into an interactive, graph-powered analytics engine, enriched with LLM embeddings and accessible via plain-English queries.

## Components

1. **Relational Store**: PostgreSQL for raw transaction data
2. **Embedding Service**: OpenAI API for semantic embeddings
3. **Graph Layer**: NetworkX + PyTorch Geometric for graph operations
4. **Graph Neural Network**: GNN for enhanced insights
5. **Natural Language Interface**: Chat UI for plain-English queries

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and database credentials
```

3. Initialize the database:
```bash
python scripts/init_db.py
```

4. Load and process data:
```bash
python scripts/load_data.py
```

5. Start the application:
```bash
streamlit run app.py
```

## Data Flow

1. CSV → DataFrame → PostgreSQL
2. Description text → OpenAI Embeddings → Store vectors
3. Build graph (nodes ← embeddings, edges ← similarity/metadata)
4. Train/run GNN for insight extraction
5. Chat UI interprets questions → backend functions → results

## Features

- **Accurate Financial Data**: All numbers stored in PostgreSQL
- **Semantic Understanding**: LLM embeddings capture transaction meaning
- **Graph Insights**: Discover multi-hop relationships and patterns
- **Natural Language Queries**: Ask questions in plain English
- **Interactive Visualizations**: Explore data with graphs and charts
