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
- **Smart Query Interface**: Context-aware responses using semantic search instead of rigid function calling
- **Thought Journal**: Record text, audio, and image entries with automatic linking to transactions
  - **Multi-modal Support**: Text, audio recordings, and images
  - **Semantic Linking**: Automatically links journal entries to similar transactions
  - **Manual Linking**: Connect specific entries to transactions
  - **Search & Analytics**: Find entries and analyze patterns
  - **Mood & Importance Tracking**: Tag entries with emotional context

## Thought Journal

The thought journal feature allows you to:

### Create Entries
- **Text entries**: Write thoughts, notes, and reflections
- **Audio entries**: Record voice notes with optional transcription
- **Image entries**: Upload photos with descriptions (receipts, screenshots, etc.)
- **Mixed entries**: Combine text, audio, and images in one entry

### Smart Linking
- **Automatic linking**: Uses semantic similarity to connect journal entries to relevant transactions
- **Manual linking**: Explicitly link entries to specific transactions
- **Similarity scoring**: See how closely related your thoughts are to your spending

### Organization & Search
- **Tags**: Organize entries with custom tags
- **Mood tracking**: Rate your emotional state (-1 to 1)
- **Importance scoring**: Mark significance (0 to 1)
- **Semantic search**: Find entries by meaning, not just keywords
- **Analytics**: View patterns in your journaling habits

### Use Cases
- **Receipt documentation**: Photo + description automatically linked to purchase
- **Spending reflections**: Text entries about financial decisions
- **Voice notes**: Quick thoughts about purchases or financial goals
- **Mood tracking**: Understand emotional patterns in spending
- **Financial planning**: Document thoughts about future expenses

## Embedding Management

The system includes robust tools for managing embeddings, especially useful after OpenAI API quota issues:

### Check Embedding Status
```bash
# Check current embedding status
python scripts/manage_embeddings.py --check

# Test single embedding creation
python scripts/manage_embeddings.py --test
```

### Repopulate Missing Embeddings
```bash
# Repopulate all missing embeddings
python scripts/manage_embeddings.py --repopulate

# Repopulate with custom batch size and retry settings
python scripts/manage_embeddings.py --repopulate --batch-size 25 --max-retries 5 --delay 2.0

# Repopulate and rebuild graph
python scripts/manage_embeddings.py --repopulate --rebuild-graph
```

### Advanced Data Loading
```bash
# Load data with embedding repopulation
python scripts/load_data.py

# Only repopulate embeddings (skip CSV loading)
python scripts/load_data.py --repopulate-only

# Check embedding status
python scripts/load_data.py --check-status
```

### System Health Monitoring
```bash
# Single health check
python scripts/monitor_health.py

# Continuous monitoring
python scripts/monitor_health.py --continuous --interval 30

# Detailed output with configuration
python scripts/monitor_health.py --detailed

# JSON output for automated monitoring
python scripts/monitor_health.py --json --log
```

## Troubleshooting

### OpenAI API Issues
- Use `--test` flag to verify API connectivity
- Check your API key and quota in the OpenAI dashboard
- Failed embeddings are stored as zero vectors and can be repopulated

### Database Issues
- Use health monitoring to check database connectivity
- Verify connection string in `.env` file
- Check PostgreSQL service status

### Embedding Repopulation
- The system automatically detects missing (null) and failed (zero) embeddings
- Repopulation is idempotent - safe to run multiple times
- Use smaller batch sizes if experiencing rate limiting
