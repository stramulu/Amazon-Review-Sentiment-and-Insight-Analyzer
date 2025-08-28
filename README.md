# **Amazon Review Sentiment and Insight Analyzer**

A comprehensive system for analyzing Amazon customer reviews using natural language processing and vector search. The application provides an interactive web interface for querying review insights and generating structured summaries. OpenAI is also used for output verification to ensure the accuracy and relevance of generated summaries.

## Features

- **Semantic Search**: Find relevant reviews using vector embeddings and similarity search
- **Intelligent Summarization**: Generate concise summaries from review sets using BART and OpenAI
- **Sentiment Analysis**: Analyze review sentiment using BERT-based models
- **Interactive Web Interface**: Streamlit-based UI for natural language queries
- **Data Preprocessing**: Automated cleaning and feature engineering for review data
- **Vector Database**: FAISS-based indexing for fast similarity search
- **Output Verification**: Summaries are validated and improved using OpenAI to ensure they accurately reflect the underlying reviews

## Architecture

The system consists of several key components:

- **Data Pipeline** (`src/data_pipeline.py`): Handles data loading, cleaning, and preprocessing
- **NLP Models** (`src/nlp_models.py`): Wrappers for sentiment analysis, summarization, and embedding models
- **Vector Search** (`src/vector_search.py`): FAISS-based similarity search implementation
- **Web Application** (`src/agent_app.py`): Streamlit interface for user interaction

## Prerequisites

- Python 3.8+
- OpenAI API key (for enhanced summarization and output verification)
- 8GB+ RAM (for model loading)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-review-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Dataset Setup

This project uses the **Amazon Product Reviews dataset** from Kaggle, which contains 568,000+ customer reviews.  

Since the file is too large to upload directly, you’ll need to download it manually:

1. Go to the Kaggle dataset page:  
   👉 [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)  

2. Download the file `Reviews.csv`.  

3. Place the file into the project’s `data/` folder:  

## Usage

### Web Application

Run the Streamlit application:
```bash
streamlit run src/agent_app.py
```

The application will:
- Load and preprocess the review data
- Build a vector index for semantic search
- Initialize NLP models for sentiment analysis and summarization
- Provide an interactive interface for querying reviews

### Example Queries

- "What do people dislike about dog food?"
- "What are the most common complaints about coffee?"
- "What features do customers love about energy drinks?"
- "What are the main issues with peanut butter products?"

### Jupyter Notebooks

Explore the data and models using the provided notebooks:
- `notebooks/01_eda.ipynb`: Exploratory data analysis
- `notebooks/02_model_prototyping.ipynb`: Model development and testing

## Configuration

Model configurations are stored in `notebooks/config.json`:

```json
{
  "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
  "summarizer_model": "facebook/bart-large-cnn", 
  "embedding_model": "multi-qa-MiniLM-L6-cos-v1"
}
```

## Model Details

### Sentiment Analysis
- **Model**: BERT-based multilingual sentiment classifier
- **Output**: 1-5 star ratings with confidence scores
- **Use Case**: Understanding customer satisfaction levels

### Summarization
- **Primary Model**: BART-large-CNN for initial summarization
- **Enhanced Model**: OpenAI GPT-3.5-turbo for validation and improvement
- **Features**: Chunking for long texts, grammar correction, hallucination prevention

### Embeddings
- **Model**: SentenceTransformers multi-qa-MiniLM-L6-cos-v1
- **Dimension**: 384-dimensional vectors
- **Use Case**: Semantic similarity search and clustering

### Vector Search
- **Index**: FAISS IndexFlatL2 for L2 distance similarity
- **Features**: Fast approximate nearest neighbor search
- **Storage**: Persistent index with pickle serialization

## Data Processing Pipeline

1. **Loading**: CSV data import with pandas
2. **Cleaning**: Text normalization, HTML removal, URL filtering
3. **Feature Engineering**: Helpfulness ratio calculation
4. **Filtering**: Removal of empty or invalid reviews
5. **Indexing**: Vector embedding generation and FAISS index construction

## Performance Considerations

- **Memory Usage**: Models are cached using Streamlit's cache_resource decorator
- **Processing Speed**: Vector search provides sub-second query response times
- **Scalability**: FAISS index supports millions of reviews efficiently
- **Model Loading**: Lazy loading prevents memory issues during startup

## Development

### Project Structure
```
customer-review-analysis/
├── data/                   # Review datasets
├── notebooks/             # Jupyter notebooks and config
├── src/                   # Source code
│   ├── agent_app.py      # Streamlit web application
│   ├── data_pipeline.py  # Data processing utilities
│   ├── nlp_models.py     # Model wrappers
│   └── vector_search.py  # Vector search implementation
├── venv/                 # Virtual environment
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Adding New Models

1. Extend the appropriate class in `nlp_models.py`
2. Update the configuration in `notebooks/config.json`
3. Modify the application logic in `agent_app.py`

### Testing

Run the notebooks to test individual components:
```bash
jupyter notebook notebooks/
```

## Author

- Shreyas Ramulu

## Acknowledgments

- Hugging Face for transformer models
- OpenAI for GPT-3.5-turbo API
- FAISS for vector similarity search
- Streamlit for web application framework
