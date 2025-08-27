"""
agent_app.py
------------
Streamlit app with LangChain agent for natural Q&A over Amazon reviews.
"""

import json
import re
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoConfig

from data_pipeline import load_data, preprocess
from nlp_models import SentimentAnalyzer, Summarizer, Embedder
from vector_search import ReviewVectorSearch

import os
from dotenv import load_dotenv
import openai

# -------------------------------
# Load Config
# -------------------------------
with open("notebooks/config.json", "r") as f:
    CONFIG = json.load(f)


# -------------------------------
# Initialize Models
# -------------------------------
@st.cache_resource
def init_models():
    sentiment = SentimentAnalyzer(CONFIG["sentiment_model"])
    summarizer = Summarizer(CONFIG["summarizer_model"])
    embedder = Embedder(CONFIG["embedding_model"])
    return sentiment, summarizer, embedder


# -------------------------------
# Build / Load Index
# -------------------------------
@st.cache_resource
def init_index(_embedder, data_path="data/Reviews.csv", sample_size=5000):
    df = load_data(data_path)
    df = preprocess(df)

    # Subsample for faster demo
    texts = df["Text_clean"].dropna().astype(str).tolist()[:sample_size]

    vs = ReviewVectorSearch(_embedder, dim=384)
    vs.build_index(texts)
    return vs, texts


# -------------------------------
# LangChain RetrievalQA
# -------------------------------
def make_retriever(vs, summarizer):
    """
    Wrap vector search + summarizer into a pseudo retriever.
    """
    def retrieve_and_summarize(query: str) -> str:
        results = vs.query(query, top_k=5)
        reviews = [r[0] for r in results]
        combined = " ".join(reviews)
        return summarizer.summarize(combined, max_len=150, min_len=40)
    return retrieve_and_summarize


# -------------------------------
# Streamlit App
# -------------------------------

# this app can be run using: streamlit run src/agent_app.py

def openai_summarize_and_validate(reviews, user_query, initial_summary):
    """
    Use OpenAI to summarize reviews and ensure the summary matches the reviews and the query.
    The summary should clearly reference and synthesize the actual content of the reviews.
    """
    # Load OpenAI API key from .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Prepare prompt for OpenAI
    prompt = (
        "You are an expert assistant. Given the following user question and a set of customer reviews, "
        "your task is to generate a concise, accurate summary that answers the question using only information from the reviews. "
        "Your summary MUST clearly reference and synthesize the actual content of the reviews, and should not include information not present in them. "
        "If the initial summary is inaccurate or does not match the reviews, improve it. "
        "Be truthful to the reviews and do not hallucinate. "
        "If possible, cite specific points or recurring themes from the reviews. "
        "If the reviews do not contain enough information to answer the question, say so explicitly.\n\n"
        f"User Question: {user_query}\n\n"
        f"Customer Reviews:\n"
        + "\n".join([f"- {r}" for r in reviews])
        + "\n\n"
        f"Initial Summary: {initial_summary}\n\n"
        "Final, improved summary (must be based only on the reviews above, and clearly reflect their content):"
    )

    # Use OpenAI to summarize and validate the summary
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant for summarizing customer reviews. "
                    "Your summaries must be strictly grounded in the provided reviews, and should reference specific points or themes from them. "
                    "If the reviews do not answer the question, say so."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=220,
            temperature=0.3,
        )
        improved_summary = response.choices[0].message.content.strip()
        return improved_summary
    except Exception as e:
        print(f"Error with OpenAI: {e}")
        # Fallback to initial summary if OpenAI fails
        return initial_summary

def main():
    st.title("🛒 Customer Review Insight Extractor")
    st.write("Ask natural questions about products that can be found on Amazon and get structured insights. Some common products you can ask about are: dog/cat food, coffee, tea, energy drinks, cookies, and peanut butter!")

    # Initialize models + index
    sentiment, summarizer, embedder = init_models()
    vs, texts = init_index(embedder)

    retriever = make_retriever(vs, summarizer)

    # Sidebar
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of reviews to consider", 3, 10, 5)

    # Input box
    query = st.text_input("Ask a question (e.g., 'What do people dislike about dog food?')")

    def fix_grammar(text):
        """
        Attempt to fix basic grammar/formatting issues.
        - Capitalize first letter.
        - Ensure ends with period.
        - Remove excessive whitespace.
        """
        import re
        text = text.strip()
        if not text:
            return text
        text = text[0].upper() + text[1:]
        if not text.endswith(('.', '!', '?')):
            text += '.'
        text = re.sub(r'\s+', ' ', text)
        return text

    if query:
        with st.spinner("Searching reviews..."):
            results = vs.query(query, top_k=top_k)
            reviews = [r[0] for r in results]

            # First, summarize with local model
            initial_summary = summarizer.summarize(" ".join(reviews))
            # Then, use OpenAI to validate and improve the summary
            summary = openai_summarize_and_validate(reviews, query, initial_summary)
            summary = fix_grammar(summary)

        st.subheader("📌 Insight Summary")
        st.write(summary)

        with st.expander("See top matched reviews"):
            for rev, dist in results:
                st.write(f"- {fix_grammar(rev)} (score={dist:.4f})")


if __name__ == "__main__":
    main()