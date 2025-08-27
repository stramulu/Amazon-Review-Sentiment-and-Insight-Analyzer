"""
data_pipeline.py
---------------
Handles loading, cleaning, and preprocessing of Amazon review data.
"""

# imports
import re
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load Amazon review dataset from CSV.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw reviews DataFrame.
    """
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """
    Clean individual review text:
    - Lowercase
    - Remove HTML tags and URLs
    - Remove non-alphanumeric chars (except spaces)
    - Collapse multiple spaces

    Args:
        text (str): Raw review text.

    Returns:
        str: Cleaned review text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                 # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)        # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()           # collapse spaces
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from the reviews DataFrame:
    - Clean Summary + Text
    - Compute helpfulness ratio
    - Drop rows with missing/empty reviews

    Args:
        df (pd.DataFrame): Raw review DataFrame.

    Returns:
        pd.DataFrame: Preprocessed reviews DataFrame.
    """
    df = df.copy()

    # Clean text fields
    df["Summary_clean"] = df["Summary"].apply(clean_text)
    df["Text_clean"] = df["Text"].apply(clean_text)

    # Compute helpfulness ratio (avoid division by zero)
    df["helpfulness_ratio"] = df.apply(
        lambda x: x["HelpfulnessNumerator"] / x["HelpfulnessDenominator"]
        if x["HelpfulnessDenominator"] > 0 else 0,
        axis=1
    )

    # Drop rows with empty cleaned reviews
    df = df[(df["Summary_clean"].str.len() > 0) | (df["Text_clean"].str.len() > 0)]

    return df


if __name__ == "__main__":
    # Example usage
    raw = load_data("data/Reviews.csv")
    processed = preprocess(raw)

    print("Raw shape:", raw.shape)
    print("Processed shape:", processed.shape)
    print(processed.head())