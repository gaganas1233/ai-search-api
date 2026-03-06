import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = np.load("embeddings/product_embeddings_v2.npy")
products_df = pd.read_csv("data/cleaned_products.csv")

def search_products(query, top_k=10):

    query_embedding = model.encode([query])

    similarities = np.dot(embeddings, query_embedding.T).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = products_df.iloc[top_indices]

    return results.to_dict(orient="records")