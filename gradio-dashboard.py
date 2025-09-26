import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables
load_dotenv()

# --------------------------
# Load Book Data Safely
# --------------------------
books_path = "books_with_emotions.csv"
if not os.path.exists(books_path):
    raise FileNotFoundError(f"CSV file not found: {books_path}")

books = pd.read_csv(books_path)

# Add large thumbnails or fallback image
books["large_thumbnail"] = books['thumbnail'].astype(str) + "&fife=w800"
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.png',
    books['large_thumbnail']
)

# --------------------------
# Load and Split Documents
# --------------------------
txt_path = "tagged_description.txt"
if not os.path.exists(txt_path):
    raise FileNotFoundError(f"Tagged description file not found: {txt_path}")

try:
    raw_documents = TextLoader(txt_path, encoding="utf-8").load()
except Exception as e:
    raise RuntimeError(f"Could not load text file {txt_path}. Error: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n'],
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(raw_documents)

# --------------------------
# Build Vector DB
# --------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

# --------------------------
# Recommendation Logic
# --------------------------
# Function to retrieve semantic recommendations
def retrieve_semantic_recommendation(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract ISBNs safely as strings
    books_list = []
    for rec in recs:
        if rec.page_content.strip():
            raw_text = rec.page_content.strip().replace('"', '').replace("'", "")
            isbn = raw_text.split()[0]  # take first token
            books_list.append(isbn)

    # Filter recommendations from the books dataset
    books_recs = books[books["isbn13"].astype(str).isin(books_list)].head(initial_top_k)

    # Filter by category
    if category != 'All':
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    # Sort by emotional tone
    if tone == 'Happy':
        books_recs = books_recs.sort_values(by='joy', ascending=False)
    elif tone == 'Surprising':
        books_recs = books_recs.sort_values(by='surprise', ascending=False)
    elif tone == 'Angry':
        books_recs = books_recs.sort_values(by='anger', ascending=False)
    elif tone == 'Suspenseful':
        books_recs = books_recs.sort_values(by='fear', ascending=False)
    elif tone == 'Sad':
        books_recs = books_recs.sort_values(by='sadness', ascending=False)

    return books_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row.get('description', 'No description available'))
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = str(row.get("authors", "Unknown")).split(":")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row.get("authors", "Unknown")

        caption = f"{row.get('title', 'Untitled')} by {authors_str}: {truncated_description}"
        results.append((row.get('large_thumbnail', 'cover-not-found.png'), caption))

    return results

# --------------------------
# Gradio UI
# --------------------------
categories = ['All'] + sorted(books['simple_categories'].dropna().unique())
tones = ['All', 'Happy', 'Suspenseful', 'Angry', 'Surprising', 'Sad']

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a description of a book",
                                placeholder='e.g., A story about forgiveness')
        category_dropdown = gr.Dropdown(choices=categories, label='Category:', value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == '__main__':
    dashboard.launch(share=True)
