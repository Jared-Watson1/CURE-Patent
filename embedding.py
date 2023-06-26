from openai.embeddings_utils import (
    get_embedding,
    cosine_similarity,
)
import dotenv
import os
import openai
import pandas as pd
import tiktoken
import numpy as np
import pinecone
dotenv.load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
from metapub import PubMedFetcher

openai.api_key = os.getenv("OPENAI_API_KEY")

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
top_n = 1000

input_datapath = "data/pm_eutils.csv"
output_embeddings_path = "data/pm_embeddings_with_id.csv"
import multiprocessing

fetch = PubMedFetcher()

def makeContext(ids):
    context = ""
    for id in ids:
        content = str(fetch.article_by_pmid(id).abstract)
        context += f"PM ID: {id}: " + content + "\n"
    return context

def process_chunk(chunk):
    chunk = chunk[["pm_id", "title", "authors", "abstract"]]
    chunk = chunk.dropna()
    chunk["combined"] = (
        "Title: " + chunk.title.str.strip() + "; Abstract: " + chunk.abstract.str.strip()
    )

    encoding = tiktoken.get_encoding(embedding_encoding)

    # Omit reviews that are too long to embed
    chunk["n_tokens"] = chunk.combined.apply(lambda x: len(encoding.encode(x)))
    chunk = chunk[chunk.n_tokens <= max_tokens]

    print(f"Processing {len(chunk)} articles in this chunk")
    chunk["embedding"] = chunk.combined.apply(
        lambda x: get_embedding(x, engine=embedding_model))

    return chunk[["pm_id", "embedding", "combined", "authors"]]

def createEmbeddings(inputFile=input_datapath, chunk_size=250):
    # Read the input file in chunks
    chunk_iterator = pd.read_csv(inputFile, iterator=True, chunksize=chunk_size)

    # Initialize an empty DataFrame to store the final embeddings
    final_embeddings_df = pd.DataFrame(columns=["pm_id", "embedding", "combined", "authors"])
    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    # Process each chunk in parallel
    processed_chunks = pool.map(process_chunk, chunk_iterator)

    # Concatenate the processed chunks into the final DataFrame
    final_embeddings_df = pd.concat(processed_chunks, ignore_index=True)

    # Save the final embeddings DataFrame to a CSV file
    final_embeddings_df.to_csv(output_embeddings_path, index=False)

    print(f"Total embeddings generated: {len(final_embeddings_df)}")


def filter(df, inp, n=3, pprint=False):
    df.head()
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    input_embedding = get_embedding(
        inp,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, input_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    if pprint:
        for r in results:
            print(r[:200])
            print()

    return pd.array(results).astype(str)


pinecone.init(os.getenv("PINECONE_API_KEY"), environment="us-east-1-aws")


def pineconeFilter(query, topK=3, index="pm-embeddings-001"):
    queryEmbedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    # print(queryEmbedding)
    index = pinecone.Index(index)
    pineconeQuery = index.query(
        namespace="PM_ID",
        vector=queryEmbedding,
        top_k=topK,
        include_values=False
    )
    # print(pineconeQuery['matches'])
    ids = []
    for i in pineconeQuery['matches']:
        try:
            ids.append(i['id'])
            print('id')
        except:
            ids.append(i['pm_id'])
            print('pm_id')
        # ids.append(i['id'])
    return makeContext(ids), ids
   

# createEmbeddings()
