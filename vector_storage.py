import itertools
from gpt_index import OpenAIEmbedding
import pinecone
import os
import dotenv
import json
import csv
import pandas as pd
import openai
import numpy as np
import itertools
from embedding import createEmbeddings


dotenv.load_dotenv()
pinecone.init(os.getenv("PINECONE_API_KEY"), environment="us-east-1-aws")
openai.api_key = os.getenv("OPENAI_API_KEY")
inputCSV = "data/pm_embeddings_with_id.csv"

indexName = "pm-embeddings-001"

index = pinecone.Index(index_name=indexName)

df = pd.read_csv(inputCSV, usecols=["pm_id", "embedding", "combined", "authors"])
df["embedding"] = df.embedding.apply(eval).apply(list)

idEmbeddingPairs = list(zip(df['pm_id'].astype(str), df['embedding']))


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


vector_dim = 128
vector_count = len(idEmbeddingPairs)

upserting = False

if upserting:
    # # Upsert data with 100 vectors per upsert request
    numChunks = 0
    for ids_vectors_chunk in chunks(idEmbeddingPairs, batch_size=100):
        # print(ids_vectors_chunk[0][0])
        print(numChunks)
        numChunks += 1
        # Assuming `index` defined elsewhere
        index.upsert(vectors=ids_vectors_chunk, namespace="PM_ID")
