import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np

load_dotenv()

INDEX_NAME = "stocks"
NAMESPACE = "stock-descriptions"

filter = {'$and': [{'Sector': {'$in': ['Technology', 'Health Care']}}, {'State': {'$in': ['CA', 'NC']}}]}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pinecone_client.Index(INDEX_NAME)

top_matches = pinecone_index.query(
    vector=np.random.rand(768).tolist(),
    namespace=NAMESPACE,
    filter=filter,
    top_k=10,
    include_metadata=True
)

print(top_matches)