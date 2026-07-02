# FAISS Index setup and search utility
import os
import json
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from build_index import build

CATALOG = {}
INDEX: Optional[FAISS] = None

# Initialize the FAISS index
def init_search_index():
    global CATALOG, INDEX
    try:
        # Load the base dictionary map
        with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            CATALOG = {str(item["entity_id"]): item for item in data if "entity_id" in item}
            
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        path = "shl_index"
        if not os.path.exists(path):
            build()
        
        INDEX = FAISS.load_local("shl_index", embeddings, allow_dangerous_deserialization=True)
        print("Loaded catalog records instantly via stored FAISS binaries.")
    except Exception as e:
        print(f"Failed to load local index binary: {e}")

# Search utility to retrieve matching catalog items
def get_matches(keywords: List[str], max_results: int = 5) -> List[dict]:
    if not INDEX or not keywords:
        return []
        
    seen = set()
    results = []
    for kw in keywords:
        hits = INDEX.similarity_search(kw, k=3)
        for doc in hits:
            eid = doc.metadata["id"]
            if eid in CATALOG and eid not in seen:
                seen.add(eid)
                results.append(CATALOG[eid])
        if len(results) >= max_results:
            break
            
    return results[:max_results]