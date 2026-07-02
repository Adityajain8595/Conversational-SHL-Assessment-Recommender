import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Building FAISS index and serializing it to a local directory
def build():
    try:
        with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        docs = []
        for item in data:
            if "entity_id" not in item or "name" not in item:
                continue
            
            eid = str(item["entity_id"])
            text = f"""
            Name: {item.get('name', '')}
            Description: {item.get('description', '')}
            Link: {item.get('link', '')}
            Keys: {' '.join(item.get('keys', []))}
            Job Levels: {' '.join(item.get('job_levels', []))}
            """
            docs.append(Document(page_content=text, metadata={"id": eid}))
            
        print("Downloading embedding model and compiling text arrays...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        
        # Serializing index 
        db.save_local("shl_index")
        print("Success! 'shl_index' folder exported.")
        
    except Exception as e:
        print(f"Compilation variance encountered: {e}")