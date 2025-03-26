# FAISS

```
!pip install faiss-cpu
!pip install sentence-transformers
```

```
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data
documents = ["This is document 1", "This is document 2", "Document 3 content"]

# Generate embeddings
embeddings = model.encode(documents)
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(np.array(embeddings))       # Add embeddings to the index
```

```
query = "What is document retrieval?"
query_embedding = model.encode([query])

# Search for top 3 nearest neighbors
D, I = index.search(np.array(query_embedding), k=2)
print("Top documents:", [documents[i] for i in I[0]])
```