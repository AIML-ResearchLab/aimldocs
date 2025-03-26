# Pinecone Vector Database

## 1. Install an SDK
[Pinecone SDKs](https://docs.pinecone.io/reference/pinecone-sdks) provide convenient programmatic access to the [Pinecone APIs](https://docs.pinecone.io/reference/api/introduction).

## Install the SDK for your preferred language(python):

```
pip install "pinecone[grpc]"

# To install without gRPC run:
# pip3 install pinecone
```

## 2. Get an API key
You need an API key to make calls to your Pinecone project.

Create a new API key in the [Pinecone console](https://app.pinecone.io/organizations/-/keys), or use the widget below to generate a key. If you don’t have a Pinecone account, the widget will sign you up for the free [Starter plan](https://www.pinecone.io/pricing/).

## Your generated API key:

```
"pcsk_6Ud5Yh_xxxxxxxxxxxx"
```

## 3. Generate vectors

A [vector embedding](https://www.pinecone.io/learn/vector-embeddings/) is a numerical representation of data that enables similarity-based search in vector databases like Pinecone. To convert data into this format, you use an embedding model.

For this quickstart, use the [multilingual-e5-large](https://docs.pinecone.io/models/multilingual-e5-large) embedding model hosted by Pinecone to [create vector embeddings](https://docs.pinecone.io/guides/inference/generate-embeddings) for sentences related to the word “apple”. Note that some sentences are about the tech company, while others are about the fruit.

```
# Import the Pinecone library
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key="pcsk_6Ud5Yh_C9wzdiMJxrbuhZAyFL1gAi5Zim2fPZ1pqDmbDKEQQpBLZKmoJw8ZCfG2S56CsVL")

# Define a sample dataset where each item has a unique ID and piece of text
data = [
    {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
    {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
    {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
    {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

# Convert the text into numerical vectors that Pinecone can index
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)

print(embeddings)
```

## 4. Create an index

In Pinecone, you store data in an [index](https://docs.pinecone.io/guides/indexes/understanding-indexes).

[Create a serverless index](https://docs.pinecone.io/guides/indexes/create-an-index) that matches the dimension (1024) and similarity metric (cosine) of the multilingual-e5-large model you used in the previous step, and choose a [cloud and region](https://docs.pinecone.io/guides/indexes/understanding-indexes#cloud-regions) for hosting the index:


```
# Create a serverless index
index_name = "vectordb-test-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
```

## 5. Upsert vectors
Target your index and use the [upsert](https://docs.pinecone.io/guides/data/upsert-data) operation to load your vector embeddings into a new namespace. [Namespaces](https://docs.pinecone.io/guides/get-started/key-features#namespaces) let you partition records within an index and are essential for [implementing multitenancy](https://docs.pinecone.io/guides/get-started/implement-multitenancy) when you need to isolate the data of each customer/user.

``
In production, target an index by its unique DNS host, not by its name.
``

```
# Target the index where you'll store the vector embeddings
index = pc.Index("vectordb-test-index")

# Prepare the records for upsert
# Each contains an 'id', the embedding 'values', and the original text as 'metadata'
records = []
for d, e in zip(data, embeddings):
    records.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="vectordb-test-namespace"
)
```

``
To load large amounts of data, [import from object storage](https://docs.pinecone.io/guides/data/understanding-imports) or [upsert in large batches](https://docs.pinecone.io/guides/data/upsert-data#upsert-records-in-batches).
``

Pinecone is eventually consistent, so there can be a delay before your upserted records are available to query. Use the [describe_index_stats](https://docs.pinecone.io/guides/data/data-freshness/check-data-freshness) operation to check if the current vector count matches the number of vectors you upserted (6):

```
time.sleep(10)  # Wait for the upserted vectors to be indexed

print(index.describe_index_stats())
```

## 6. Search the index
With data in your index, let’s say you now want to search for information about “Apple” the tech company, not “apple” the fruit.

Use the the multilingual-e5-large model to convert your query into a vector embedding, and then use the [query](https://docs.pinecone.io/guides/data/query-data) operation to search for the three vectors in the index that are most semantically similar to the query vector:

```
# Define your query
query = "Tell me about the tech company known as Apple."

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

# Search the index for the three most similar vectors
results = index.query(
    namespace="vectordb-test-namespace",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)
```

## Notice that the response includes only sentences about the tech company, not the fruit:

## 7. Clean up

When you no longer need the vectordb-test-index, use the [delete_index](https://docs.pinecone.io/guides/indexes/delete-an-index) operation to delete it:


```
pc.delete_index(index_name)
```
