## 🔹 What is Text Chunking in RAG?
In Retrieval-Augmented Generation (RAG), text chunking is a foundational step that significantly impacts the performance of retrieval and generation. Choosing the right chunking strategy depends on your domain, model size, use case (e.g., question answering, summarization), and latency requirements.

Text chunking is the process of breaking large documents into smaller pieces (chunks) before embedding and storing them in a vector database for retrieval.

## 🔹 Common Chunking Strategies (Modes)


| **Mode**                    | **Description**                                                                 | **Use Case**                                                                 | **Model Details**                                                                 |
|----------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Fixed-size chunks**      | Split text by fixed number of tokens (e.g., 512 tokens)                         | General-purpose, fast and simple                                             | No model needed; tokenizers like `tiktoken`, `sentencepiece`, or `nltk`         |
| **Sliding window**         | Overlapping chunks (e.g., 512 tokens with 100-token overlap)                    | Ensures context continuity                                                   | Simple algorithmic logic with tokenizer support                                 |
| **Sentence-based**         | Break by sentence boundaries                                                    | Preserves semantic boundaries                                                | `nltk.sent_tokenize`, `spacy`, `textsplit`                                      |
| **Paragraph-based**        | Keep full paragraphs as chunks                                                  | Ideal for long-form documents                                               | Regex or NLP libraries like `nltk`, `spacy`                                     |
| **Semantic chunking**      | Split by topics or natural breakpoints using ML models                          | Best for maintaining topic coherence                                        | `BERTopic`, `SBERT`, `KeyBERT`, LLMs with attention-based segmentation          |
| **Markdown/Heading-based** | Use headings/subheadings in documents to split                                  | For structured docs like manuals, PDFs                                      | Regex, `BeautifulSoup`, `Markdown`, PDF parsers (e.g., `pdfplumber`, `PyMuPDF`) |
| **Recursive Character Split** | Hierarchical chunking (LangChain-style) from large > small (section > para > line) | Works well in structured + unstructured docs                                | `LangChain`’s `RecursiveCharacterTextSplitter`, regex                          |
| **Query-Aware Chunking**   | Splits and prioritizes chunks based on relevance to a query                     | RAG pipelines, QA systems                                                    | `BM25`, `FAISS`, `Chroma`, `OpenAI Embeddings`, `ColBERT`, `RetrieverMixin`     |
| **Token-Density-Based**    | Chunks based on token density, complexity, or information richness              | Summarization, content prioritization                                        | Token counters, `OpenAI Tokenizer`, statistical methods                         |
| **Entity-Based Chunking**  | Splits text by named entities (e.g., person, organization)                      | Information extraction, knowledge graphs                                    | `spacy`, `flair`, `transformers` NER models                                     |
| **Event-Based Chunking**   | Breaks text by narrative events or transitions                                  | Story summarization, news timeline generation                               | `EventBERT`, `GPT-4`, `NarrativeQA`, `BART` fine-tuned on events                |
| **Dialogue/Turn-Based**    | Splits by speaker turns in conversations                                        | Chatbot training, transcript analysis                                        | `Whisper`, `assemblyAI`, or transcript parsers + speaker diarization tools      |
| **Table/Structure-Aware**  | Handles structured data like tables and lists differently                       | PDFs, spreadsheets, forms                                                   | `Pandas`, `tabula-py`, `Camelot`, `layoutLM`, `PDFPlumber`, `Unstructured.io`   |
| **Page-Based Chunking**    | Chunks created per page or visual layout (often OCR-based)                      | Invoices, scanned documents, academic papers                                | `Tesseract OCR`, `PyMuPDF`, `pdf2image`, `layoutLM`, `Donut`                    |
| **Visual/Layout-Aware**    | Uses layout cues like headers, font, or boxes to define chunks                  | Magazines, academic PDFs, websites                                          | `layoutLMv3`, `Donut`, `PubLayNet`, `DocFormer`, `Unstructured.io`              |
| **Code/Function-Based**    | Chunks by logical programming units like functions or classes                   | Code summarization, documentation, Copilot tools                            | `tree-sitter`, `jedi`, `CodeBERT`, `PolyCoder`, `StarCoder`, `GPT-4-Code`       |


## 🔹 Best Practices for Chunking in RAG

**Optimal Chunk Size:**

  - **300–500 tokens** is often optimal (balances semantic completeness and context window limits).
  - **Too small:** loses context; Too large: hurts retrieval relevance.

**Use Overlap:**

  - Add **10–20% token overlap** (e.g., 100 tokens) to maintain context across chunk boundaries.

**Embed with CLS or Average Pooling:**

  - Use models like ```sentence-transformers, all-MiniLM, bge-base, text-embedding-ada-002```, etc., which are tuned for sentence-level semantics.

**Preprocessing:**

  - Clean HTML, remove noise, normalize whitespace.
  - Keep metadata like title, section headings, page number, etc.


## 🔹 Models for Chunking / Semantic Splitting

These help in semantic-aware chunking:

| **Model/Library**                   | **Purpose**                                     |
| ----------------------------------- | ----------------------------------------------- |
| **SpaCy**                           | Sentence segmentation                           |
| **NLTK**                            | Token/sentence splitting                        |
| **TextSplit**                       | Token-aware splitting                           |
| **LangChain RecursiveTextSplitter** | Recursive chunking by structure                 |
| **SemanticTextSplitter** (OpenAI)   | Uses embeddings to break at semantic boundaries |
| **Unstructured.io**                 | Parsing PDFs, emails, HTMLs semantically        |


## 🔹 Best Chunking Strategy by RAG Type

| **Use Case**         | **Best Chunking Mode**                         |
| -------------------- | ---------------------------------------------- |
| Q\&A over documents  | Sentence-based + sliding window + overlap      |
| Legal/medical docs   | Paragraph + heading-based + semantic splitting |
| Technical manuals    | Markdown/headings + semantic splitting         |
| Long PDFs            | RecursiveTextSplitter + metadata + overlap     |
| Chatbot (multi-turn) | Short sentence + sliding window for continuity |
| Enterprise search    | Paragraph-based with metadata indexing         |



## 🔹 Popular Embedding + Chunking Stack in RAG Pipelines

**LangChain:**
  - ```RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter```

**Haystack:**
  - ```PreProcessor(split_by="sentence", split_length=10, overlap=3)```

**LlamaIndex:**
  - ```SentenceSplitter, SemanticSplitterNodeParser```

## 🔹 Tools/Models Summary

| Tool / Lib        | Use For Chunking? | Notes                               |
| ----------------- | ----------------- | ----------------------------------- |
| `LangChain`       | ✅                 | Multiple chunking classes available |
| `Haystack`        | ✅                 | Flexible chunking + preprocessing   |
| `LlamaIndex`      | ✅                 | Node-based semantic chunking        |
| `SpaCy`, `NLTK`   | ✅                 | Sentence/word segmentation          |
| `unstructured.io` | ✅                 | Parsing HTML, PDFs, etc.            |
| `textsplit`       | ✅                 | Heuristics-based chunking           |


## ✅ Recommendation for Best RAG Chunking (General Purpose)

- **Chunk Size:** 350–500 tokens
- **Overlap:** 50–100 tokens
- **Method:** Recursive/semantic chunking with fallback to sentence-level
- **Tools:** LangChain RecursiveTextSplitter or LlamaIndex SemanticSplitterNodeParser


## ✅ 1. LangChain – RecursiveTextSplitter

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

texts = text_splitter.split_text(your_document_text)
print(f"Total chunks: {len(texts)}")
```

## ➡️ Best for: unstructured text (articles, reports, raw documents)


## ✅ 2. LangChain – MarkdownHeaderTextSplitter

For structured markdown documents:

```
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """# Title\nSome intro.\n## Section 1\nDetails here.\n## Section 2\nMore details."""
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Title"), ("##", "Section")])
docs = splitter.split_text(markdown_text)

for doc in docs:
    print(doc.metadata)  # includes Title, Section
    print(doc.page_content)
```

## ➡️ Best for: docs with clear heading structure (e.g., technical manuals, markdown)

## ✅ 3. LlamaIndex – SemanticSplitterNodeParser

```
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index import Document

# Load your document
documents = [Document(text=your_document_text)]

# Semantic-aware splitting
parser = SemanticSplitterNodeParser(
    embed_model=OpenAIEmbedding(),
    llm=OpenAI(),
    chunk_size=512
)

nodes = parser.get_nodes_from_documents(documents)

for node in nodes:
    print(node.text)
```

## ➡️ Best for: semantic coherence (topic-wise), LLM-assisted splitting

## ✅ 4. Haystack – PreProcessor

```
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    split_by="sentence",
    split_length=5,
    split_overlap=1,
    split_respect_sentence_boundary=True
)

# Load your raw text into a dict format
raw_docs = [{"content": your_document_text}]
processed_docs = preprocessor.process(raw_docs)

print(f"Total processed chunks: {len(processed_docs)}")
```

## ➡️ Best for: sentence-level RAG, Q&A over documents

## ✅ Bonus: unstructured for parsing raw files

```
from unstructured.partition.auto import partition

elements = partition(filename="sample.pdf")
text = "\n".join([el.text for el in elements if el.text is not None])
```

## ➡️ Use this before chunking, especially for PDFs, HTML, DOCX, etc.







