# DanceLight - LED AI RAG Chatbot and attribute searching

## Project Description

This project builds an intelligent product information retrieval system that integrates two major features: **AI customer service chatbot** and **model attribute search**. The system employs RAG technology, combining Docling PDF parsing, semantic search, and OpenAI GPT models to accurately extract information from product catalogs and answer user questions in natural language.

This system solves the pain point of traditional product catalogs requiring manual queries, which are time-consuming and labor-intensive, enabling quick access to needed product specifications, model information, and technical details, significantly improving customer service efficiency and user experience.

### Core Technical Features

- **RAG Hybrid Retrieval Architecture**: Embedding model (BGE-M3) for initial filtering + Reranker model (BGE-Reranker-v2-m3) for precise ranking, ensuring retrieval accuracy
- **Intelligent Query Expansion**: Automatically expand query keywords using GPT-4o-mini to improve recall rate
- **Paginated PDF Parsing**: Use Docling to process large PDF catalogs page by page, supporting OCR and table structure recognition
- **Hybrid Device Optimization**: Embedding uses CPU (stable), Reranker uses MPS/GPU (accelerated), balancing performance and resources
- **Vector Caching Mechanism**: Pre-calculate and cache page embeddings, only need to encode query during search, improving response speed

## Getting Started

### Prerequisites

- **Python**: 3.8 or above
- **Operating System**: Windows / macOS / Linux
- **Hardware**: GPU recommended (CUDA or MPS support) to accelerate Reranker, but CPU can also run
- **OpenAI API Key**: Valid OpenAI API key required for RAG generation

### Installation Steps

1. **Clone the Project**

```bash
git clone https://github.com/yourusername/DanceLight--LED-AI-Chatbot-and-attribute-searching.git
cd DanceLight--LED-AI-Chatbot-and-attribute-searching
```

2. **Install Dependencies**

```bash
pip install PyQt5 torch sentence-transformers openai docling PyMuPDF tqdm numpy
```
or 

```bash
pip install -r requirements.txt
```

3. **Configure OpenAI API Key**

   Edit `.env` and replace with your actual API key:
```
   OPENAI_API_KEY=sk-your-api-key-here
```

4. **Prepare Product Catalog PDF**

**Please visit: https://www.dancelight.com.tw/tw/catalog to manually download, name the catalog "2025舞光LED21st(單頁水印可搜尋).pdf" and add it to the project root directory**

5. **Run the Program**

```bash
python homePage.py
```

After the system starts, the main page will be displayed, providing two entry points: "AI Customer Service" and "Model Search".

## File Structure

```
DanceLight--LED-AI-Chatbot-and-attribute-searching/
│
├── homePage.py                  # Main page - Application entry point, provides AI customer service and model search menu
├── ai_chat_page.py             # AI customer service page - Implements chatbot interface and RAG system integration
├── model_search_page.py        # Model search page - Provides product model search and result display functionality
├── docling_rag_v5.py           # RAG core engine - Contains PDF parsing, vector retrieval, Query expansion and generation logic
│
├── dancelight_logo.jpg         
├── 2025舞光LED21st(單頁水印可搜尋).pdf  # Product catalog data source (need to prepare yourself)
│
├── docling_cache/              # Cache directory - Stores parsed PDF page content and pre-calculated embeddings
│   └── parsed_data.pkl
│
├── temp_pages/                 
├── requirements.txt            
├── README.md                   
└── .gitignore                  
```

### Main File Descriptions

- **`homePage.py`**: Application entry point, creates main window and provides function menu buttons
- **`ai_chat_page.py`**: AI customer service chat interface, uses multi-threading (`RAGWorker`) to avoid UI freezing, supports typewriter animation for AI responses
- **`model_search_page.py`**: Model search page, provides search box and result card display (with animation effects)
- **`docling_rag_v5.py`**: System's core RAG engine, includes:
  - `PDFSplitter`: Splits large PDF into single-page files
  - `DoclingParser`: Uses Docling to parse PDF and extract Markdown format content
  - `QueryExpander`: Uses GPT-4o-mini to expand query keywords
  - `search_with_hybrid_device()`: Hybrid device retrieval function (Embedding CPU + Reranker MPS/GPU)
  - `RAG`: RAG class for generating final answers
  - `DoclingRAGSystem`: Main system class integrating all components

## Analysis

### RAG Retrieval Process

This system adopts a **two-stage hybrid retrieval strategy**, balancing accuracy and performance:

```
User Query
    ↓
1. Query Expansion (GPT-4o-mini)
    ├─ Original query: "bathroom light"
    └─ Expanded query: ["waterproof light", "IP65", "IP66", "bathroom", "moisture-proof"]
    ↓
2. Embedding Initial Filtering (BAAI/bge-m3 on CPU)
    ├─ Use pre-calculated page vectors (cached)
    ├─ Only need to encode query
    └─ Quickly filter top 50-80 candidates from hundreds of pages
    ↓
3. Reranker Precision Ranking (BAAI/bge-reranker-v2-m3 on MPS/GPU)
    ├─ Perform precise relevance scoring on candidate pages
    └─ Take top 25 pages as final context
    ↓
4. RAG Generation (OpenAI GPT-4o)
    ├─ Use retrieved page content as context
    ├─ Combine user question to generate natural language answer
    └─ Cite specific page numbers to improve credibility
    ↓
User Receives Answer
```

### Key Technical Decisions

1. **Hybrid Device Strategy**:
   - Embedding model runs on CPU: Avoids GPU out of memory (OOM) issues, and embedding is only calculated and cached on first run
   - Reranker model runs on MPS/GPU: Accelerates precision ranking process

2. **Vector Caching Mechanism**:
   - Calculate embeddings for all pages when first parsing PDF and serialize for storage
   - Subsequent queries only need to encode query vector and perform dot product with cached page vectors to complete initial filtering
   - Significantly reduces latency

3. **Paginated Processing**:
   - Use PyMuPDF to split large PDF into single-page files
   - Docling parses page by page, reducing single memory consumption
   - Supports OCR and table structure recognition, ensuring complete information extraction

4. **Query Expansion**:
   - Use small GPT model (gpt-4o-mini) to automatically generate related keywords
   - Expand query semantics to avoid missing important pages


## Results

### Retrieval Performance (Based on Actual Testing)

| Metric                     | Value                          |
|------------------------|-------------------------------|
| **Total Pages**              | Approximately 388 pages                       |
| **First Parse Time**         | Approximately 5-10 minutes (only needs to run once)     |
| **Cache Load Time**         | <5 seconds                         |
| **Total Query Time**          | Approximately 3-8 seconds (including Query expansion + retrieval + generation)|
| **Embedding Filtering Time**  | <1 second (using pre-calculated vectors)          |
| **Reranker Ranking Time**   | Approximately 2-4 seconds (processing 50-80 page candidates)   |
| **GPT Generation Time**        | Approximately 1-3 seconds (depending on answer length)      |


## Contributors

- **陳昊暄** - PM, responsible for AI RAG chatbot
- **陳為政** - Responsible for AI RAG chatbot
- **何承澔** - Frontend page integration
- **高毅峻** - Model attribute search

## Acknowledgments

- DanceLight: Provided product catalog data and business requirement guidance
- Professor 卞中佩: Provided technical guidance for the project

## References

### Data Sources

- Product Catalog: 2025舞光LED21st(單頁水印可搜尋).pdf (Source: https://www.dancelight.com.tw/tw/catalog)

### Analysis Methods and Tools

- Embedding Model:
  - [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 
  
- Reranker Model:
  - [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)   

- PDF Parsing:
  - [Docling](https://github.com/DS4SD/docling) 
  - [PyMuPDF](https://pymupdf.readthedocs.io/)
