import os
import json
import pickle
import hashlib
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import fitz
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from dotenv import load_dotenv # Import the tool
load_dotenv()                  # Trigger it to read your .env file

@dataclass
class Config:
    pdf_path: str = "2025舞光LED21st(單頁水印可搜尋).pdf"
    cache_dir: str = "docling_cache"
    temp_dir: str = "temp_pages"
    force_reparse: bool = False
    
    # Docling
    enable_ocr: bool = True
    enable_table_structure: bool = True
    
    # Search
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_candidates: int = 40
    final_pages: int = 20
    reranker_batch_size: int = 8
    
    # Query expansion
    enable_query_expansion: bool = True
    expansion_model: str = "gpt-4o-mini"
    max_keywords: int = 5
    
    # Generation
    generation_model: str = "gpt-4o"
    max_tokens: int = 10000
    temperature: float = 0.1


@dataclass
class Page:
    page_no: int
    content: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(page_no=d["page_no"], content=d["content"])


class Models:
    """Lazy-loaded singleton for ML models."""
    _instance = None
    
    def __new__(cls, config: Config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models(config)
        return cls._instance
    
    def _init_models(self, config: Config):
        self.reranker_device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Devices: Embedding(cpu) / Reranker({self.reranker_device})")
        
        self.embedder = SentenceTransformer(config.embedding_model, device="cpu")
        self.reranker = CrossEncoder(config.reranker_model, device=self.reranker_device)
        
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=True,
            batch_size=16, convert_to_numpy=True
        )
    
    def rerank(self, query: str, docs: list[str], batch_size: int) -> np.ndarray:
        if self.reranker_device == "mps":
            torch.mps.empty_cache()
        
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True)
        
        if self.reranker_device == "mps":
            torch.mps.empty_cache()
        
        return np.array(scores)


class PDFParser:
    def __init__(self, config: Config, models: Models):
        self.config = config
        self.models = models
        self.cache_file = Path(config.cache_dir) / "parsed_data.pkl"
        Path(config.cache_dir).mkdir(exist_ok=True)
        self._converter = None
    
    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            opts = PdfPipelineOptions()
            opts.do_ocr = self.config.enable_ocr
            opts.do_table_structure = self.config.enable_table_structure
            if self.config.enable_table_structure:
                opts.table_structure_options.do_cell_matching = True
            
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return self._converter
    
    def parse(self) -> tuple[list[Page], np.ndarray]:
        if self._cache_valid():
            return self._load_cache()
        
        pages = self._extract_pages()
        embeddings = self.models.embed([p.content for p in pages])
        self._save_cache(pages, embeddings)
        return pages, embeddings
    
    def _cache_valid(self) -> bool:
        if self.config.force_reparse or not self.cache_file.exists():
            return False
        try:
            with open(self.cache_file, "rb") as f:
                cached = pickle.load(f)
            return cached.get("pdf_hash") == self._pdf_hash()
        except Exception:
            return False
    
    def _pdf_hash(self) -> str:
        with open(self.config.pdf_path, "rb") as f:
            return hashlib.md5(f.read(1024 * 1024)).hexdigest()
    
    def _extract_pages(self) -> list[Page]:
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        try:
            pdf = fitz.open(self.config.pdf_path)
            pages = []
            
            for i in tqdm(range(len(pdf)), desc="Parsing pages"):
                page_path = temp_dir / f"page_{i+1:04d}.pdf"
                
                # Split single page
                single = fitz.open()
                single.insert_pdf(pdf, from_page=i, to_page=i)
                single.save(str(page_path))
                single.close()
                
                # Extract with Docling
                result = self.converter.convert(str(page_path))
                md = result.document.export_to_markdown().strip()
                if md:
                    pages.append(Page(page_no=i + 1, content=md))
            
            pdf.close()
            return pages
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _save_cache(self, pages: list[Page], embeddings: np.ndarray):
        data = {
            "pdf_hash": self._pdf_hash(),
            "parsed_at": datetime.now().isoformat(),
            "pages": [p.to_dict() for p in pages],
            "embeddings": embeddings,
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached {len(pages)} pages")
    
    def _load_cache(self) -> tuple[list[Page], np.ndarray]:
        with open(self.cache_file, "rb") as f:
            data = pickle.load(f)
        pages = [Page.from_dict(p) for p in data["pages"]]
        print(f"Loaded {len(pages)} pages from cache")
        return pages, data["embeddings"]


class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.models = Models(config)
        self.parser = PDFParser(config, self.models)
        self.pages: list[Page] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def initialize(self):
        print("Initializing...")
        self.pages, self.embeddings = self.parser.parse()
        print(f"Ready: {len(self.pages)} pages indexed")
    
    def query(self, question: str) -> dict:
        # Expand query
        expanded = self._expand_query(question) if self.config.enable_query_expansion else question
        
        # Two-stage retrieval
        candidates = self._embedding_filter(expanded)
        ranked = self._rerank(expanded, candidates)
        
        # Generate
        return self._generate(question, ranked)
    
    def _expand_query(self, query: str) -> str:
        try:
            resp = self.models.client.chat.completions.create(
                model=self.config.expansion_model,
                messages=[
                    {"role": "system", "content": f"提供最多{self.config.max_keywords}個相關關鍵詞，JSON格式：{{\"keywords\": []}}"},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            kw = json.loads(resp.choices[0].message.content).get("keywords", [])
            return f"{query} {' '.join(kw)}" if kw else query
        except Exception:
            return query
    
    def _embedding_filter(self, query: str) -> list[Page]:
        if len(self.pages) <= self.config.embedding_candidates:
            return self.pages
        
        q_emb = self.models.embed([query])[0]
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[: self.config.embedding_candidates]
        return [self.pages[i] for i in top_idx]
    
    def _rerank(self, query: str, candidates: list[Page]) -> list[tuple[Page, float]]:
        scores = self.models.rerank(query, [p.content for p in candidates], self.config.reranker_batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.config.final_pages]
    
    def _generate(self, question: str, pages: list[tuple[Page, float]]) -> dict:
        if not pages:
            return {"answer": "未找到相關內容", "pages": []}
        
        context = "\n\n".join(
            f"【Page {p.page_no}】(score: {s:.3f})\n{p.content}"
            for p, s in pages
        )
        
        resp = self.models.client.chat.completions.create(
            model=self.config.generation_model,
            messages=[
                {"role": "system", "content": "根據型錄內容詳細統整所有產品及規格，引用頁碼。"},
                {"role": "user", "content": f"問題：{question}\n\n型錄內容：\n{context}"},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return {
            "answer": resp.choices[0].message.content,
            "pages": [p.page_no for p, _ in pages],
            "tokens": resp.usage.total_tokens,
        }


def main():
    config = Config()
    system = RAGSystem(config)
    system.initialize()
    
    print("\n輸入 'q' 退出\n")
    while True:
        q = input("Question > ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        
        result = system.query(q)
        print(f"\n{result['answer']}")
        print(f"\n[Pages: {result['pages']}, Tokens: {result['tokens']}]\n")


if __name__ == "__main__":
    main()
