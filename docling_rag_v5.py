"""
Docling PDF RAG 系統 - 混合設備版本
Embedding 用 CPU（穩定）+ Reranker 用 MPS（加速）
最佳配置：避免 OOM + 保持速度
"""

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import gc

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ========== 配置 ==========
@dataclass
class DoclingRAGConfig:
    """配置"""
    pdf_path: str = "2025舞光LED21st(單頁水印可搜尋).pdf"
    
    # 分頁設置
    temp_dir: str = "temp_pages"
    keep_temp_files: bool = False
    
    # Docling OCR 設置
    enable_ocr: bool = True
    enable_table_structure: bool = True
    
    # 快取設置
    cache_dir: str = "docling_cache"
    force_reparse: bool = False
    
    # Embedding 初篩設置
    enable_embedding_filter: bool = True
    embedding_model: str = "BAAI/bge-m3"
    max_embedding_candidates: int = 50
    
    # Rerank 設置
    max_final_pages: int = 25
    reranker_batch_size: int = 8  # MPS batch size
    show_top_scores: int = 30
    
    # Query 擴展
    enable_query_expansion: bool = True
    query_expansion_model: str = "gpt-4o-mini"
    max_keywords: int = 5
    
    # RAG 設置
    openai_model: str = "gpt-4o"
    max_response_tokens: int = 10000
    temperature: float = 0.1

# ========== OpenAI ==========
try:
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
except Exception as e:
    raise RuntimeError(f"OpenAI 初始化失敗：{e}")

# ========== 模型 ==========
# 混合設備模式：Embedding CPU（避免 OOM）+ Reranker MPS（加速）
embedding_device = "cpu"
reranker_device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\n✓ 混合設備模式：")
print(f"  - Embedding: CPU")
print(f"  - Reranker: {reranker_device.upper()}（加速）")

print("\n載入模型...")
# Embedding 模型 - 用 CPU 避免 OOM
embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("✓ Embedding 模型載入完成 (BAAI/bge-m3 on CPU)")

# Reranker 模型 - 用 MPS 加速
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=reranker_device)
print(f"✓ Reranker 模型載入完成 (BAAI/bge-reranker-v2-m3 on {reranker_device.upper()})")

# 清理顯存
if reranker_device == "mps":
    torch.mps.empty_cache()

# ========== 頁面結構 ==========
@dataclass
class PageContent:
    """頁面內容"""
    page_no: int
    content: str
    text_length: int
    source_file: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

# ========== PDF 分頁工具 ==========
class PDFSplitter:
    """PDF 分頁工具"""
    
    def __init__(self, temp_dir: str = "temp_pages"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def split_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """將 PDF 切成單頁文件"""
        
        print(f"\n{'='*80}")
        print(f"分割 PDF：{pdf_path}")
        print(f"{'='*80}")
        
        pdf = fitz.open(pdf_path)
        total_pages = len(pdf)
        
        print(f"總頁數：{total_pages}")
        
        page_files = []
        
        for page_num in tqdm(range(total_pages), desc="分頁中"):
            page_no = page_num + 1
            
            single_page_pdf = fitz.open()
            single_page_pdf.insert_pdf(pdf, from_page=page_num, to_page=page_num)
            
            page_file = self.temp_dir / f"page_{page_no:04d}.pdf"
            single_page_pdf.save(str(page_file))
            single_page_pdf.close()
            
            page_files.append((page_no, str(page_file)))
        
        pdf.close()
        
        print(f"分頁完成！文件保存在：{self.temp_dir}")
        
        return page_files
    
    def cleanup(self):
        """清理臨時文件"""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"✓ 已清理臨時文件：{self.temp_dir}")

# ========== Docling 解析器 ==========
class DoclingParser:
    """Docling PDF 解析器（分頁）"""
    
    def __init__(self, config: DoclingRAGConfig):
        self.config = config
        self.cache_file = os.path.join(config.cache_dir, "parsed_data.pkl")
        os.makedirs(config.cache_dir, exist_ok=True)
        
        self.splitter = PDFSplitter(config.temp_dir)
        self._init_converter()
    
    def _init_converter(self):
        """初始化 Docling 轉換器"""
        print("初始化 Docling...")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        pipeline_options.do_table_structure = self.config.enable_table_structure
        
        if self.config.enable_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        print(f"Docling 已初始化（OCR: {self.config.enable_ocr}）")
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        """計算 PDF hash"""
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read(1024 * 1024)).hexdigest()
    
    def should_reparse(self, pdf_path: str) -> bool:
        """判斷是否需要重新解析"""
        if self.config.force_reparse:
            return True
        if not os.path.exists(self.cache_file):
            return True
        
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data.get('pdf_hash') != self.get_pdf_hash(pdf_path)
        except:
            return True
    
    def parse_pdf(self, pdf_path: str) -> Tuple[List[PageContent], Optional[np.ndarray]]:
        """解析 PDF（分頁處理）+ 計算 embeddings"""
        
        if not self.should_reparse(pdf_path):
            print("載入快取...")
            return self.load_cache()
        
        print(f"\n{'='*80}")
        print(f"解析 PDF（分頁處理模式）")
        print(f"{'='*80}")
        
        start = time.time()
        
        page_files = self.splitter.split_pdf(pdf_path)
        
        print(f"\n處理 {len(page_files)} 頁...")
        pages_content = self._process_pages(page_files)
        
        if not self.config.keep_temp_files:
            try:
                self.splitter.cleanup()
            except Exception as e:
                print(f"暫存資料夾刪除失敗（不影響你的資料）：{e}")
        
        print(f"\n✓ 解析完成！耗時：{time.time() - start:.2f} 秒")
        self._print_stats(pages_content)
        
        # 計算 embeddings
        print(f"\n計算頁面 embeddings（首次建立，之後會快取）...")
        embeddings = self._compute_embeddings(pages_content)
        
        # 保存快取（含 embeddings）
        self._save_cache(pages_content, pdf_path, embeddings)
        
        return pages_content, embeddings
    
    def _compute_embeddings(self, pages: List[PageContent]) -> np.ndarray:
        """計算所有頁面的 embeddings"""
        page_texts = [page.content for page in pages]
        
        embeddings = embedding_model.encode(
            page_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16,
            convert_to_numpy=True
        )
        
        print(f"Embeddings完成")
        return embeddings
    
    def _process_pages(self, page_files: List[Tuple[int, str]]) -> List[PageContent]:
        """處理所有頁面"""
        
        pages = []
        
        for page_no, file_path in tqdm(page_files, desc="Docling 處理"):
            if page_no % 10 == 0:
                gc.collect()
            try:
                content = self._process_single_page(page_no, file_path)
                if content:
                    pages.append(content)
            except Exception as e:
                print(f"\n頁面 {page_no} 處理失敗：{e}")
        
        return pages
    
    def _process_single_page(self, page_no: int, file_path: str) -> Optional[PageContent]:
        """處理單頁"""
        
        result = self.converter.convert(file_path)
        document = result.document
        
        markdown = document.export_to_markdown()
        
        if not markdown.strip():
            return None
        
        return PageContent(
            page_no=page_no,
            content=markdown.strip(),
            text_length=len(markdown.strip()),
            source_file=file_path
        )
    
    def _save_cache(self, pages: List[PageContent], pdf_path: str, embeddings: Optional[np.ndarray] = None):
        """保存快取（包含 embeddings）"""
        print(f"\n保存快取到：{self.cache_file}")
        
        data = {
            'pdf_path': pdf_path,
            'pdf_hash': self.get_pdf_hash(pdf_path),
            'parsed_at': datetime.now().isoformat(),
            'pages': [p.to_dict() for p in pages],
            'embeddings': embeddings
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        print("快取已保存（含 embeddings）")
    
    def load_cache(self) -> Tuple[List[PageContent], Optional[np.ndarray]]:
        """載入快取（包含 embeddings）"""
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        
        pages = [PageContent.from_dict(p) for p in data['pages']]
        embeddings = data.get('embeddings', None)
        
        print(f"載入 {len(pages)} 頁")
        if embeddings is not None:
            print(f"載入預計算的 embeddings")
        self._print_stats(pages)
        
        return pages, embeddings
    
    def _print_stats(self, pages: List[PageContent]):
        """打印統計"""
        if not pages:
            print("沒有頁面")
            return
        
        total_chars = sum(p.text_length for p in pages)
        avg_chars = total_chars / len(pages)
        
        page_nos = sorted([p.page_no for p in pages])
        missing_pages = []
        if page_nos:
            for i in range(page_nos[0], page_nos[-1] + 1):
                if i not in page_nos:
                    missing_pages.append(i)
        
        print(f"\n統計：")
        print(f"總頁數：{len(pages)}")
        print(f"頁碼範圍：{page_nos[0]} ~ {page_nos[-1]}")
        if missing_pages:
            print(f"缺失頁：{missing_pages}")
        else:
            print(f"頁碼連續完整")
        print(f"總字數：{total_chars:,}")
        print(f"平均每頁：{avg_chars:.0f} 字")

# ========== Query 擴展 ==========
class QueryExpander:
    """Query 擴展"""
    
    def __init__(self, config: DoclingRAGConfig):
        self.config = config
    
    def expand_query(self, query: str) -> Dict:
        """擴展查詢"""
        
        if not self.config.enable_query_expansion:
            return {
                "original_query": query,
                "keywords": [],
                "expanded_query": query
            }
        
        print(f"\n擴展查詢：「{query}」")
        
        try:
            response = client.chat.completions.create(
                model=self.config.query_expansion_model,
                messages=[
                    {"role": "system", "content": "你是燈品專家，請根據query，提供可能的高度相關關鍵詞（JSON格式）：{\"keywords\": [\"詞1\", \"詞2\"]}"},
                    {"role": "user", "content": f"查詢：{query}\n提供最多{self.config.max_keywords}個關鍵詞"}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            keywords = result.get("keywords", [])
            expanded = f"{query} {' '.join(keywords)}" if keywords else query
            
            print(f"關鍵詞：{', '.join(keywords)}")
            
            return {
                "original_query": query,
                "keywords": keywords,
                "expanded_query": expanded
            }
        except Exception as e:
            print(f"擴展失敗：{e}")
            return {
                "original_query": query,
                "keywords": [],
                "expanded_query": query
            }

# ========== 混合設備檢索（使用預計算 embeddings）==========
def search_with_hybrid_device(
    query: str,
    pages: List[PageContent],
    page_embeddings: np.ndarray,  # 新增：使用預計算的 embeddings
    config: DoclingRAGConfig
) -> List[Tuple[PageContent, float]]:
    """
    混合設備檢索：
    1. Embedding 初篩（使用預計算的 embeddings，超快！）
    2. Reranker MPS 精排（快速）
    """
    
    print(f"\n混合設備檢索...")
    print(f"總頁數：{len(pages)}")
    
    start = time.time()
    
    # ========== 階段 1: Embedding 初篩（使用預計算 embeddings）==========
    if config.enable_embedding_filter and len(pages) > config.max_embedding_candidates:
        print(f"\n階段 1: Embedding 初篩（使用快取，前 {config.max_embedding_candidates} 頁）")
        
        # 只需要編碼 query（1 秒內！）
        print(f"編碼查詢...")
        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        
        # 使用預計算的頁面 embeddings 計算相似度
        print(f"使用預計算的頁面 embeddings 計算相似度...")
        similarities = page_embeddings @ query_embedding
        
        # 取前 N 個候選
        top_indices = np.argsort(-similarities)[:config.max_embedding_candidates]
        candidate_pages = [pages[i] for i in top_indices]
        
        print(f"篩選出 {len(candidate_pages)} 個候選頁面")
        print(f"Embedding Top 10 頁碼：{[pages[i].page_no for i in top_indices[:10]]}")
        
    else:
        candidate_pages = pages
        print(f"跳過 embedding 初篩（頁數較少）")
    
    # ========== 階段 2: Reranker MPS 精排 ==========
    print(f"\n階段 2: Reranker 精排 ({reranker_device.upper()})（{len(candidate_pages)} 頁）")
    
    # 清理 MPS 顯存
    if reranker_device == "mps":
        torch.mps.empty_cache()
    
    # 構建 query-page 對
    pairs = [(query, page.content) for page in candidate_pages]
    
    # Reranker 打分
    print(f"計算精確相關性分數...")
    scores = reranker.predict(
        pairs,
        batch_size=config.reranker_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # 清理 MPS 顯存
    if reranker_device == "mps":
        torch.mps.empty_cache()
    
    # 排序
    page_scores = list(zip(candidate_pages, scores))
    page_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 取前 N 個
    top_pages = page_scores[:config.max_final_pages]
    
    elapsed = time.time() - start
    
    # 顯示統計
    print(f"\n檢索完成！總耗時：{elapsed:.2f} 秒")
    print(f"速度：{len(pages) / elapsed:.1f} 頁/秒")
    print(f"最高分：{page_scores[0][1]:.4f}")
    print(f"最低分：{page_scores[-1][1]:.4f}")
    print(f"平均分：{np.mean(scores):.4f}")
    
    # 顯示前幾名
    print(f"\nTop {min(config.show_top_scores, len(top_pages))} 頁面：")
    for i, (page, score) in enumerate(top_pages[:config.show_top_scores], 1):
        print(f"{i:2d}. 第 {page.page_no:3d} 頁 - {score:.4f}")
    
    return top_pages

# ========== RAG ==========
class RAG:
    """RAG 生成器"""
    
    def __init__(self, config: DoclingRAGConfig):
        self.config = config
    
    def generate(
        self,
        query: str,
        relevant_pages: List[Tuple[PageContent, float]],
        query_info: Optional[Dict] = None
    ) -> Dict:
        """生成回答"""
        
        if not relevant_pages:
            return {
                "answer": "抱歉，找不到相關內容。",
                "pages_used": [],
                "tokens_used": 0
            }
        
        print(f"\n生成回答...")
        
        # 構建 context
        context = "\n\n".join([
            f"【參考資料 {i}】（PDF 第 {page.page_no} 頁，相關度：{score:.4f}）\n{page.content}"
            for i, (page, score) in enumerate(relevant_pages, 1)
        ])
        
        # Prompt
        messages = [
            {"role": "system", "content": "請把我提供給你的型錄內容內的所有產品詳細的統整出來，不能遺漏。回答時引用頁碼。"},
            {"role": "user", "content": f"問題：{query}\n\n型錄內容：\n{context}\n\n請回答："}
        ]
        
        try:
            start = time.time()
            
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=self.config.max_response_tokens,
                temperature=self.config.temperature
            )
            
            return {
                "answer": response.choices[0].message.content,
                "pages_used": [{"page_no": p.page_no, "score": f"{s:.4f}"} for p, s in relevant_pages],
                "tokens_used": response.usage.total_tokens,
                "time": f"{time.time() - start:.2f}秒",
                "query_info": query_info
            }
        except Exception as e:
            return {
                "answer": f"錯誤：{e}",
                "pages_used": [],
                "tokens_used": 0,
                "error": str(e)
            }

# ========== 主系統 ==========
class DoclingRAGSystem:
    """Docling RAG 系統（混合設備版 + 預計算 embeddings）"""
    
    def __init__(self, config: DoclingRAGConfig):
        self.config = config
        self.parser = DoclingParser(config)
        self.expander = QueryExpander(config)
        self.rag = RAG(config)
        
        self.pages = None
        self.page_embeddings = None  
        self.history = []
    
    def initialize(self):
        """初始化"""
        print("="*80)
        print("初始化 Docling RAG 系統（混合設備版 + 預計算 embeddings）")
        print("="*80)
        
        # 解析 PDF 並計算/載入 embeddings
        self.pages, self.page_embeddings = self.parser.parse_pdf(self.config.pdf_path)
        
        # 檢查是否需要計算 embeddings
        if self.page_embeddings is None:
            print(f"\n偵測到舊版快取，正在計算 embeddings...")
            self.page_embeddings = self.parser._compute_embeddings(self.pages)
            
            # 更新快取
            print(f"更新快取（新增 embeddings）...")
            self.parser._save_cache(
                self.pages, 
                self.config.pdf_path, 
                self.page_embeddings
            )
            print(f"快取已更新")
        
        print(f"\n系統就緒！")
        print(f"模式：Embedding (CPU) + Reranker ({reranker_device.upper()})")
        print(f"Embedding：BAAI/bge-m3 on CPU（預計算並快取）")
        print(f"Reranker：BAAI/bge-reranker-v2-m3 on {reranker_device.upper()}（快速）")
        print(f"總頁數：{len(self.pages)}")
        print(f"Embeddings：已預計算並快取（查詢時只需編碼 query，<1秒）")
        if self.config.enable_embedding_filter:
            print(f"檢索策略：{len(self.pages)} 頁 → Embedding 快速篩選 {self.config.max_embedding_candidates} 頁 → Reranker {reranker_device.upper()} 精排 {self.config.max_final_pages} 頁")
    
    def query(self, question: str) -> Dict:
        """查詢"""
        
        print("\n" + "="*80)
        print(f"{question}")
        print("="*80)
        
        start = time.time()
        
        # Query 擴展
        query_info = self.expander.expand_query(question)
        
        # 混合設備檢索（使用預計算的 embeddings）
        relevant = search_with_hybrid_device(
            query_info["expanded_query"],
            self.pages,
            self.page_embeddings,  # 傳入預計算的 embeddings
            self.config
        )
        
        # 生成
        result = self.rag.generate(question, relevant, query_info)
        result["total_time"] = f"{time.time() - start:.2f}秒"
        
        self.history.append(result)
        
        return result

# ========== 主程式 ==========
def main():
    """主程式"""
    
    config = DoclingRAGConfig(
        pdf_path="2025舞光LED21st(單頁水印可搜尋).pdf",
        
        # Docling 設置
        enable_ocr=True,
        enable_table_structure=True,
        keep_temp_files=False,
        
        # Query 擴展
        enable_query_expansion=True,
        
        # Embedding 初篩 (CPU)
        enable_embedding_filter=True,
        embedding_model="BAAI/bge-m3",
        max_embedding_candidates=80,
        
        # Reranker 精排 (MPS)
        max_final_pages=25,
        reranker_batch_size=8,  # MPS batch size
        show_top_scores=30
    )
    
    system = DoclingRAGSystem(config)
    system.initialize()
    
    print("\n輸入問題（'q' 離開）")
    print("-"*80)
    
    while True:
        question = input("\n問題 > ").strip()
        
        if question.lower() in ['q', 'quit', 'exit']:
            break
        
        if not question:
            continue
        
        result = system.query(question)
        
        print("\n" + "="*80)
        print("回答")
        print("="*80)
        print(f"\n{result['answer']}")
        
        print(f"\n統計：")
        print(f"使用頁面：{[p['page_no'] for p in result['pages_used'][:10]]}")
        print(f"Token：{result['tokens_used']:,}")
        print(f"時間：{result['total_time']}")

if __name__ == "__main__":
    main()