# Medical RAG

Hệ thống Hỏi - Đáp Y khoa tiếng Việt (Vietnamese Medical Q&A) dựa trên kho tri thức y tế YouMed đã được crawl và xử lý phân đoạn ngữ nghĩa. Hệ thống kết hợp tìm kiếm lai đa phương thức (**Hybrid Retrieval** giữa Dense Vector và BM25 Sparse Vector với thuật toán **RRF Fusion** trong Qdrant), mô hình tái xếp hạng (**Reranking**), cùng tầng sinh sinh ngữ cảnh (**Grounding LLM**) với trích dẫn tài liệu tham khảo chính xác dạng `[n]`.

---

## Kiến trúc hệ thống (System Architecture)

```mermaid
flowchart TD
    subgraph UI["Frontend Layer (Next.js 15 / React 19)"]
        ChatUI["Chat Interface (/)<br/>• Quản lý lịch sử hội thoại (Today / Previous)<br/>• Trích dẫn tương tác [n] liên kết Source Card<br/>• Cảnh báo y tế (Medical Disclaimer)"]
        InspectUI["Retrieval Inspector (/retrieval)<br/>• Debug truy vấn theo mode: Hybrid / Dense / Sparse<br/>• Tùy chỉnh tham số K<br/>• Hiển thị trực quan điểm Retrieval Score & Rerank Score"]
    end

    subgraph API["Backend API Layer (FastAPI :8000)"]
        ChatEndpoint["POST /api/chat<br/>(E2E Question Answering)"]
        RetrieveEndpoint["POST /api/retrieve<br/>(Chunk Inspection & Scoring)"]
        HealthEndpoint["GET /api/health<br/>(Cluster Status & Point Count)"]
    end

    subgraph Retrieval["Hybrid Retrieval & Reranking Engine"]
        PyVi["Vietnamese NLP Segmentation<br/>(PyVi ViTokenizer)"]
        DenseEnc["Dense Embedding Service<br/>(google/embeddinggemma-300m :8001 / CUDA)"]
        SparseEnc["Sparse Vector Encoder<br/>(BM25Encoder)"]
        Qdrant[("Qdrant Vector Database<br/>• Dense Vectors (300D Cosine)<br/>• Sparse Vectors (BM25 Index)<br/>• Reciprocal Rank Fusion (RRF)")]
        Reranker["Cross-Encoder Reranker<br/>(ms-marco-MiniLM / Custom Server)"]
    end

    subgraph Generation["Generation Layer (Grounding & Multi-LLM Router)"]
        PromptBuilder["Medical Context Builder<br/>• Đánh số ngữ cảnh [1]..[K]<br/>• Chống suy diễn / Hallucination Guardrail<br/>• Khuyến cáo chuyên môn y tế"]
        LLMRouter["Multi-Provider LLM Router<br/>(backend/medical_rag/llm.py)"]
        subgraph Providers["Hỗ trợ đa nền tảng LLM"]
            vLLM["Custom vLLM / OpenAI-compatible<br/>(e.g., Qwen3 qua Cloudflare Tunnel / Local)"]
            Groq["Groq Cloud LPUs<br/>(e.g., qwen/qwen3.8-27b, LLaMA 3.1)"]
            OpenRouter["OpenRouter API<br/>(Gemini 2.5 Flash, LLaMA, etc.)"]
        end
    end

    subgraph IngestEval["Offline Processing & Evaluation"]
        Ingestion["Ingestion Pipeline<br/>Articles Markdown -> Section Chunking -> Indexing"]
        RetrievalEval["eval_retrieval.py<br/>Recall@K, HitRate@K, MRR@K, nDCG@K"]
        E2EEval["eval_e2e_rag.py<br/>Tạo tập 200 case benchmark cho LLM-as-a-judge (JSONL)"]
    end

    %% Luồng Chat
    ChatUI -->|1. Câu hỏi người dùng| ChatEndpoint
    InspectUI -->|1. Truy vấn kiểm tra| RetrieveEndpoint

    ChatEndpoint -->|2. Tách từ| PyVi
    RetrieveEndpoint -->|2. Tách từ| PyVi

    PyVi -->|3a. Text phân đoạn| DenseEnc
    PyVi -->|3b. Text phân đoạn| SparseEnc

    DenseEnc -->|4a. Dense Query Vector| Qdrant
    SparseEnc -->|4b. Sparse Term Vector| Qdrant

    Qdrant -->|5. Hợp nhất RRF Candidate Chunks| Reranker
    Reranker -->|6. Chunks kèm điểm xếp hạng| RetrieveEndpoint
    Reranker -->|6. Top-K Chunks phù hợp nhất| PromptBuilder

    PromptBuilder -->|7. Prompt chuẩn hóa + Context| LLMRouter
    LLMRouter --> vLLM
    LLMRouter --> Groq
    LLMRouter --> OpenRouter

    vLLM -->|8. Câu trả lời kèm trích dẫn [n]| ChatEndpoint
    Groq -->|8. Câu trả lời kèm trích dẫn [n]| ChatEndpoint
    OpenRouter -->|8. Câu trả lời kèm trích dẫn [n]| ChatEndpoint

    ChatEndpoint -->|9. Answer + Sources Panel| ChatUI
    RetrieveEndpoint -->|9. Ranked Chunks + Dual Scores| InspectUI

    Ingestion -.->|Nạp dữ liệu định kỳ| Qdrant
    Qdrant -.-> RetrievalEval
    ChatEndpoint -.-> E2EEval
```

---

## Cấu trúc thư mục (Repository Layout)

```
backend/
├── medical_rag/
│   ├── ingestion/       # Thuật toán phân đoạn Markdown ngữ nghĩa và nạp vào Qdrant
│   ├── encoders.py      # Client kết nối Embedding Server và BM25 Sparse Encoder
│   ├── store.py         # Khởi tạo và cấu hình Qdrant Collection (Dense + Sparse)
│   ├── retrieval.py     # Bộ tìm kiếm Hybrid (RRF) và tích hợp Reranker
│   ├── generation.py    # Xây dựng System Prompt y tế và đóng gói Context
│   ├── llm.py           # Multi-provider LLM Client (OpenAI-compatible, Groq, OpenRouter)
│   └── api.py           # FastAPI service phục vụ /api/chat, /api/retrieve, /api/health
├── scripts/
│   ├── eval_retrieval.py  # Đo lường IR metrics (Recall@K, HitRate@K, MRR, nDCG)
│   ├── eval_e2e_rag.py    # Đánh giá E2E 200 case và export JSONL cho LLM-as-a-judge
│   └── validate_ingest.py # Kiểm tra tính toàn vẹn của dữ liệu sau ingest
└── tests/                 # Unit test suite toàn diện (pytest)

frontend/
├── app/
│   ├── page.tsx         # Giao diện Chat chính
│   ├── retrieval/       # Giao diện Retrieval Inspector
│   └── globals.css      # Toàn bộ design token và styling
├── components/          # Header, Sidebar, MessageList, SourcesPanel, RetrievalApp...
├── hooks/useChat.ts     # State management và lưu lịch sử hội thoại LocalStorage
└── lib/                 # Typed API client, citation parsing, source helpers

serve_model/
├── embed_server.py      # Microservice FastAPI host mô hình google/embeddinggemma-300m
├── requirements.txt     # Phụ thuộc dành riêng cho môi trường GPU
└── serve_qwen3_kaggle.ipynb # Notebook host vLLM / Reranker trên Kaggle GPU

data/                    # Kho văn bản YouMed (disease, drug, medicine, body-part)
test_case/               # Bộ câu hỏi đánh giá thực nghiệm theo từng chuyên mục
eval_results/            # Kết quả xuất ra từ các script benchmark (JSONL, Markdown)
```

---

## Hướng dẫn cài đặt (Setup)

Chạy tất cả các lệnh từ **thư mục gốc của repository**:

```bash
# 1. Cài đặt Backend
pip install -e "backend[dev]"
cp .env.example .env        # Cập nhật các API key và cấu hình cần thiết
```

Embedding Server chạy trong môi trường riêng biệt (cần PyTorch tương thích CUDA):

```bash
cd serve_model
python -m venv .venv && .venv/Scripts/activate      # Trên Windows; Linux dùng .venv/bin/activate
pip install -r requirements.txt torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Hướng dẫn khởi chạy (Run)

1. **Khởi động Embedding Server** (`google/embeddinggemma-300m`, cổng 8001):
   ```bash
   cd serve_model && uvicorn embed_server:app --port 8001 --env-file ../.env
   ```

2. **Nạp dữ liệu vào Qdrant (Ingest)** *(chỉ chạy 1 lần hoặc khi thay đổi logic chunking)*:
   ```bash
   python -m medical_rag.ingestion.ingest --embed-url http://localhost:8001
   ```

3. **Khởi động Backend API** (FastAPI, cổng 8000):
   ```bash
   uvicorn medical_rag.api:app --env-file .env --port 8000 --app-dir backend
   ```

4. **Khởi động Frontend** (Next.js, mở http://localhost:3000):
   ```bash
   cd frontend && npm install && npm run dev        # Production: npm run build && npm start
   ```

> [!NOTE]
> Qdrant lưu trữ dạng file local tại `qdrant_data/` và khóa độc quyền tiến trình. Hãy dừng tiến trình API trước khi chạy script Ingest hoặc chạy script đánh giá độc lập.

---

## Chi tiết API (Endpoints)

| Phương thức | Đường dẫn | Tham số chính | Mô tả |
|---|---|---|---|
| `POST` | `/api/chat` | `{"question": "..."}` | Trả về câu trả lời y tế đã trích dẫn kèm danh sách các chunk nguồn `sources: [{n, title, url, section_path, type, text, score}]`. |
| `POST` | `/api/retrieve` | `{"question": "...", "mode": "hybrid\|dense\|sparse", "k": 10}` | Trả về top-K chunk đã rank cùng 2 cột điểm riêng biệt: `retrieval_score` (Vector/BM25/RRF) và `rerank_score`. |
| `GET` | `/api/health` | Không | Kiểm tra trạng thái hệ thống và số lượng chunk đang được lập chỉ mục trong Qdrant. |

---

## Cấu hình môi trường (.env Configuration)

| Biến môi trường | Bắt buộc | Mặc định | Ý nghĩa |
|---|:---:|---|---|
| `EMBED_URL` | Có | `http://localhost:8001` | URL dịch vụ tạo embedding dense vector. |
| `HF_TOKEN` | Cho Embed | - | Token HuggingFace để tải mô hình `google/embeddinggemma-300m`. |
| `LLM_BASE_URL` | Tùy chọn | - | Endpoint OpenAI-compatible tùy chỉnh (ví dụ: vLLM qua Cloudflare Tunnel). |
| `LLM_MODEL` | Tùy chọn | `google/gemini-2.5-flash` | Tên model LLM sinh câu trả lời (ví dụ: `Qwen/Qwen3-4B-Instruct-2507`, `qwen/qwen3.8-27b`). |
| `GROQ_API_KEY` | Tùy chọn | - | Khóa API Groq Cloud (chạy cực nhanh với kiến trúc LPU). |
| `OPENROUTER_API_KEY` | Tùy chọn | - | Khóa API OpenRouter khi sử dụng Gemini hoặc các mô hình mã nguồn mở. |
| `QDRANT_PATH` | Không | `qdrant_data` | Thư mục lưu trữ cơ sở dữ liệu vector Qdrant. |
| `RERANK_URL` | Không | - | URL dịch vụ reranker cross-encoder. Nếu không đặt, giữ nguyên thứ tự xếp hạng RRF. |
| `API_URL` | Không | `http://localhost:8000` | URL Backend API mà Next.js proxy tới. |

---

## Đánh giá và Kiểm thử (Evaluation & Testing)

```bash
# 1. Chạy offline test suite (Unit test & Integration test)
cd backend && python -m pytest -q

# 2. Đánh giá chuyên sâu Retrieval Phase (đo Recall@K, Precision@K, HitRate@K, MRR@K, nDCG@K với K in {1, 3, 5, 10})
python backend/scripts/eval_retrieval.py --per-type 50 --k-values 1,3,5,10

# 3. Chạy End-to-End RAG trên 200 case và xuất dataset JSONL phục vụ LLM-as-a-judge
python backend/scripts/eval_e2e_rag.py --per-type 50 --seed 42 --k 5 --mode hybrid --output eval_results/rag_e2e_200_cases.jsonl
```
