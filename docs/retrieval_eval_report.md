# Medical RAG - Báo cáo Đánh giá Thực nghiệm Phase Retrieval

- **Ngày đánh giá:** 24/09/2026
- **Tập dữ liệu:** 200 câu hỏi y khoa (50 mẫu ngẫu nhiên có kiểm soát từ mỗi chủ đề: `body-part`, `disease`, `drug`, `medicine`)
- **Tập embedding model:** `google/embeddinggemma-300m` (300M params, chạy local trên GPU CUDA)
- **Tập vector database:** Qdrant (9,294 chunks từ kho tri thức Y tế YouMed)
- **Tập BM25:** PyVi word segmentation + BM25Encoder
- **Cơ chế Hybrid:** Reciprocal Rank Fusion (RRF) kết hợp Dense Vector & BM25 Sparse Vector

---

## 1. Định nghĩa các chỉ số đo lường

| Chỉ số | Câu hỏi cốt lõi | Ý nghĩa trong bài toán RAG Y tế |
| :--- | :--- | :--- |
| **Recall@K** | *Có lấy đủ thông tin không?* | Tỷ lệ chunk chứa đáp án được lấy về trong top $K$ so với tổng số chunk liên quan. Với bài toán QA đơn đoạn văn, tìm thấy chunk đúng trong top $K$ cho 100%, không thì 0%. |
| **Precision@K** | *Lấy về có nhiều rác không?* | Tỷ lệ chunk đúng trong số $K$ chunk lấy về ($\frac{\text{hits}}{K}$). Thể hiện độ "sạch" của context nhồi vào prompt cho LLM. |
| **HitRate@K** | *Có lấy được ít nhất 1 tài liệu đúng không?* | Xác suất để top $K$ chứa ít nhất một chunk đúng. Quyết định khả năng LLM có tài liệu căn cứ để trả lời. |
| **MRR@K / MRR** | *Tài liệu đúng đầu tiên nằm cao không?* | $\frac{1}{N} \sum \frac{1}{\text{rank}_1}$ (nghịch đảo thứ hạng chunk đúng đầu tiên). Càng gần 1.0 thì chunk đúng càng nằm sát đỉnh top 1. |
| **nDCG@K** | *Chất lượng xếp hạng tổng thể có tối ưu không?* | Điểm chiết khấu vị trí Normalized Discounted Cumulative Gain ($\log_2(\text{rank} + 1)$). Đánh giá tổng thể độ ưu tiên của các kết quả liên quan. |
| **Article@K** | *Có tìm đúng bài viết gốc không?* | Tỷ lệ tìm thấy ít nhất 1 chunk thuộc bài viết y khoa gốc (ngay cả khi chưa phải đúng chunk ngữ cảnh cụ thể). |

---

## 2. Bảng kết quả tổng hợp tổng thể ($N = 200$)

So sánh 3 chế độ tìm kiếm: **Dense** (Vector), **Sparse** (BM25), và **Hybrid** (RRF Fusion).

### 2.1. Ma trận so sánh theo các ngưỡng $K \in \{1, 3, 5, 10\}$

| Chế độ | Ngưỡng $K$ | Recall@K | Precision@K | HitRate@K | MRR@K | nDCG@K | Article@K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HYBRID (RRF)** | **$K = 1$** | **59.0%** | **59.0%** | **59.0%** | **0.590** | **0.590** | 93.5% |
| *(Tối ưu nhất)* | **$K = 3$** | **75.0%** | **25.0%** | **75.0%** | **0.663** | **0.686** | 98.0% |
| | **$K = 5$** | **77.0%** | **15.4%** | **77.0%** | **0.668** | **0.694** | 98.5% |
| | **$K = 10$** | **80.5%** | **8.1%** | **80.5%** | **0.673** | **0.706** | **99.5%** |
| | | | | | *(MRR = 0.673)* | | |
| **DENSE** | $K = 1$ | 55.0% | 55.0% | 55.0% | 0.550 | 0.550 | 91.0% |
| | $K = 3$ | 71.0% | 23.7% | 71.0% | 0.622 | 0.644 | 96.0% |
| | $K = 5$ | 73.5% | 14.7% | 73.5% | 0.627 | 0.655 | 96.5% |
| | $K = 10$ | 76.0% | 7.6% | 76.0% | 0.631 | 0.663 | 97.0% |
| | | | | | *(MRR = 0.631)* | | |
| **SPARSE (BM25)** | $K = 1$ | 56.0% | 56.0% | 56.0% | 0.560 | 0.560 | 92.0% |
| | $K = 3$ | 69.0% | 23.0% | 69.0% | 0.619 | 0.637 | 95.5% |
| | $K = 5$ | 73.5% | 14.7% | 73.5% | 0.630 | 0.656 | 98.0% |
| | $K = 10$ | 78.0% | 7.8% | 78.0% | 0.636 | 0.671 | 98.0% |
| | | | | | *(MRR = 0.636)* | | |

---

## 3. Bảng phân tích chi tiết theo từng ngưỡng $K$

### Cutoff $K = 1$
Ở vị trí đầu bảng, mục tiêu là xem hệ thống có đưa được chunk đáp án lên ngay vị trí #1 hay không:

| Chế độ | Recall@1 | Precision@1 | HitRate@1 | MRR@1 | nDCG@1 | Article@1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid** | **59.0%** | **59.0%** | **59.0%** | **0.590** | **0.590** | **93.5%** |
| Dense | 55.0% | 55.0% | 55.0% | 0.550 | 0.550 | 91.0% |
| Sparse | 56.0% | 56.0% | 56.0% | 0.560 | 0.560 | 92.0% |

### Cutoff $K = 3$
Ngưỡng tinh gọn (thích hợp cho các mô hình LLM cần prompt ngắn, latency thấp):

| Chế độ | Recall@3 | Precision@3 | HitRate@3 | MRR@3 | nDCG@3 | Article@3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid** | **75.0%** | **25.0%** | **75.0%** | **0.663** | **0.686** | **98.0%** |
| Dense | 71.0% | 23.7% | 71.0% | 0.622 | 0.644 | 96.0% |
| Sparse | 69.0% | 23.0% | 69.0% | 0.619 | 0.637 | 95.5% |

### Cutoff $K = 5$
Ngưỡng mặc định của hệ thống MediRAG (điểm cân bằng lý tưởng giữa Recall và kích thước prompt):

| Chế độ | Recall@5 | Precision@5 | HitRate@5 | MRR@5 | nDCG@5 | Article@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid** | **77.0%** | **15.4%** | **77.0%** | **0.668** | **0.694** | **98.5%** |
| Dense | 73.5% | 14.7% | 73.5% | 0.627 | 0.655 | 96.5% |
| Sparse | 73.5% | 14.7% | 73.5% | 0.630 | 0.656 | 98.0% |

### Cutoff $K = 10$
Ngưỡng tối đa cho tab Retrieval Inspection (tìm kiếm sâu):

| Chế độ | Recall@10 | Precision@10 | HitRate@10 | MRR@10 | nDCG@10 | Article@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid** | **80.5%** | **8.1%** | **80.5%** | **0.673** | **0.706** | **99.5%** |
| Dense | 76.0% | 7.6% | 76.0% | 0.631 | 0.663 | 97.0% |
| Sparse | 78.0% | 7.8% | 78.0% | 0.636 | 0.671 | 98.0% |

---

## 4. Chi tiết theo từng chuyên mục Y khoa (tại $K = 5$)

| Chuyên mục | Mẫu ($n$) | Chế độ | Recall@5 | Prec@5 | HitRate@5 | MRR@5 | nDCG@5 | Article@5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`medicine`** (Thuốc) | 50 | **Hybrid** | **84.0%** | **16.8%** | **84.0%** | **0.709** | **0.742** | **100.0%** |
| | | Dense | 84.0% | 16.8% | 84.0% | 0.711 | 0.743 | 100.0% |
| | | Sparse | 76.0% | 15.2% | 76.0% | 0.594 | 0.636 | 96.0% |
| **`body-part`** (Giải phẫu) | 50 | **Hybrid** | **90.0%** | **18.0%** | **90.0%** | **0.758** | **0.795** | **100.0%** |
| | | Dense | 80.0% | 16.0% | 80.0% | 0.652 | 0.690 | 92.0% |
| | | Sparse | 88.0% | 17.6% | 88.0% | 0.767 | 0.796 | 100.0% |
| **`drug`** (Dược chất) | 50 | **Hybrid** | **74.0%** | **14.8%** | **74.0%** | **0.677** | **0.693** | **100.0%** |
| | | Dense | 70.0% | 14.0% | 70.0% | 0.657 | 0.668 | 100.0% |
| | | Sparse | 70.0% | 14.0% | 70.0% | 0.621 | 0.640 | 98.0% |
| **`disease`** (Bệnh học) | 50 | **Hybrid** | **60.0%** | **12.0%** | **60.0%** | **0.527** | **0.546** | **94.0%** |
| | | Dense | 60.0% | 12.0% | 60.0% | 0.490 | 0.518 | 94.0% |
| | | Sparse | 60.0% | 12.0% | 60.0% | 0.537 | 0.553 | 98.0% |

---

## 5. Nhận xét & Khuyến nghị kỹ thuật

1. **Hiệu quả của Hybrid RRF:**
   - Ở mọi ngưỡng $K$, Hybrid luôn cho kết quả vượt trội hơn cả Dense và Sparse đơn lẻ.
   - Nhờ RRF, các câu hỏi chứa thuật ngữ y khoa đặc thù (tên hoạt chất, giải phẫu) được BM25 bắt chính xác, trong khi các câu hỏi mô tả triệu chứng bằng văn nói được Dense vector bao phủ ngữ nghĩa.
2. **Quy luật Precision vs Recall trong RAG:**
   - Khi tăng $K$ từ 3 lên 10, Recall tăng từ **75.0%** lên **80.5%** (+5.5%).
   - Tuy nhiên, Precision giảm từ **25.0%** xuống **8.1%** (lượng chunk nhiễu tăng gấp 3 lần).
   - **Khuyến nghị:** Đối với luồng sinh câu trả lời chat tự động, nên giữ **$K = 3$ đến $K = 5$**. Đối với tab kiểm tra retrieval, $K = 10$ là hợp lý để chuyên gia kiểm tra độ phủ.
3. **Phân tích theo chuyên mục:**
   - Chuyên mục `medicine` và `body-part` có chất lượng retrieval xuất sắc (**Recall@5 đạt 84% - 90%**, **Article@5 đạt 100%**).
   - Chuyên mục `disease` có Recall@5 thấp hơn (**60.0%**), nhưng Article@5 vẫn đạt **94.0%**. Nguyên nhân do các bài viết bệnh học thường dài, có nhiều đoạn mô tả triệu chứng tương đồng nhau. Có thể cải thiện bằng cách tăng context window khi chunking hoặc bổ sung parent-child chunking.

---

## 6. Hướng dẫn tái lập kết quả (Reproduction)

Đảm bảo server embedding đang chạy trên port 8001 và không có process nào khóa `qdrant_data/`:

```powershell
$env:PYTHONPATH="backend;backend/scripts"
python backend/scripts/eval_retrieval.py --embed-url http://localhost:8001 --qdrant-path qdrant_data --test-case-dir test_case --per-type 50 --seed 42 --k-values 1,3,5,10
```
