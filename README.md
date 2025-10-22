# 🧠 Agentic AI Research System

**Agentic AI Research System** là một công cụ Python tự động hóa quy trình **nghiên cứu khoa học** trong lĩnh vực *Agentic AI*.
Hệ thống có khả năng **thu thập, tóm tắt, phân tích xu hướng** và **đề xuất ý tưởng nghiên cứu** dựa trên các bài báo khoa học được lấy từ **ArXiv** và **Semantic Scholar**.

---

## 1️⃣ Giới thiệu

Dự án này mô phỏng một **AI Research Assistant** với khả năng hoạt động tự chủ trong quy trình nghiên cứu:

* Thu thập dữ liệu nghiên cứu mới nhất từ nhiều nguồn học thuật mở.
* Tự động tóm tắt nội dung bài báo khoa học.
* Phân tích xu hướng và trích xuất từ khóa nổi bật.
* Đề xuất ý tưởng nghiên cứu tiềm năng dựa trên các xu hướng mới.
* Sinh **báo cáo Markdown** để tổng hợp và trình bày kết quả một cách chuyên nghiệp.

---

## 2️⃣ Kiến trúc hệ thống

Hệ thống được xây dựng theo mô hình **đa tác nhân (Multi-Agent Architecture)**, trong đó mỗi *agent* phụ trách một vai trò độc lập trong pipeline R&D tự động:

| Agent                           | Vai trò chính                                                   |
| ------------------------------- | --------------------------------------------------------------- |
| **DataCollectorAgent**          | Thu thập papers từ ArXiv và Semantic Scholar                    |
| **SummarizerAgent**             | Tóm tắt văn bản bằng phương pháp extractive summarization       |
| **TrendAnalysisAgent**          | Phân tích xu hướng, trích xuất từ khóa và cụm từ quan trọng     |
| **IdeaGeneratorAgent**          | Tạo ý tưởng nghiên cứu mới dựa trên xu hướng và từ khóa nổi bật |
| **ReportGeneratorAgent**        | Sinh báo cáo tổng hợp (Markdown)                                |
| **AgenticResearchOrchestrator** | Điều phối toàn bộ pipeline từ thu thập đến báo cáo              |

### 🔧 Sơ đồ kiến trúc hệ thống

<img width="1180" height="785" alt="System Architecture" src="https://github.com/user-attachments/assets/af80b6a4-52a9-4b19-be60-5d4cf954385d" />

---

## 3️⃣ Cấu trúc thư mục

```
agentic_ai_research/
├── main.py              # Pipeline chính: thu thập, phân tích, sinh báo cáo
├── app.py               # Web app tương tác với người dùng (Streamlit UI)
├── data/                # Lưu papers đã thu thập (JSON)
├── reports/             # Báo cáo Markdown sinh tự động
└── requirements.txt     # Danh sách thư viện cần cài đặt
```

---

## 4️⃣ Cài đặt

### 🔹 Yêu cầu hệ thống

* Python ≥ 3.9
* Kết nối Internet (để truy cập ArXiv và Semantic Scholar API)

### 🔹 Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:

```bash
pip install arxiv requests streamlit
```

---

## 5️⃣ Cách sử dụng

### 🚀 Chạy pipeline nghiên cứu tự động

```bash
python main.py
```

**Cấu hình mặc định:**

* Từ khóa tìm kiếm: `"Agentic AI OR AI Agents OR Autonomous AI"`
* Thời gian: 30 ngày gần nhất
* Giới hạn: 50 papers / nguồn

**Pipeline thực hiện:**

1. Thu thập dữ liệu từ ArXiv và Semantic Scholar
2. Tóm tắt nội dung bài báo
3. Phân tích xu hướng nghiên cứu
4. Sinh ý tưởng nghiên cứu mới
5. Xuất báo cáo Markdown trong thư mục `reports/`

---

## 6️⃣ Web App (Tùy chọn)

Chạy ứng dụng web để tương tác và xem kết quả:

```bash
streamlit run app.py
```

Giao diện cho phép:

* Nhập từ khóa tìm kiếm mới
* Xem kết quả tóm tắt papers
* Theo dõi biểu đồ xu hướng và từ khóa nổi bật

---

## 7️⃣ Đầu ra

Ví dụ báo cáo tự động:

```
reports/research_report_20251022_153045.md
```

Báo cáo bao gồm:

* Tổng hợp số lượng bài báo theo nguồn
* Từ khóa và cụm từ nổi bật
* Phân tích chủ đề nghiên cứu chính
* Đề xuất hướng nghiên cứu mới
* Danh sách bài báo kèm tóm tắt

---

## 8️⃣ Hướng phát triển

* Tích hợp **LLM** để nâng cao chất lượng tóm tắt và sinh ý tưởng.
* Mở rộng **phân tích xu hướng theo từng lĩnh vực con** (ví dụ: Multi-Agent Systems, Cognitive AI).
* Tích hợp **dashboard phân tích trực quan** bằng Streamlit hoặc FastAPI.
* Thêm **bộ nhớ nghiên cứu liên tục** (Persistent Knowledge Base) để học theo thời gian.

---
