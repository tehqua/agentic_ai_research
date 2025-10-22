# 🧠 Agentic AI Research System

**Agentic AI Research System** là một pipeline tự động hóa quy trình nghiên cứu và tổng hợp xu hướng trong lĩnh vực **Agentic AI**.  
Hệ thống này hoạt động như một "trợ lý nghiên cứu" — tự động thu thập bài báo mới, tóm tắt nội dung, phân tích xu hướng, gợi ý ý tưởng nghiên cứu và tạo báo cáo hoàn chỉnh ở định dạng Markdown.

---

## 🚀 Tính năng chính

| Module | Mô tả |
|--------|-------|
| **DataCollectorAgent** | Thu thập dữ liệu nghiên cứu từ ArXiv và Semantic Scholar API. |
| **SummarizerAgent** | Tóm tắt văn bản khoa học bằng phương pháp extractive summarization. |
| **TrendAnalysisAgent** | Phân tích xu hướng từ khóa, cụm từ và chủ đề nổi bật. |
| **IdeaGeneratorAgent** | Tạo ý tưởng nghiên cứu mới dựa trên xu hướng và khoảng trống tri thức. |
| **ReportGeneratorAgent** | Sinh báo cáo Markdown tổng hợp toàn bộ quy trình nghiên cứu. |
| **AgenticResearchOrchestrator** | Điều phối toàn bộ workflow từ thu thập đến báo cáo. |

---

## 📁 Cấu trúc thư mục
agentic-research/
│
├── data/ # Nơi lưu các file papers thu thập được
├── reports/ # Nơi lưu các báo cáo Markdown được tạo tự động
├── main.py # File chạy chính (chứa toàn bộ pipeline)
├── requirements.txt # Danh sách thư viện cần thiết
└── README.md # Tài liệu hướng dẫn này


---

## ⚙️ Cài đặt môi trường

### 1️⃣ Clone repository
```bash
git clone https://github.com/<your-username>/agentic-research.git
cd agentic-research

### 2️⃣ Tạo môi trường ảo và cài đặt thư viện
python -m venv venv
source venv/bin/activate        # Trên macOS/Linux
venv\Scripts\activate           # Trên Windows

pip install -r requirements.txt

### 3️⃣ File requirements.txt gợi ý:
arxiv
requests

🧩 Cách sử dụng
Chạy pipeline đầy đủ
python main.py


Hệ thống sẽ tự động:

Thu thập các bài báo mới nhất về Agentic AI từ ArXiv và Semantic Scholar.

Tóm tắt nội dung từng bài.

Phân tích xu hướng từ khóa và cụm từ phổ biến.

Sinh ý tưởng nghiên cứu mới.

Xuất báo cáo chi tiết tại thư mục reports/.
