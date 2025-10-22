import streamlit as st
from pathlib import Path
from datetime import datetime

# Import các lớp Agent từ file main.py
from main import DataCollectorAgent, MLTrendAnalysisAgent, AdvancedIdeaGeneratorAgent, ReportGeneratorAgent

# Khởi tạo thư mục
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Khởi tạo các agent
collector = DataCollectorAgent()
trend_analyzer = MLTrendAnalysisAgent()
idea_generator = AdvancedIdeaGeneratorAgent()
reporter = ReportGeneratorAgent()

# ==============================
# 🎨 GIAO DIỆN STREAMLIT
# ==============================
st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI Research Assistant")
st.markdown("Tự động thu thập, phân tích và tạo báo cáo nghiên cứu từ **arXiv.org** và **Semantic Scholar**")

# --- Input form ---
with st.form("research_form"):
    keyword = st.text_input("🔍 Nhập từ khóa nghiên cứu:", value="Agentic AI")
    paper_limit = st.slider("📄 Số lượng paper mỗi nguồn:", 5, 100, 20, step=5)
    submitted = st.form_submit_button("🚀 Bắt đầu phân tích")

# --- Khi người dùng nhấn nút ---
if submitted:
    st.info("⏳ Đang thu thập dữ liệu từ ArXiv và Semantic Scholar...")
    papers = collector.fetch_all_papers(query=keyword, limit_per_source=paper_limit)

    if not papers:
        st.error("Không tìm thấy bài báo nào. Hãy thử từ khóa khác.")
        st.stop()

    st.success(f"✅ Đã thu thập {len(papers)} bài báo!")

    # --- Phân tích xu hướng ---
    st.info("📈 Đang phân tích xu hướng...")
    trends = trend_analyzer.identify_trends(papers)

    # --- Sinh ý tưởng ---
    st.info("💡 Đang đề xuất ý tưởng nghiên cứu...")
    ideas = idea_generator.generate_research_ideas(trends, papers)

    # --- Tạo báo cáo ---
    st.info("🧾 Đang tạo báo cáo tổng hợp...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"report_{timestamp}.md"

    report_path = reporter.create_markdown_report(
        papers=papers,
        trends=trends,
        ideas=ideas,
        output_file=output_file
    )

    st.success("✅ Báo cáo đã sẵn sàng!")
    st.download_button(
        label="📥 Tải xuống báo cáo Markdown",
        data=open(report_path, "r", encoding="utf-8").read(),
        file_name=output_file,
        mime="text/markdown"
    )

    # --- Hiển thị nội dung trên UI ---
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    st.markdown("---")
    st.subheader("📘 Báo cáo Nghiên Cứu")
    st.markdown(report_content)
