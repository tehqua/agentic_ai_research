import os
import json
import arxiv
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time

# Cấu hình thư mục
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

class DataCollectorAgent:
    """Agent thu thập dữ liệu từ ArXiv và các nguồn miễn phí"""
    
    def __init__(self):
        self.arxiv_client = arxiv.Client()
    
    def fetch_arxiv_papers(self, query: str, max_results: int = 50, days_back: int = 30) -> List[Dict]:
        """Thu thập papers từ ArXiv"""
        print(f" Đang tìm kiếm papers về '{query}' trên ArXiv...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for result in self.arxiv_client.results(search):
            if result.published.replace(tzinfo=None) < cutoff_date:
                continue
                
            paper_data = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.isoformat(),
                'pdf_url': result.pdf_url,
                'categories': result.categories,
                'source': 'arxiv'
            }
            papers.append(paper_data)
        
        print(f" Đã thu thập {len(papers)} papers từ ArXiv")
        return papers
    
    def fetch_semantic_scholar(self, query: str, limit: int = 20) -> List[Dict]:
        """Thu thập papers từ Semantic Scholar API (miễn phí)"""
        print(f" Đang tìm kiếm papers trên Semantic Scholar...")
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,publicationDate,citationCount,url'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                papers = []
                
                for paper in data.get('data', []):
                    if paper.get('abstract'):
                        paper_data = {
                            'title': paper.get('title', ''),
                            'authors': [a.get('name', '') for a in paper.get('authors', [])],
                            'abstract': paper.get('abstract', ''),
                            'published': paper.get('publicationDate', ''),
                            'citations': paper.get('citationCount', 0),
                            'url': paper.get('url', ''),
                            'source': 'semantic_scholar'
                        }
                        papers.append(paper_data)
                
                print(f" Đã thu thập {len(papers)} papers từ Semantic Scholar")
                return papers
        except Exception as e:
            print(f" Lỗi khi truy cập Semantic Scholar: {e}")
        
        return []
    
    def save_papers(self, papers: List[Dict], filename: str = "papers.json"):
        """Lưu papers vào file JSON"""
        filepath = DATA_DIR / filename
        
        # Load existing papers
        existing_papers = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_papers = json.load(f)
        
        # Merge and remove duplicates
        all_papers = existing_papers + papers
        unique_papers = {p['title']: p for p in all_papers}.values()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(list(unique_papers), f, indent=2, ensure_ascii=False)
        
        print(f" Đã lưu {len(unique_papers)} papers vào {filepath}")
        return filepath


class SummarizerAgent:
    """Agent tóm tắt papers sử dụng kỹ thuật extractive summarization miễn phí"""
    
    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Tóm tắt văn bản bằng phương pháp extractive đơn giản"""
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Chọn các câu quan trọng dựa trên độ dài và vị trí
        important_sentences = []
        
        # Luôn lấy câu đầu tiên (thường là ý chính)
        if sentences:
            important_sentences.append(sentences[0])
        
        # Lấy các câu dài nhất (thường chứa nhiều thông tin)
        remaining = sentences[1:]
        remaining_sorted = sorted(remaining, key=len, reverse=True)
        important_sentences.extend(remaining_sorted[:num_sentences-1])
        
        return ' '.join(important_sentences)
    
    def summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """Tóm tắt tất cả papers"""
        print(f" Đang tóm tắt {len(papers)} papers...")
        
        for paper in papers:
            if 'summary' not in paper:
                paper['summary'] = self.extractive_summarize(paper['abstract'], num_sentences=3)
        
        print(" Hoàn thành tóm tắt")
        return papers


class TrendAnalysisAgent:
    """Agent phân tích xu hướng từ papers"""
    
    def extract_keywords(self, papers: List[Dict], top_n: int = 20) -> Dict:
        """Trích xuất từ khóa quan trọng"""
        from collections import Counter
        import re
        
        print(" Đang phân tích xu hướng...")
        
        # Common stopwords
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                        'was', 'are', 'been', 'be', 'this', 'that', 'these', 
                        'those', 'we', 'our', 'can', 'will', 'using', 'used'])
        
        all_text = ' '.join([
            p['title'] + ' ' + p['abstract'] 
            for p in papers
        ]).lower()
        
        # Extract words (2-3 word phrases)
        words = re.findall(r'\b[a-z]+\b', all_text)
        bigrams = [' '.join([words[i], words[i+1]]) 
                   for i in range(len(words)-1) 
                   if words[i] not in stopwords and words[i+1] not in stopwords]
        
        # Count frequency
        word_freq = Counter([w for w in words if w not in stopwords and len(w) > 3])
        bigram_freq = Counter(bigrams)
        
        top_words = dict(word_freq.most_common(top_n))
        top_bigrams = dict(bigram_freq.most_common(top_n))
        
        return {
            'keywords': top_words,
            'phrases': top_bigrams,
            'total_papers': len(papers)
        }
    
    def identify_trends(self, papers: List[Dict]) -> Dict:
        """Xác định các xu hướng chính"""
        trends = self.extract_keywords(papers)
        
        # Phân loại theo chủ đề
        categories = {
            'AI Agents': 0,
            'LLM & Foundation Models': 0,
            'RAG & Knowledge': 0,
            'Automation & Tools': 0,
            'Healthcare': 0,
            'Security': 0,
            'Other': 0
        }
        
        keywords_map = {
            'AI Agents': ['agent', 'agentic', 'autonomous', 'multi-agent'],
            'LLM & Foundation Models': ['llm', 'language model', 'gpt', 'transformer'],
            'RAG & Knowledge': ['rag', 'retrieval', 'knowledge', 'embedding'],
            'Automation & Tools': ['automation', 'tool', 'workflow', 'pipeline'],
            'Healthcare': ['medical', 'health', 'clinical', 'patient'],
            'Security': ['security', 'privacy', 'adversarial', 'attack']
        }
        
        for paper in papers:
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            categorized = False
            
            for category, keywords in keywords_map.items():
                if any(kw in text for kw in keywords):
                    categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                categories['Other'] += 1
        
        trends['categories'] = categories
        print(" Hoàn thành phân tích xu hướng")
        return trends


class IdeaGeneratorAgent:
    """Agent tạo ý tưởng nghiên cứu dựa trên xu hướng"""
    
    def generate_research_ideas(self, trends: Dict, papers: List[Dict]) -> List[Dict]:
        """Tạo ý tưởng nghiên cứu"""
        print(" Đang tạo ý tưởng nghiên cứu...")
        
        ideas = []
        
        # Dựa trên từ khóa hot nhất
        top_keywords = list(trends['keywords'].keys())[:5]
        top_phrases = list(trends['phrases'].keys())[:5]
        
        # Template ý tưởng
        templates = [
            "Nghiên cứu ứng dụng {} trong các hệ thống {}",
            "Phát triển framework {} cho {}",
            "Đánh giá hiệu quả của {} trong bối cảnh {}",
            "Tối ưu hóa {} sử dụng kỹ thuật {}",
            "Kết hợp {} và {} để cải thiện performance"
        ]
        
        # Tạo ý tưởng từ keywords
        for i, phrase in enumerate(top_phrases[:5]):
            if i < len(top_keywords):
                idea = {
                    'title': templates[i % len(templates)].format(phrase, top_keywords[i]),
                    'description': f"Xu hướng mới nổi với {trends['phrases'][phrase]} mentions",
                    'related_keywords': [phrase, top_keywords[i]],
                    'priority': 'High' if i < 2 else 'Medium'
                }
                ideas.append(idea)
        
        # Ý tưởng từ gap analysis
        categories = trends['categories']
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for i, (cat, count) in enumerate(sorted_cats[:3]):
            if count > 2:
                idea = {
                    'title': f"Khảo sát toàn diện về {cat} trong Agentic AI",
                    'description': f"Có {count} papers gần đây về chủ đề này, cần tổng hợp hệ thống",
                    'related_keywords': [cat.lower()],
                    'priority': 'High' if i == 0 else 'Medium'
                }
                ideas.append(idea)
        
        print(f" Đã tạo {len(ideas)} ý tưởng nghiên cứu")
        return ideas


class ReportGeneratorAgent:
    """Agent tạo báo cáo markdown"""
    
    def create_markdown_report(self, papers: List[Dict], trends: Dict, 
                               ideas: List[Dict], output_file: str = None) -> str:
        """Tạo báo cáo markdown chi tiết"""
        print(" Đang tạo báo cáo...")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"research_report_{timestamp}.md"
        
        filepath = REPORTS_DIR / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("#  Báo Cáo Nghiên Cứu Agentic AI\n\n")
            f.write(f"**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            f.write(f"**Tổng số papers:** {len(papers)}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Tóm Tắt Điều Hành\n\n")
            f.write(f"Báo cáo này tổng hợp {len(papers)} bài báo nghiên cứu mới nhất ")
            f.write("về Agentic AI, phân tích xu hướng công nghệ và đề xuất ")
            f.write("các hướng nghiên cứu tiềm năng.\n\n")
            
            # Trend Analysis
            f.write("## Phân Tích Xu Hướng\n\n")
            
            f.write("### Từ Khóa Nổi Bật\n\n")
            f.write("| Từ Khóa | Tần Suất |\n")
            f.write("|---------|----------|\n")
            for kw, freq in list(trends['keywords'].items())[:10]:
                f.write(f"| {kw} | {freq} |\n")
            f.write("\n")
            
            f.write("### Cụm Từ Quan Trọng\n\n")
            f.write("| Cụm Từ | Tần Suất |\n")
            f.write("|---------|----------|\n")
            for phrase, freq in list(trends['phrases'].items())[:10]:
                f.write(f"| {phrase} | {freq} |\n")
            f.write("\n")
            
            f.write("### Phân Bố Theo Chủ Đề\n\n")
            for category, count in sorted(trends['categories'].items(), 
                                         key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / trends['total_papers']) * 100
                    bar = "█" * int(percentage / 5)
                    f.write(f"**{category}**: {count} papers ({percentage:.1f}%) {bar}\n\n")
            
            # Research Ideas
            f.write("##  Đề Xuất Ý Tưởng Nghiên Cứu\n\n")
            for i, idea in enumerate(ideas, 1):
                f.write(f"### {i}. {idea['title']}\n\n")
                f.write(f"**Mô tả:** {idea['description']}\n\n")
                f.write(f"**Độ ưu tiên:** {idea['priority']}\n\n")
                f.write(f"**Từ khóa liên quan:** {', '.join(idea['related_keywords'])}\n\n")
            
            # Recent Papers
            f.write("##  Các Bài Báo Mới Nhất\n\n")
            
            # Sort by date
            sorted_papers = sorted(papers, 
                                  key=lambda x: x.get('published', ''), 
                                  reverse=True)
            
            for i, paper in enumerate(sorted_papers[:20], 1):
                f.write(f"### {i}. {paper['title']}\n\n")
                f.write(f"**Tác giả:** {', '.join(paper['authors'][:3])}")
                if len(paper['authors']) > 3:
                    f.write(f" et al.")
                f.write("\n\n")
                f.write(f"**Ngày xuất bản:** {paper.get('published', 'N/A')[:10]}\n\n")
                f.write(f"**Nguồn:** {paper['source']}\n\n")
                
                if 'summary' in paper:
                    f.write(f"**Tóm tắt:** {paper['summary']}\n\n")
                else:
                    f.write(f"**Abstract:** {paper['abstract'][:300]}...\n\n")
                
                if 'pdf_url' in paper:
                    f.write(f"**Link:** [{paper['pdf_url']}]({paper['pdf_url']})\n\n")
                elif 'url' in paper:
                    f.write(f"**Link:** [{paper['url']}]({paper['url']})\n\n")
                
                f.write("---\n\n")
            
            # Footer
            f.write("## Ghi Chú\n\n")
            f.write("Báo cáo này được tạo tự động bởi Agentic AI Research System.\n")
            f.write("Dữ liệu được thu thập từ ArXiv và Semantic Scholar.\n\n")
        
        print(f" Đã tạo báo cáo: {filepath}")
        return str(filepath)


class AgenticResearchOrchestrator:
    """Orchestrator điều phối toàn bộ quy trình"""
    
    def __init__(self):
        self.collector = DataCollectorAgent()
        self.summarizer = SummarizerAgent()
        self.trend_analyzer = TrendAnalysisAgent()
        self.idea_generator = IdeaGeneratorAgent()
        self.report_generator = ReportGeneratorAgent()
    
    def run_research_pipeline(self, query: str = "Agentic AI", 
                             days_back: int = 30,
                             max_papers: int = 50):
        """Chạy toàn bộ pipeline nghiên cứu"""
        print("=" * 60)
        print(" BẮT ĐẦU AGENTIC RESEARCH PIPELINE")
        print("=" * 60)
        print()
        
        # Step 1: Thu thập dữ liệu
        print("BƯỚC 1: THU THẬP DỮ LIỆU")
        print("-" * 60)
        arxiv_papers = self.collector.fetch_arxiv_papers(query, max_papers, days_back)
        time.sleep(2)  # Rate limiting
        semantic_papers = self.collector.fetch_semantic_scholar(query, limit=20)
        
        all_papers = arxiv_papers + semantic_papers
        self.collector.save_papers(all_papers)
        print()
        
        # Step 2: Tóm tắt
        print("BƯỚC 2: TÓM TẮT PAPERS")
        print("-" * 60)
        summarized_papers = self.summarizer.summarize_papers(all_papers)
        print()
        
        # Step 3: Phân tích xu hướng
        print("BƯỚC 3: PHÂN TÍCH XU HƯỚNG")
        print("-" * 60)
        trends = self.trend_analyzer.identify_trends(summarized_papers)
        print()
        
        # Step 4: Tạo ý tưởng
        print("BƯỚC 4: TẠO Ý TƯỞNG NGHIÊN CỨU")
        print("-" * 60)
        ideas = self.idea_generator.generate_research_ideas(trends, summarized_papers)
        print()
        
        # Step 5: Tạo báo cáo
        print("BƯỚC 5: TẠO BÁO CÁO")
        print("-" * 60)
        report_path = self.report_generator.create_markdown_report(
            summarized_papers, trends, ideas
        )
        print()
        
        print("=" * 60)
        print(" HOÀN THÀNH PIPELINE!")
        print("=" * 60)
        print(f"\n Báo cáo đã được lưu tại: {report_path}")
        print(f" Tổng số papers: {len(all_papers)}")
        print(f" Số ý tưởng đề xuất: {len(ideas)}")
        print()
        
        return {
            'papers': summarized_papers,
            'trends': trends,
            'ideas': ideas,
            'report_path': report_path
        }


def main():
    
    orchestrator = AgenticResearchOrchestrator()
    
    # Cấu hình
    query = "Agentic AI OR AI Agents OR Autonomous AI"
    days_back = 30
    max_papers = 50
    
    print(f" Cấu hình:")
    print(f"   - Từ khóa tìm kiếm: {query}")
    print(f"   - Thời gian: {days_back} ngày gần nhất")
    print(f"   - Số lượng papers tối đa: {max_papers}")
    print()
    
    # Chạy pipeline
    results = orchestrator.run_research_pipeline(query, days_back, max_papers)
    
    print(" Hệ thống đã hoàn thành!")
    print(" Kiểm tra thư mục 'reports' để xem báo cáo chi tiết")
    

if __name__ == "__main__":
    main()