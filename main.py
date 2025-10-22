import os
import json
import arxiv
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import time
import numpy as np
from collections import Counter, defaultdict
import re

# Cấu hình thư mục
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class DataCollectorAgent:
    """Agent thu thập dữ liệu từ ArXiv và các nguồn miễn phí với limit linh hoạt"""
    
    def __init__(self):
        self.arxiv_client = arxiv.Client()
    
    def fetch_arxiv_papers(self, query: str, limit: int = 50, days_back: int = 30) -> List[Dict]:
        """Thu thập papers từ ArXiv với limit tùy chỉnh"""
        print(f"🔍 Đang tìm kiếm papers về '{query}' trên ArXiv (limit: {limit})...")
        
        search = arxiv.Search(
            query=query,
            max_results=limit,
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
        
        print(f"✅ Đã thu thập {len(papers)} papers từ ArXiv")
        return papers
    
    def fetch_semantic_scholar(self, query: str, limit: int = 50) -> List[Dict]:
        """Thu thập papers từ Semantic Scholar API với limit tùy chỉnh"""
        print(f"🔍 Đang tìm kiếm papers trên Semantic Scholar (limit: {limit})...")
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': min(limit, 100),  # API limit
            'fields': 'title,abstract,authors,year,publicationDate,citationCount,url,venue'
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
                            'venue': paper.get('venue', ''),
                            'source': 'semantic_scholar'
                        }
                        papers.append(paper_data)
                
                print(f"✅ Đã thu thập {len(papers)} papers từ Semantic Scholar")
                return papers
        except Exception as e:
            print(f"⚠️ Lỗi khi truy cập Semantic Scholar: {e}")
        
        return []
    
    def fetch_all_papers(self, query: str, limit_per_source: int = 50, days_back: int = 30) -> List[Dict]:
        """Thu thập papers từ tất cả nguồn với limit thống nhất"""
        print(f"\n📚 Thu thập dữ liệu với limit: {limit_per_source} papers/nguồn")
        print("=" * 60)
        
        arxiv_papers = self.fetch_arxiv_papers(query, limit_per_source, days_back)
        time.sleep(2)  # Rate limiting
        semantic_papers = self.fetch_semantic_scholar(query, limit_per_source)
        
        all_papers = arxiv_papers + semantic_papers
        print(f"\n📊 Tổng cộng: {len(all_papers)} papers từ {2} nguồn")
        return all_papers
    
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
        
        print(f"💾 Đã lưu {len(unique_papers)} papers vào {filepath}")
        return filepath


class AdvancedSummarizerAgent:
    """Agent tóm tắt papers với thuật toán TextRank và TF-IDF"""
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load English stopwords"""
        return set([
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
            'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
            'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
            'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
            'its', 'itself', 'just', 'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no',
            'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
            'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than',
            'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
            'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
            'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
            'would', 'you', 'your', 'yours', 'yourself', 'yourselves', 'also', 'may', 'however',
            'within', 'show', 'using', 'used', 'use', 'based', 'can', 'paper', 'propose', 'proposed',
            'show', 'shows', 'study', 'studies', 'present', 'presented', 'approach', 'approaches'
        ])
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý văn bản"""
        sentences = re.split(r'[.!?]+', text.lower())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Tính độ tương đồng giữa 2 câu dựa trên Jaccard similarity"""
        words1 = set(re.findall(r'\b\w+\b', sent1)) - self.stopwords
        words2 = set(re.findall(r'\b\w+\b', sent2)) - self.stopwords
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Xây dựng ma trận tương đồng giữa các câu"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._calculate_sentence_similarity(
                        sentences[i], sentences[j]
                    )
        
        return similarity_matrix
    
    def _pagerank(self, similarity_matrix: np.ndarray, damping: float = 0.85, 
                  max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Thuật toán PageRank để xếp hạng câu"""
        n = similarity_matrix.shape[0]
        
        # Normalize matrix
        row_sums = similarity_matrix.sum(axis=1)
        normalized_matrix = similarity_matrix / (row_sums[:, np.newaxis] + 1e-10)
        
        # Initialize ranks
        ranks = np.ones(n) / n
        
        # Iterate
        for _ in range(max_iter):
            prev_ranks = ranks.copy()
            ranks = (1 - damping) / n + damping * normalized_matrix.T.dot(ranks)
            
            if np.linalg.norm(ranks - prev_ranks) < tol:
                break
        
        return ranks
    
    def textrank_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Tóm tắt văn bản bằng thuật toán TextRank"""
        sentences = self._preprocess_text(text)
        
        if len(sentences) <= num_sentences:
            return '. '.join(sentences) + '.'
        
        similarity_matrix = self._build_similarity_matrix(sentences)
        ranks = self._pagerank(similarity_matrix)
        
        top_indices = ranks.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary = '. '.join([sentences[i] for i in top_indices]) + '.'
        return summary.capitalize()
    
    def tfidf_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Tóm tắt văn bản bằng TF-IDF"""
        sentences = self._preprocess_text(text)
        
        if len(sentences) <= num_sentences:
            return '. '.join(sentences) + '.'
        
        word_freq = Counter()
        for sentence in sentences:
            words = [w for w in re.findall(r'\b\w+\b', sentence) if w not in self.stopwords]
            word_freq.update(words)
        
        doc_freq = Counter()
        for sentence in sentences:
            words = set([w for w in re.findall(r'\b\w+\b', sentence) if w not in self.stopwords])
            doc_freq.update(words)
        
        n_docs = len(sentences)
        idf = {word: np.log(n_docs / (freq + 1)) for word, freq in doc_freq.items()}
        
        sentence_scores = []
        for sentence in sentences:
            words = [w for w in re.findall(r'\b\w+\b', sentence) if w not in self.stopwords]
            if not words:
                sentence_scores.append(0)
                continue
            
            score = sum(word_freq[w] * idf.get(w, 0) for w in words) / len(words)
            sentence_scores.append(score)
        
        top_indices = np.argsort(sentence_scores)[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary = '. '.join([sentences[i] for i in top_indices]) + '.'
        return summary.capitalize()
    
    def summarize_papers(self, papers: List[Dict], method: str = 'textrank') -> List[Dict]:
        """Tóm tắt tất cả papers"""
        print(f"📝 Đang tóm tắt {len(papers)} papers bằng phương pháp {method.upper()}...")
        
        for paper in papers:
            if 'summary' not in paper:
                try:
                    if method == 'textrank':
                        paper['summary'] = self.textrank_summarize(paper['abstract'], num_sentences=3)
                    elif method == 'tfidf':
                        paper['summary'] = self.tfidf_summarize(paper['abstract'], num_sentences=3)
                    else:
                        paper['summary'] = paper['abstract'][:300] + '...'
                except Exception as e:
                    print(f"⚠️ Lỗi khi tóm tắt: {e}")
                    paper['summary'] = paper['abstract'][:300] + '...'
        
        print("✅ Hoàn thành tóm tắt")
        return papers


class MLTrendAnalysisAgent:
    """Agent phân tích xu hướng với Machine Learning"""
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load stopwords"""
        return set([
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
            'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
            'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
            'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
            'its', 'itself', 'just', 'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no',
            'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
            'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than',
            'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
            'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
            'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
            'would', 'you', 'your', 'yours', 'yourself', 'yourselves', 'paper', 'study', 'research'
        ])
    
    def extract_ngrams(self, text: str, n: int = 2, top_k: int = 50) -> List[Tuple[str, int]]:
        """Trích xuất n-grams với TF-IDF weighting"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stopwords]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        ngram_freq = Counter(ngrams)
        return ngram_freq.most_common(top_k)
    
    def cluster_topics_kmeans(self, papers: List[Dict], n_clusters: int = 5) -> Dict:
        """Clustering topics bằng K-Means với TF-IDF vectors"""
        print("🤖 Đang phân cụm topics bằng K-Means...")
        
        texts = [p['title'] + ' ' + p['abstract'] for p in papers]
        
        vocabulary = Counter()
        for text in texts:
            words = [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) 
                    if w not in self.stopwords]
            vocabulary.update(words)
        
        top_words = [word for word, _ in vocabulary.most_common(500)]
        word_to_idx = {word: idx for idx, word in enumerate(top_words)}
        
        vectors = []
        for text in texts:
            words = [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) 
                    if w not in self.stopwords and w in word_to_idx]
            
            word_freq = Counter(words)
            vector = np.zeros(len(top_words))
            
            for word, freq in word_freq.items():
                idx = word_to_idx[word]
                vector[idx] = freq
            
            vectors.append(vector)
        
        X = np.array(vectors)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        
        n_clusters = min(n_clusters, len(papers))
        centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
        
        for _ in range(50):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) 
                                     else centroids[i] for i in range(n_clusters)])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        clusters = defaultdict(list)
        cluster_keywords = {}
        
        for idx, label in enumerate(labels):
            clusters[label].append(papers[idx])
        
        for cluster_id, cluster_papers in clusters.items():
            cluster_text = ' '.join([p['title'] + ' ' + p['abstract'] 
                                    for p in cluster_papers])
            
            words = [w for w in re.findall(r'\b[a-z]{3,}\b', cluster_text.lower()) 
                    if w not in self.stopwords]
            
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(10)]
            
            cluster_keywords[f"Cluster {cluster_id}"] = {
                'keywords': top_keywords,
                'size': len(cluster_papers),
                'papers': cluster_papers[:3]
            }
        
        print(f"✅ Đã phân thành {len(cluster_keywords)} clusters")
        return cluster_keywords
    
    def identify_trends(self, papers: List[Dict]) -> Dict:
        """Xác định xu hướng với ML clustering"""
        print("📊 Đang phân tích xu hướng với Machine Learning...")
        
        all_text = ' '.join([p['title'] + ' ' + p['abstract'] for p in papers])
        
        unigrams = self.extract_ngrams(all_text, n=1, top_k=30)
        keywords = dict(unigrams)
        
        bigrams = self.extract_ngrams(all_text, n=2, top_k=30)
        phrases = dict(bigrams)
        
        trigrams = self.extract_ngrams(all_text, n=3, top_k=20)
        
        clusters = self.cluster_topics_kmeans(papers, n_clusters=7)
        
        temporal_trends = self._analyze_temporal_trends(papers)
        
        print("✅ Hoàn thành phân tích xu hướng")
        
        return {
            'keywords': keywords,
            'phrases': phrases,
            'trigrams': dict(trigrams),
            'clusters': clusters,
            'temporal_trends': temporal_trends,
            'total_papers': len(papers)
        }
    
    def _analyze_temporal_trends(self, papers: List[Dict]) -> Dict:
        """Phân tích xu hướng theo thời gian"""
        temporal_data = defaultdict(int)
        
        for paper in papers:
            pub_date = paper.get('published', '')
            if pub_date:
                month = pub_date[:7]
                temporal_data[month] += 1
        
        return dict(sorted(temporal_data.items()))


class AdvancedIdeaGeneratorAgent:
    """Agent tạo ý tưởng nghiên cứu thông minh với scoring system"""
    
    def __init__(self):
        self.idea_templates = [
            "Phát triển {method} cho {application} sử dụng {technique}",
            "Nghiên cứu tích hợp {concept1} và {concept2} để {goal}",
            "Đánh giá hiệu quả của {approach} trong {domain}",
            "Tối ưu hóa {system} thông qua {optimization}",
            "Khảo sát toàn diện về {topic} và ứng dụng trong {field}",
            "Xây dựng framework {name} cho {purpose}",
            "Phân tích so sánh giữa {method1} và {method2} trong {context}",
            "Cải thiện {aspect} của {system} bằng {technique}",
        ]
    
    def _score_idea(self, keywords: List[str], trends: Dict) -> float:
        """Tính điểm cho ý tưởng dựa trên xu hướng"""
        score = 0.0
        
        for keyword in keywords:
            if keyword in trends['keywords']:
                score += trends['keywords'][keyword]
            if keyword in trends['phrases']:
                score += trends['phrases'][keyword] * 1.5
        
        return score
    
    def _find_research_gaps(self, clusters: Dict) -> List[Dict]:
        """Tìm các gap trong nghiên cứu"""
        gaps = []
        
        cluster_sizes = [(name, info['size']) for name, info in clusters.items()]
        cluster_sizes.sort(key=lambda x: x[1])
        
        for cluster_name, size in cluster_sizes[:3]:
            if size < 5:
                gap = {
                    'area': cluster_name,
                    'papers': size,
                    'keywords': clusters[cluster_name]['keywords'][:5],
                    'type': 'underexplored'
                }
                gaps.append(gap)
        
        return gaps
    
    def _generate_combination_ideas(self, clusters: Dict) -> List[Dict]:
        """Tạo ý tưởng từ việc kết hợp các clusters"""
        ideas = []
        cluster_list = list(clusters.items())
        
        for i in range(min(3, len(cluster_list))):
            for j in range(i + 1, min(5, len(cluster_list))):
                cluster1_name, cluster1_info = cluster_list[i]
                cluster2_name, cluster2_info = cluster_list[j]
                
                keywords1 = cluster1_info['keywords'][:3]
                keywords2 = cluster2_info['keywords'][:3]
                
                idea = {
                    'title': f"Tích hợp {keywords1[0]} và {keywords2[0]} trong hệ thống AI",
                    'description': f"Kết hợp insights từ {cluster1_name} và {cluster2_name}",
                    'keywords': keywords1 + keywords2,
                    'clusters': [cluster1_name, cluster2_name],
                    'type': 'combination'
                }
                ideas.append(idea)
        
        return ideas
    
    def generate_research_ideas(self, trends: Dict, papers: List[Dict]) -> List[Dict]:
        """Tạo ý tưởng nghiên cứu thông minh"""
        print("💡 Đang tạo ý tưởng nghiên cứu với AI...")
        
        ideas = []
        
        # 1. Ý tưởng từ trending keywords
        top_keywords = list(trends['keywords'].keys())[:10]
        top_phrases = list(trends['phrases'].keys())[:10]
        
        for i in range(min(5, len(top_phrases))):
            phrase = top_phrases[i]
            keyword = top_keywords[i] if i < len(top_keywords) else top_phrases[i]
            
            idea = {
                'title': self.idea_templates[i % len(self.idea_templates)].format(
                    method=phrase,
                    application=keyword,
                    technique=top_keywords[(i+1) % len(top_keywords)],
                    concept1=phrase,
                    concept2=top_phrases[(i+1) % len(top_phrases)],
                    goal="cải thiện hiệu suất và độ chính xác",
                    approach=phrase,
                    domain="Agentic AI systems",
                    system=keyword,
                    optimization=phrase,
                    topic=phrase,
                    field=keyword,
                    name=phrase.title(),
                    purpose=keyword,
                    method1=phrase,
                    method2=top_phrases[(i+1) % len(top_phrases)],
                    context=keyword,
                    aspect="performance"
                ),
                'description': f"Dựa trên {trends['keywords'].get(keyword, 0)} mentions",
                'keywords': [phrase, keyword],
                'score': self._score_idea([phrase, keyword], trends),
                'priority': 'High' if i < 2 else 'Medium',
                'type': 'trending'
            }
            ideas.append(idea)
        
        # 2. Ý tưởng từ cluster analysis
        if 'clusters' in trends:
            combination_ideas = self._generate_combination_ideas(trends['clusters'])
            ideas.extend(combination_ideas[:3])
            
            # 3. Ý tưởng từ research gaps
            gaps = self._find_research_gaps(trends['clusters'])
            for gap in gaps:
                idea = {
                    'title': f"Khám phá {gap['area']}: Hướng nghiên cứu mới",
                    'description': f"Chỉ có {gap['papers']} papers, cần nghiên cứu sâu hơn",
                    'keywords': gap['keywords'],
                    'priority': 'High',
                    'type': 'gap_filling',
                    'score': 100 / (gap['papers'] + 1)
                }
                ideas.append(idea)
        
        # 4. Ý tưởng từ temporal trends
        if 'temporal_trends' in trends:
            recent_months = list(trends['temporal_trends'].keys())[-3:]
            if recent_months:
                idea = {
                    'title': "Phân tích xu hướng mới nhất trong Agentic AI",
                    'description': f"Tập trung vào các nghiên cứu từ {recent_months[0]} đến nay",
                    'keywords': top_keywords[:5],
                    'priority': 'High',
                    'type': 'temporal',
                    'score': sum(trends['temporal_trends'][m] for m in recent_months)
                }
                ideas.append(idea)
        
        # Sort by score
        ideas.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"✅ Đã tạo {len(ideas)} ý tưởng nghiên cứu")
        return ideas[:15]  # Top 15 ideas


class ReportGeneratorAgent:
    """Agent tạo báo cáo markdown"""

    def create_markdown_report(self, papers: List[Dict], trends: Dict, 
                               ideas: List[Dict], output_file: str = None) -> str:
        """Tạo báo cáo markdown chi tiết"""
        print("🧾 Đang tạo báo cáo...")

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"research_report_{timestamp}.md"

        filepath = REPORTS_DIR / output_file

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("# 📘 Báo Cáo Nghiên Cứu Agentic AI\n\n")
            f.write(f"**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            f.write(f"**Tổng số papers:** {len(papers)}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## 🧠 Tóm Tắt Điều Hành\n\n")
            f.write(f"Báo cáo này tổng hợp {len(papers)} bài báo nghiên cứu mới nhất ")
            f.write("về Agentic AI, phân tích xu hướng công nghệ và đề xuất ")
            f.write("các hướng nghiên cứu tiềm năng trong lĩnh vực này.\n\n")

            # Trend Analysis
            f.write("## 📈 Phân Tích Xu Hướng\n\n")

            # Keywords
            if 'keywords' in trends:
                f.write("### 🔑 Từ Khóa Nổi Bật\n\n")
                f.write("| Từ Khóa | Tần Suất |\n")
                f.write("|----------|-----------|\n")
                for kw, freq in list(trends['keywords'].items())[:10]:
                    f.write(f"| {kw} | {freq} |\n")
                f.write("\n")

            # Phrases
            if 'phrases' in trends:
                f.write("### 🧩 Cụm Từ Quan Trọng\n\n")
                f.write("| Cụm Từ | Tần Suất |\n")
                f.write("|----------|-----------|\n")
                for phrase, freq in list(trends['phrases'].items())[:10]:
                    f.write(f"| {phrase} | {freq} |\n")
                f.write("\n")

            # Cluster Analysis
            if 'clusters' in trends:
                f.write("### 🪄 Phân Cụm Chủ Đề (Clusters)\n\n")
                for cluster_name, info in trends['clusters'].items():
                    f.write(f"#### 📚 {cluster_name}\n\n")
                    f.write(f"- **Số lượng papers:** {info['size']}\n")
                    f.write(f"- **Từ khóa nổi bật:** {', '.join(info['keywords'][:10])}\n\n")

            # Temporal Trends
            if 'temporal_trends' in trends:
                f.write("### ⏳ Xu Hướng Theo Thời Gian\n\n")
                for month, count in trends['temporal_trends'].items():
                    bar = "█" * min(count, 20)
                    f.write(f"- {month}: {count} papers {bar}\n")
                f.write("\n")

            # Research Ideas
            f.write("## 💡 Đề Xuất Ý Tưởng Nghiên Cứu\n\n")
            for i, idea in enumerate(ideas, 1):
                f.write(f"### {i}. {idea['title']}\n\n")
                f.write(f"**Mô tả:** {idea.get('description', 'Không có mô tả.')}\n\n")
                f.write(f"**Độ ưu tiên:** {idea.get('priority', 'N/A')}\n\n")
                f.write(f"**Loại:** {idea.get('type', 'N/A')}\n\n")
                f.write(f"**Từ khóa liên quan:** {', '.join(idea.get('keywords', []))}\n\n")
                f.write(f"**Điểm đánh giá:** {idea.get('score', 0):.2f}\n\n")

            # Recent Papers
            f.write("## 📰 Các Bài Báo Mới Nhất\n\n")
            sorted_papers = sorted(
                papers, key=lambda x: x.get('published', ''), reverse=True
            )

            for i, paper in enumerate(sorted_papers[:20], 1):
                f.write(f"### {i}. {paper['title']}\n\n")
                authors = ', '.join(paper.get('authors', [])[:3])
                if len(paper.get('authors', [])) > 3:
                    authors += " et al."
                f.write(f"**Tác giả:** {authors}\n\n")
                f.write(f"**Ngày xuất bản:** {paper.get('published', 'N/A')[:10]}\n\n")
                f.write(f"**Nguồn:** {paper.get('source', 'N/A')}\n\n")
                f.write(f"**Tóm tắt:** {paper.get('summary', paper.get('abstract', '')[:300])}...\n\n")
                if 'pdf_url' in paper:
                    f.write(f"[Tải PDF]({paper['pdf_url']})\n\n")
                elif 'url' in paper:
                    f.write(f"[Đọc bài báo]({paper['url']})\n\n")
                f.write("---\n\n")

        print(f"✅ Báo cáo đã được tạo tại: {filepath}")
        return str(filepath)



class AgenticResearchOrchestrator:
    """Orchestrator điều phối toàn bộ quy trình"""
    
    def __init__(self):
        self.collector = DataCollectorAgent()
        self.summarizer = AdvancedSummarizerAgent() 
        self.trend_analyzer = MLTrendAnalysisAgent()
        self.idea_generator = AdvancedIdeaGeneratorAgent() 
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