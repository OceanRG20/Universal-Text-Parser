# ─── Standard Library ─────────────────────────────────────────────
import os
import re
import sys
import tempfile
import logging
from multiprocessing import Process, Pipe
from urllib.parse import quote, urlparse, parse_qs, unquote

# ─── Third-Party Libraries ────────────────────────────────────────
import requests
import scrapy
import fitz  # PyMuPDF
import torch
import trafilatura
import nltk
from nltk.tokenize import sent_tokenize
from nltk.data import find

from bs4 import BeautifulSoup
from goose3 import Goose
from boilerpy3 import extractors
from inscriptis import get_text as inscriptis_text
from readability import Document
from newspaper import Article
from youtube_transcript_api import (
    YouTubeTranscriptApi, TranscriptsDisabled,
    NoTranscriptFound, VideoUnavailable
)
import lxml.html as lxml_html

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.linkextractors import LinkExtractor

from playwright.sync_api import sync_playwright

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

import spacy
from spacy.util import is_package

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from duckduckgo_search import DDGS as ddg

# Set the correct Twisted reactor before importing Scrapy
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from twisted.internet import asyncioreactor
try:
    asyncioreactor.install()
except Exception:
    pass  # Already installed


# Set the correct Twisted reactor before importing Scrapy
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from twisted.internet import asyncioreactor
try:
    asyncioreactor.install()
except Exception:
    pass  # Already installed

class WebSearch(scrapy.Spider):
    name = "web_search"
    custom_settings = {
        'DEPTH_LIMIT': 1,
        'CONCURRENT_REQUESTS': 4,
        'DOWNLOAD_DELAY': 0.3,
        'AUTOTHROTTLE_ENABLED': True,
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'DOWNLOAD_HANDLERS': {
            'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
        },
    }

    def __init__(self, query=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not query:
            raise ValueError("Usage: scrapy crawl web_search -a query='your question'")
        self.query = query
        self.scraped_texts = []
        self.scraped_hashes = set()
        self.visited_urls = set()

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.query_embedding = self.embedding_model.encode(self.query, convert_to_tensor=True)
        self.SIMILARITY_THRESHOLD = 0.75
        self.link_extractor = LinkExtractor(canonicalize=True, restrict_css='main, article')
        self.target_urls = self.search_and_rank(query)
        self.start_urls = self.target_urls

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse_page, meta={'depth': 1})

    def parse_page(self, response):
        page_url = response.url
        if page_url in self.visited_urls:
            return
        self.visited_urls.add(page_url)
        depth = response.meta.get('depth', 1)

        if re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', page_url):
            return  # skip YouTube, handled elsewhere if needed

        html_content = response.text
        final_text = self.extract_text_fallbacks(response.url, html_content, response)

        if not final_text:
            return

        hash_val = hash(final_text)
        if hash_val in self.scraped_hashes:
            return
        self.scraped_hashes.add(hash_val)
        self.scraped_texts.append(final_text)

        if depth < self.custom_settings['DEPTH_LIMIT']:
            for link in response.css('a::attr(href)').getall():
                link_url = response.urljoin(link)
                if not self.is_ad_link(link_url):
                    yield scrapy.Request(
                        url=link_url,
                        callback=self.parse_page,
                        meta={'depth': depth + 1}
                    )

    def extract_text_fallbacks(self, url, html, response):
        try:
            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), 'html.parser')
            text_readability = soup.get_text(" ", strip=True)
        except Exception:
            text_readability = ""

        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            text_newspaper = article.text or ""
        except Exception:
            text_newspaper = ""

        try:
            goose_text = Goose().extract(raw_html=html).cleaned_text
        except Exception:
            goose_text = ""

        try:
            trafilatura_text = trafilatura.extract(html)
        except Exception:
            trafilatura_text = ""

        try:
            boilerpy_text = extractors.ArticleExtractor().get_content(html)
        except Exception:
            boilerpy_text = ""

        try:
            lxml_tree = lxml_html.fromstring(html)
            lxml_text = " ".join(lxml_tree.xpath('//p//text()'))
        except Exception:
            lxml_text = ""

        try:
            soup = BeautifulSoup(html, 'html.parser')
            bs_text = " ".join(p.get_text() for p in soup.find_all('p'))
        except Exception:
            bs_text = ""

        try:
            inscriptis_parsed = inscriptis_text(html)
        except Exception:
            inscriptis_parsed = ""

        text_options = [
            text_newspaper, text_readability, goose_text, trafilatura_text,
            boilerpy_text, lxml_text, bs_text, inscriptis_parsed
        ]
        final = max(text_options, key=lambda t: len(t.strip()) if t else 0, default="")

        if not final:
            try:
                pdf_text = self.webpage_to_text(url)
                if pdf_text:
                    return pdf_text
            except Exception:
                pass

        if not final:
            try:
                title = response.css('h1::text').get("") or ""
                paragraphs = response.css('p::text').getall()
                final = title + " " + " ".join(paragraphs)
            except Exception:
                final = ""

        if not final:
            try:
                sentences = re.findall(r'([A-Z][^.!?]*[.!?])', html)
                final = " ".join(sentences)
            except Exception:
                final = ""

        return final.strip()

    def webpage_to_text(self, url: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf_path = tmp_pdf.name

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url, wait_until='networkidle')
                page.pdf(path=pdf_path, format="A4", print_background=True)
                browser.close()

            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()

            return text.strip()
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def is_ad_link(self, url):
        ad_keywords = ['advert', 'ads', 'doubleclick', 'sponsor', 'promo']
        return any(term in url.lower() for term in ad_keywords)

        def closed(self, reason):
            combined_text = " ".join(self.scraped_texts)
            if not combined_text.strip():
                self.logger.warning("No content extracted.")
                return
    
            chunks = chunk_text_by_context(combined_text, num_chunks=5)
            summary = summarize_relevant_clusters(
                self.query,
                chunks,
                similarity_threshold=None,
                num_clusters=5
            )
            self.logger.info(f"====== Final Summary ======\n{summary}\n")
            # answer_query(query=self.query, context=summary)  # Removed undefined function call
            global query_answer
            query_answer = summary
    
    def summarize_relevant_clusters(query, chunks, similarity_threshold=None, num_clusters=5):
        """
        Summarizes the most relevant clusters of text chunks to the query using sentence embeddings.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(query, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
    
        # Cluster the chunks
        if len(chunks) < num_clusters:
            num_clusters = len(chunks)
        if num_clusters <= 1:
            selected_chunks = [chunks[int(similarities.argmax())]]
        else:
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(chunk_embeddings.cpu().numpy())
            selected_chunks = []
            for cluster_id in range(num_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                if not cluster_indices:
                    continue
                # Pick the most relevant chunk in each cluster
                best_idx = max(cluster_indices, key=lambda i: similarities[i])
                selected_chunks.append(chunks[best_idx])
    
        # Optionally filter by similarity threshold
        if similarity_threshold is not None:
            selected_chunks = [c for i, c in enumerate(selected_chunks) if similarities[i] >= similarity_threshold]
    
        # Concatenate selected chunks
        summary = "\n\n".join(selected_chunks)
        return summary

logging.basicConfig(level=logging.INFO)

def chunk_text_by_context(text, num_chunks=5):
    """
    Splits the input text into num_chunks chunks, attempting to preserve sentence boundaries.
    """
    sentences = sent_tokenize(text)
    if num_chunks <= 0 or len(sentences) == 0:
        return [text]
    avg = max(1, len(sentences) // num_chunks)
    chunks = []
    for i in range(0, len(sentences), avg):
        chunk = " ".join(sentences[i:i+avg])
        chunks.append(chunk)
    return chunks

def duckduckgo_search(query, max_results=10):
    with DDGS() as ddgs:
        results = [r["href"] for r in ddgs.text(query, max_results=max_results)]
    return [r['href'] for r in results if 'href' in r]

def resulthunter_search(query, max_results=10):
    try:
        response = requests.get(
            f"https://www.resulthunter.com/search?q={query}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a["href"] for a in soup.select("a") if a.get("href", "").startswith("http")]
        return links[:max_results]
    except Exception:
        return []

def combined_search(query, max_results=10):
    duck_links = duckduckgo_search(query, max_results)
    resulthunter_links = resulthunter_search(query, max_results)
    combined = list(dict.fromkeys(duck_links + resulthunter_links))  # de-duplicate
    return combined[:max_results]

def rank_results_with_sentence_transformer(query, urls, top_n=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([query] + urls, convert_to_tensor=True)
    query_embedding = embeddings[0]
    url_embeddings = embeddings[1:]
    scores = util.cos_sim(query_embedding, url_embeddings)[0]
    ranked = sorted(zip(urls, scores), key=lambda x: x[1], reverse=True)
    return [url for url, _ in ranked[:top_n]]


def run_spider(query, urls):
    """
    Runs the WebSearch spider with the given query and urls, and returns the summary.
    """
    from multiprocessing import Pipe
    parent_conn, child_conn = Pipe()

    def f(conn):
        process = CrawlerProcess(get_project_settings())
        process.crawl(WebSearch, query=query, start_urls=urls)
        process.start()
        # After spider closes, get the global query_answer
        global query_answer
        conn.send(globals().get("query_answer", ""))
        conn.close()

    p = Process(target=f, args=(child_conn,))
    p.start()
    summary = parent_conn.recv()
    p.join()
    return summary

def search_web(query):
    print("Searching...")
    all_urls = combined_search(query)
    if not all_urls:
        return "❌ No URLs found."

    ranked_urls = rank_results_with_sentence_transformer(query, all_urls)
    print("Crawling top URLs...")

    summary = run_spider(query, ranked_urls)

    if not summary:
        return "⚠️ Web Search is currently unavailable. Please check your internet connection and try again."

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    return summary


if __name__ == "__main__":
    question = "What is the best free photo editor in 2025?"
    result = search_web(question)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)