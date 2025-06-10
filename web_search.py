import scrapy
from lxml import html as lxml_html
from goose3 import Goose
from boilerpy3 import extractors
from inscriptis import get_text as inscriptis_text
import trafilatura
import tempfile
import os
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse, parse_qs, unquote
import nltk
from nltk.data import find
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import spacy
from spacy.util import is_package
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from readability import Document
from newspaper import Article
from scrapy.linkextractors import LinkExtractor
from nltk.tokenize import sent_tokenize
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from multiprocessing import Process, Pipe
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import re
import torch
import sys

# Set the correct Twisted reactor before importing Scrapy
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from twisted.internet import asyncioreactor
try:
    asyncioreactor.install()
except Exception:
    pass  # Already installed


# Custom path for storing nltk data
NLTK_DATA_PATH = "models/nltk"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Add custom path to nltk search paths
nltk.data.path.insert(0, NLTK_DATA_PATH)

# Check and download 'punkt' if not already present
try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt_tab", download_dir=NLTK_DATA_PATH)

MODEL_PATH = "models/all-MiniLM-L6-v2"
if os.path.exists(MODEL_PATH):
    EMBEDDING_MODEL = SentenceTransformer(MODEL_PATH)
else:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_MODEL.save(MODEL_PATH)
SUMMARIZER_PATH = "models/distilbart-cnn-12-6"

if os.path.exists(SUMMARIZER_PATH):
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PATH)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_PATH)
else:
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    summarizer_model.save_pretrained(SUMMARIZER_PATH)
    summarizer_tokenizer.save_pretrained(SUMMARIZER_PATH)

SUMMARIZER = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)
TOKENIZER = summarizer_tokenizer

SPACY_MODEL_PATH = "models/spacy/en_core_web_sm"
SPACY_MODEL_NAME = "en_core_web_sm"

#QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
#QA_MODEL_PATH = f"models/{QA_MODEL_NAME}"

#qa_model = pipeline("question-answering", model=QA_MODEL_PATH)


try:
    # If already downloaded locally, load from path
    if os.path.exists(SPACY_MODEL_PATH):
        NLP = spacy.load(SPACY_MODEL_PATH)
    # If model is installed in spaCy registry, load by name
    elif is_package(SPACY_MODEL_NAME):
        NLP = spacy.load(SPACY_MODEL_NAME)
    else:
        # Download and save to the spaCy default directory
        from spacy.cli import download
        download(SPACY_MODEL_NAME)
        NLP = spacy.load(SPACY_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model: {e}")

query_answer=""
class WebSearch(scrapy.Spider):
    name = "web_search"
    custom_settings = {
        # Enforce a crawl depth limit of 1
        'DEPTH_LIMIT': 1,
        'DEPTH_STATS_VERBOSE': True,
        # Throttle and set concurrency
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

        # Use global embedding model
        self.embedding_model = EMBEDDING_MODEL
        self.query_embedding = self.embedding_model.encode(self.query, convert_to_tensor=True)
        # Similarity threshold for relevance
        self.SIMILARITY_THRESHOLD = 0.75

        # LinkExtractor restricted to main/article tags to avoid navbars/footers
        self.link_extractor = LinkExtractor(canonicalize=True, restrict_css='main, article')

        # Step 1: run search_and_rank() to get top URLs
        self.target_urls = self.search_and_rank(query)
        self.start_urls = self.target_urls

    def start_requests(self):
        """
        Feed each URL into parse_page().
        """
        for url in self.start_urls:
            print(f"Reading:{url}")
            yield scrapy.Request(url=url, callback=self.parse_page, meta={'depth': 1})

    def universal_page_parser(self, url, html, response=None, use_browser=False):
        """
        Universal parser to extract readable text from any webpage, using multiple strategies and robust fallbacks.
        """
        import logging
        logger = logging.getLogger("universal_page_parser")
        text = ""
        tried = []
        # 1. Special case: YouTube
        youtube_match = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
        if youtube_match:
            video_id = youtube_match.group(1)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join(entry['text'] for entry in transcript).strip()
                tried.append("youtube_transcript_api")
                if text:
                    return text
            except Exception as e:
                logger.warning(f"YouTube transcript failed: {e}")

        # 2. Try trafilatura (best for news, blogs, forums, e-commerce)
        try:
            trafilatura_text = trafilatura.extract(html, include_comments=False, include_tables=True)
            tried.append("trafilatura")
            if trafilatura_text and len(trafilatura_text.strip()) > 100:
                return trafilatura_text.strip()
        except Exception as e:
            logger.warning(f"trafilatura failed: {e}")

        # 3. Try boilerpy3
        try:
            boilerpy_text = extractors.ArticleExtractor().get_content(html)
            tried.append("boilerpy3")
            if boilerpy_text and len(boilerpy_text.strip()) > 100:
                return boilerpy_text.strip()
        except Exception as e:
            logger.warning(f"boilerpy3 failed: {e}")

        # 4. Try readability-lxml
        try:
            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), 'html.parser')
            text_readability = soup.get_text(" ", strip=True)
            tried.append("readability-lxml")
            if text_readability and len(text_readability.strip()) > 100:
                return text_readability.strip()
        except Exception as e:
            logger.warning(f"readability-lxml failed: {e}")

        # 5. Try newspaper3k
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            text_newspaper = article.text or ""
            tried.append("newspaper3k")
            if text_newspaper and len(text_newspaper.strip()) > 100:
                return text_newspaper.strip()
        except Exception as e:
            logger.warning(f"newspaper3k failed: {e}")

        # 6. Try goose3
        try:
            goose_text = Goose().extract(raw_html=html).cleaned_text
            tried.append("goose3")
            if goose_text and len(goose_text.strip()) > 100:
                return goose_text.strip()
        except Exception as e:
            logger.warning(f"goose3 failed: {e}")

        # 7. Try inscriptis
        try:
            inscriptis_parsed = inscriptis_text(html)
            tried.append("inscriptis")
            if inscriptis_parsed and len(inscriptis_parsed.strip()) > 100:
                return inscriptis_parsed.strip()
        except Exception as e:
            logger.warning(f"inscriptis failed: {e}")

        # 8. Try lxml (all <p> tags)
        try:
            lxml_tree = lxml_html.fromstring(html)
            lxml_text = " ".join(lxml_tree.xpath('//p//text()'))
            tried.append("lxml <p>")
            if lxml_text and len(lxml_text.strip()) > 100:
                return lxml_text.strip()
        except Exception as e:
            logger.warning(f"lxml <p> failed: {e}")

        # 9. Try BeautifulSoup (all <p>, <li>, <span>, <div>, headers)
        try:
            soup = BeautifulSoup(html, 'html.parser')
            tags = ['p', 'li', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            bs_text = " ".join([t.get_text(" ", strip=True) for tag in tags for t in soup.find_all(tag)])
            tried.append("BeautifulSoup all tags")
            if bs_text and len(bs_text.strip()) > 100:
                return bs_text.strip()
        except Exception as e:
            logger.warning(f"BeautifulSoup all tags failed: {e}")

        # 10. Try PDF extraction if the URL looks like a PDF
        if url.lower().endswith('.pdf') or (response and 'application/pdf' in response.headers.get('Content-Type', b'').decode(errors='ignore')):
            try:
                pdf_text = self.webpage_to_text(url)
                tried.append("pdf")
                if pdf_text and len(pdf_text.strip()) > 100:
                    return pdf_text.strip()
            except Exception as e:
                logger.warning(f"PDF extraction failed: {e}")

        # 11. Try browser rendering for JS-heavy sites (optional, slow)
        if use_browser:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    page.goto(url, wait_until='networkidle')
                    content = page.content()
                    browser.close()
                soup = BeautifulSoup(content, 'html.parser')
                browser_text = " ".join([t.get_text(" ", strip=True) for t in soup.find_all(['p', 'li', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
                tried.append("playwright browser render")
                if browser_text and len(browser_text.strip()) > 100:
                    return browser_text.strip()
            except Exception as e:
                logger.warning(f"playwright browser render failed: {e}")

        # 12. Heuristic fallback: extract all visible text
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for script in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "input", "button", "svg", "canvas", "iframe", "object", "embed", "img", "video", "audio"]):
                script.decompose()
            visible_text = soup.get_text(" ", strip=True)
            tried.append("heuristic visible text")
            if visible_text and len(visible_text.strip()) > 50:
                return visible_text.strip()
        except Exception as e:
            logger.warning(f"heuristic visible text failed: {e}")

        logger.warning(f"Universal parser failed to extract substantial text from {url}. Tried: {tried}")
        return ""

    def parse_page(self, response):
        
        page_url = response.url
        if page_url in self.visited_urls:
            return
        self.visited_urls.add(page_url)
        depth = response.meta.get('depth', 1)

        youtube_match = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', page_url)
        if youtube_match:
            video_id = youtube_match.group(1)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                final_text = " ".join(entry['text'] for entry in transcript).strip()
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception):
                return
        else:
            html_content = response.text
            # Use the new universal parser
            final_text = self.universal_page_parser(response.url, html_content, response)

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
                        meta={'depth': depth + 1, 'playwright': True, 'proxy': 'http://your-proxy:port'}
                    )

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
        """
        Heuristic check to filter out advertisement URLs.
        """
        ad_keywords = ['advert', 'ads', 'doubleclick', 'sponsor', 'promo']
        return any(term in url.lower() for term in ad_keywords)

    def closed(self, reason):
        """
        After all pages are fetched, combine texts → chunk → cluster → summarize → save.
        """

        combined_text = " ".join(self.scraped_texts)
        if not combined_text.strip():
            self.logger.warning("No content extracted from any URL.")
            return

        # Chunk into 5 context‐based segments
        chunks = chunk_text_by_context(combined_text, num_chunks=5)

        # Summarize relevant clusters (up to 5 clusters)
        summary = summarize_relevant_clusters(
            self.query,
            chunks,
            similarity_threshold=None,
            num_clusters=5
        )

        # Log summary
        self.logger.info(f"====== Final Summary ======\n{summary}\n")
        answer_query(query=self.query, context=summary)
        global query_answer
      #  print(f"query_answer:{query_answer}")
    '''
        # Save summary to JSON for further use
        try:
            with open("summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info("Summary written to summary.json")
        except Exception as e:
            self.logger.error(f"Failed to write summary.json: {e}")
    '''
    # ---- Helper functions ---- #

    def search_and_rank(self, query):
        """
        Get top URLs by running DuckDuckGo + ResultHunter search, then rank them
        using SentenceTransformer.
        """
        print("Searching...")
        combined = self.combined_search(query)
        return self.rank_results_with_sentence_transformer(query, combined)

    def normalize_url(self, url):
        # Handle resulthunter redirect URLs
        if "resulthunter.com" in url:
            qs = parse_qs(urlparse(url).query)
            if "url" in qs:
                return unquote(qs["url"][0])

        # Fix relative YouTube proxy-style URLs
        if url.startswith("/videos/watch/"):
            parsed = urlparse(url)
            path_parts = parsed.path.split("/")
            if len(path_parts) >= 3:
                video_id = path_parts[3]
                if len(video_id) >= 11:  # crude check for YouTube ID length
                    return f"https://www.youtube.com/watch?v={video_id}"

        # Absolute fallback
        return url if url.startswith("http") else f"https://www.resulthunter.com{url}"

    def duckduckgo_search(self, query, num_results=10):
        """
        Perform a DuckDuckGo search by scraping the non-JavaScript HTML version
        and return the top search results with titles, URLs, and snippets.
        """
        # Encode the query for URL
        encoded_query = quote(query)
        # Construct the search URL
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        # Set headers to mimic a browser
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            # Send a GET request to the search URL with timeout and retry
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # Find all result containers
        for result in soup.find_all("div", class_="result__body", limit=num_results):
            # Extract the title and URL
            link_tag = result.find("a", class_="result__a")
            snippet_tag = result.find("a", class_="result__snippet")
            if link_tag and link_tag.get("href"):
                title = link_tag.get_text(strip=True)
                url = link_tag["href"]
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append((title, url, snippet))

        return results

    def resulthunter_search(self, query, num_results=10):
        """
        Perform a ResultHunter search and return the top search results with titles, URLs, and snippets.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        encoded_query = quote(query)
        url = f"https://www.resulthunter.com/search?q={encoded_query}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.find_all('div', class_='web-result')[:num_results]:
            link_tag = result.find('a', href=True)
            snippet_tag = result.find('p', class_='web-result-desc')
            if link_tag:
                title = link_tag.get_text(strip=True)
                link = link_tag['href']
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append((title, link, snippet))
        return results

    def combined_search(self, query, num_results=50):
        results = self.duckduckgo_search(query, num_results) + self.resulthunter_search(query, num_results)
        seen = set()
        deduped_results = []
        for title, url, snippet in results:
            norm_url = self.normalize_url(url)
            if norm_url not in seen:
                seen.add(norm_url)
                deduped_results.append((title, norm_url, snippet))
        return deduped_results

    def rank_results_with_sentence_transformer(self, query, results, top_k: int = 3):
        """
        Encode (query + (title + snippet)) with SentenceTransformer, compute cosine similarity,
        and return the top‐`top_k` URLs.

        Parameters:
        -----------
        query : str
            The search query to compare against each (title + snippet).
        results : List[Tuple[str, str, str]]
            A list of tuples, where each tuple is (title, url, snippet).
        top_k : int, optional (default=3)
            How many URLs to pick (highest cosine similarity).

        Returns:
        --------
        List[str]
            A list of the top‐`top_k` URLs from `results`, ranked by cosine similarity
            between `query` and each (title + snippet).
        """
        if not results:
            return []

        # Extract titles, URLs, and snippets from each result
        titles = [r[0] for r in results]
        urls = [r[1] for r in results]
        snippets = [r[2] for r in results]

        # Combine title + snippet so snippet influences ranking
        combined_pages = [
            f"{titles[i]} {snippets[i]}" if snippets[i] else titles[i]
            for i in range(len(results))
        ]

        # 1) Encode the query
        # 2) Encode each (title + snippet) as its own entry
        texts_to_encode = [query] + combined_pages
        embeddings = self.embedding_model.encode(
            texts_to_encode,
            convert_to_tensor=True
        )

        query_emb = embeddings[0]  # shape = (embedding_dim,)
        page_embs = embeddings[1:]  # shape = (num_results, embedding_dim)

        # Compute cosine similarities between query and each (title+snippet)
        sim_scores = util.cos_sim(query_emb, page_embs)[0]  # shape = (len(combined_pages),)

        # Pick the top‐`top_k` indices (clamped to the available number of results)
        k = min(top_k, len(results))
        top_indices = sim_scores.argsort(descending=True)[:k].tolist()

        # Return only the URLs of the top‐`top_k`
        return [urls[i] for i in top_indices]


# ==== Chunk + Cluster + Summarize Helpers ==== #

def chunk_text_by_context(text, num_chunks=50):
    sentences = sent_tokenize(text)
    embeddings = EMBEDDING_MODEL.encode(sentences)

    # Dimensionality reduction for stability
    n_components = min(50, embeddings.shape[1])
    reduced_embeddings = PCA(n_components=n_components).fit_transform(embeddings)

    clustering = AgglomerativeClustering(n_clusters=num_chunks)
    labels = clustering.fit_predict(reduced_embeddings)

    chunks = {}
    for sent, lbl in zip(sentences, labels):
        chunks.setdefault(lbl, []).append(sent)
    organized = [" ".join(chunks[i]) for i in sorted(chunks.keys())]
    return organized


def safe_summarize_iterative(text: str,
                             max_length: int = 500,
                             min_length: int = 400,
                             overlap: int = 100) -> str:
    """
    Iterative summarizer that never recurses. Splits and processes in a queue.
    """
    MAX_MODEL_TOKENS = 1024
    WORD_COUNT_THRESHOLD = max_length

    # If already short by word count, return as-is
    if len(text.split()) <= WORD_COUNT_THRESHOLD:
        return text

    from collections import deque
    queue = deque([text])
    summaries = []

    while queue:
        current = queue.popleft()
        # If current chunk is short enough, keep it
        if len(current.split()) <= WORD_COUNT_THRESHOLD:
            summaries.append(current)
            continue

        tokens = TOKENIZER.encode(current, add_special_tokens=False)
        # If under token limit, summarize directly
        if len(tokens) <= MAX_MODEL_TOKENS:
            try:
                s = SUMMARIZER(current, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            except Exception:
                s = current  # fallback
            summaries.append(s)
            continue

        # Otherwise: split on sentences if possible
        sentences = sent_tokenize(current)
        if len(sentences) > 1:
            mid = len(sentences) // 2
            queue.append(" ".join(sentences[:mid]))
            queue.append(" ".join(sentences[mid:]))
        else:
            # Single very long sentence: split by token-chunk
            start = 0
            while start < len(tokens):
                end = min(start + MAX_MODEL_TOKENS, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = TOKENIZER.decode(
                    chunk_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                queue.append(chunk_text)
                if end == len(tokens):
                    break
                start = end - overlap

    # Once we have a list of partial summaries, combine and return
    combined = " ".join(summaries)
    return combined


def safe_summarize(text: str,
                   max_length: int = 750,
                   min_length: int = 500,
                   overlap: int = 250,
                   depth: int = 0,
                   max_depth: int = 25) -> str:
    """
    Recursive summarizer:
      - Skips summarization for short-by-word-count text.
      - Keeps recursing until under model token limits,
        splitting on sentences or tokens as needed.
      - Wraps recursive calls in try/except RecursionError to fall back.
      - If depth > max_depth, returns text as-is.
    """

    MAX_MODEL_TOKENS = 1024
    WORD_COUNT_THRESHOLD = max_length

    # ——— Recursion guard —————————————
    if depth > max_depth:
        # too many nested calls; fallback to iterative approach
        return safe_summarize_iterative(text, max_length, min_length, overlap)

    # Skip summarization if word count is already short
    if len(text.split()) <= WORD_COUNT_THRESHOLD:
        return text

    # Tokenize to check length
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    if len(tokens) <= MAX_MODEL_TOKENS:
        try:
            return SUMMARIZER(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        except Exception:
            # if summarizer errors out, just return the original text
            return text

    # Too many tokens—split on sentences if possible
    sentences = sent_tokenize(text)
    if len(sentences) > 1:
        mid = len(sentences) // 2
        first_half = " ".join(sentences[:mid])
        second_half = " ".join(sentences[mid:])

        try:
            summary1 = safe_summarize(first_half, max_length, min_length, overlap, depth + 1, max_depth)
            summary2 = safe_summarize(second_half, max_length, min_length, overlap, depth + 1, max_depth)
            combined = f"{summary1} {summary2}"
            return safe_summarize(combined, max_length, min_length, overlap, depth + 1, max_depth)
        except RecursionError:
            # Recursion blew past the limit—fallback to iterative
            return safe_summarize_iterative(text, max_length, min_length, overlap)

    else:
        # Only one (very long) sentence—split by token chunks
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_MODEL_TOKENS, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = TOKENIZER.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - overlap

        # Summarize each chunk (or skip if already short)
        chunk_summaries = []
        for chunk in chunks:
            if len(chunk.split()) <= WORD_COUNT_THRESHOLD:
                chunk_summaries.append(chunk)
            else:
                try:
                    cs = SUMMARIZER(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
                except Exception:
                    cs = chunk  # fallback
                chunk_summaries.append(cs)

        combined = " ".join(chunk_summaries)
        try:
            return safe_summarize(combined, max_length, min_length, overlap, depth + 1, max_depth)
        except RecursionError:
            return combined  # fallback if recursion fails


def summarize_relevant_clusters(input_query, texts, similarity_threshold=None, num_clusters=200):
    embeddings = EMBEDDING_MODEL.encode(texts, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # 1) Automatically set threshold if not provided
    if similarity_threshold is None:
        upper_tri_indices = torch.triu_indices(sim_matrix.size(0),
                                               sim_matrix.size(1),
                                               offset=1)
        similarities = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        similarity_threshold = similarities.mean().item()

    # 2) Deduplicate
    keep_indices = []
    for i in range(len(texts)):
        if not any(sim_matrix[i][j] > similarity_threshold for j in keep_indices):
            keep_indices.append(i)

    dedup_texts = [texts[i] for i in keep_indices]
    if not dedup_texts:
        return []

    dedup_embeddings = embeddings[keep_indices]

    # 3) KMeans clustering
    kmeans = KMeans(n_clusters=min(num_clusters, len(dedup_texts)), random_state=0)
    labels = kmeans.fit_predict(dedup_embeddings.cpu().numpy())

    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(dedup_texts[idx])

    # 4) Filter clusters by relevance to `input_query`
    doc = NLP(input_query)
    keywords = {token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "VERB")}

    def is_cluster_relevant(cluster_texts):
        joined = " ".join(cluster_texts).lower()
        return any(k in joined for k in keywords)

    relevant_clusters = [c for c in clusters.values() if is_cluster_relevant(c)]
    if not relevant_clusters:
        relevant_clusters = list(clusters.values())

    # 5) Summarize each cluster safely
    cluster_summaries = []
    for cluster in relevant_clusters:
        combined_text = " ".join(cluster)
        # Use our recursive-safe summarizer
        cluster_summary = safe_summarize(combined_text)
        cluster_summaries.append(cluster_summary)

    # 6) Final combined summary
    final_text = " ".join(cluster_summaries)
    final_summary = safe_summarize(final_text)
    return [final_summary]

def answer_query(query, context):
    if isinstance(context, list):
        context = " ".join(context)
        #print("  [Info] Joined list into a single string of length", len(context))
    '''
    print("Answering...")
    print("step1")
    global query_answer
    print("step2")
    result = qa_model(question=query, context=context)
    print("Done")
    query_answer=result['answer']
    print(f"Answer:{query_answer}")
    '''
    global query_answer
    query_answer=context
  #  print(f"query_anserer::{query_answer}")
def _crawl(query, conn):
    """
    Wrapper that actually runs the Scrapy crawl. This must be top‐level
    (not nested) so Windows multiprocessing can pickle it.
    """
    process = CrawlerProcess(get_project_settings())
    process.crawl(WebSearch, query=query)
    process.start()  # blocks until crawl is finished
    conn.send(query_answer)  # send it back to the parent
    conn.close()
   # answer = query_answer
    #conn.send(answer)
    #conn.close()
'''
    # After crawl, read summary from file
    try:
        with open("summary.json", "r") as f:
            summary = json.load(f)
    except Exception as e:
        summary = [f"Error reading summary.json: {e}"]

    conn.send(summary)
    conn.close()
'''

def search_web(query: str):
    parent_conn, child_conn = Pipe()
    p = Process(target=_crawl, args=(query, child_conn))
    p.start()
    # In the parent, child_conn is not needed:
    child_conn.close()

    # Wait for the child to finish, and read from parent_conn:
    retrieved_answer = parent_conn.recv()  # this blocks until child sends
    p.join()
    parent_conn.close()

    if not retrieved_answer:
        return "Web Search is currently unavailable please check you internet connection and try again"
    return retrieved_answer

if __name__ == "__main__":
   # print(search_web("Oblivion Remastered duplication glitch"))
   # print(search_web("current president of USA?"))
   # print(search_web("what are the latest AI models?"))
   # print(search_web("current price of mountain dew?"))
   # print(search_web("current temperature in Kansas City MO?"
    
        query_value = "What is the best free photo editor in 2025?"
        result = search_web(query_value)

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(result)