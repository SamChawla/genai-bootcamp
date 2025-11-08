# modules/news.py  (UPDATED: normalize chain outputs -> string; UI formatting + safe download)
import os
import traceback
from typing import Optional, List, Dict, Any

import streamlit as st

# --- Lazy imports to match the "classic" working stack ---
Article = None
RecursiveCharacterTextSplitter = None
PromptTemplate = None
load_summarize_chain = None
Document = None
ChatOpenAI = None
ChatOllama = None

def _try_imports():
    global Article, RecursiveCharacterTextSplitter, PromptTemplate, load_summarize_chain, Document, ChatOpenAI, ChatOllama
    try:
        from newspaper import Article as _Article
        Article = _Article
    except Exception:
        Article = None

    try:
        from langchain_classic.text_splitter import RecursiveCharacterTextSplitter as _R
        RecursiveCharacterTextSplitter = _R
    except Exception:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter as _R2
            RecursiveCharacterTextSplitter = _R2
        except Exception:
            RecursiveCharacterTextSplitter = None

    try:
        from langchain_core.prompts import PromptTemplate as _P
        PromptTemplate = _P
    except Exception:
        PromptTemplate = None

    try:
        from langchain_classic.chains.summarize import load_summarize_chain as _L
        load_summarize_chain = _L
    except Exception:
        load_summarize_chain = None

    try:
        from langchain_core.documents import Document as _Doc
        Document = _Doc
    except Exception:
        Document = None

    try:
        from langchain_openai import ChatOpenAI as _COpen
        ChatOpenAI = _COpen
    except Exception:
        ChatOpenAI = None

    try:
        from langchain_community.chat_models import ChatOllama as _CO
        ChatOllama = _CO
    except Exception:
        ChatOllama = None

_try_imports()

# Simple fallback splitter if RecursiveCharacterTextSplitter is missing
class _SimpleSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_text(self, text: str):
        if not text:
            return []
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        cur = ""
        for s in sentences:
            if not s:
                continue
            if len(cur) + 1 + len(s) <= self.chunk_size:
                cur = (cur + " " + s).strip() if cur else s
            else:
                if cur:
                    chunks.append(cur)
                if len(s) > self.chunk_size:
                    step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
                    for i in range(0, len(s), step):
                        chunks.append(s[i:i+step])
                    cur = ""
                else:
                    cur = s
        if cur:
            chunks.append(cur)
        return chunks

def _make_splitter(chunk_size=1000, chunk_overlap=200, separators=None):
    if RecursiveCharacterTextSplitter:
        try:
            if separators:
                return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception:
            try:
                return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception:
                pass
    return _SimpleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

class NewsArticleSummarizer:
    def __init__(self, api_key: str = None, model_type: str = "openai", model_name: str = "gpt-4o-mini",
                 chunk_size: int = 2000, chunk_overlap: int = 200):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if ChatOpenAI is None:
                raise ImportError("ChatOpenAI (langchain_openai) not available.")
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_type == "ollama":
            if ChatOllama is None:
                raise ImportError("ChatOllama not available.")
            self.llm = ChatOllama(model=model_name, temperature=0, timeout=120)
        else:
            raise ValueError("Unsupported model_type")

        self.text_splitter = _make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])

    def fetch_article(self, url: str) -> Optional[object]:
        # Try newspaper3k first
        if Article is not None:
            try:
                art = Article(url)
                art.download()
                art.parse()
                return art
            except Exception as e:
                err_msg = str(e)
                if "lxml.html.clean" in err_msg or "lxml_html_clean" in err_msg:
                    st.warning('newspaper3k failed due to missing html_clean support. Install: pip install "lxml[html_clean]" or pip install lxml_html_clean')
                else:
                    st.warning(f"newspaper3k failed parsing URL — falling back to simpler extractor: {err_msg}")

        # Fallback: requests + BeautifulSoup
        try:
            import requests
            from bs4 import BeautifulSoup
        except Exception:
            return None

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            article_tag = soup.find("article")
            paragraphs = []
            if article_tag:
                paragraphs = [p.get_text(strip=True) for p in article_tag.find_all("p")]
            if not paragraphs:
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n\n".join([p for p in paragraphs if p])
            title = None
            title_tag = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name":"twitter:title"})
            if title_tag and title_tag.get("content"):
                title = title_tag.get("content")
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            authors = []
            author_tag = soup.find("meta", attrs={"name":"author"}) or soup.find("meta", property="article:author")
            if author_tag and author_tag.get("content"):
                authors = [author_tag.get("content")]
            publish_date = None
            date_tag = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name":"pubdate"})
            if date_tag and date_tag.get("content"):
                publish_date = date_tag.get("content")

            class SimpleArticle:
                def __init__(self, text, title=None, authors=None, publish_date=None):
                    self.text = text
                    self.title = title
                    self.authors = authors or []
                    self.publish_date = publish_date

            return SimpleArticle(text=text, title=title, authors=authors, publish_date=publish_date)
        except Exception:
            return None

    def create_documents(self, text: str) -> List[Any]:
        chunks = self.text_splitter.split_text(text)
        if Document is None:
            return [{"page_content": c} for c in chunks]
        return [Document(page_content=c) for c in chunks]

    def _build_prompts(self, summary_type: str):
        if PromptTemplate is None:
            raise ImportError("PromptTemplate not available.")
        if summary_type == "detailed":
            map_prompt_template = 'Write a detailed summary of the following text:\n\n"{text}"\n\nDETAILED SUMMARY:'
            combine_prompt_template = 'Write a detailed summary that combines the previous summaries:\n\n"{text}"\n\nFINAL DETAILED SUMMARY:'
        else:
            map_prompt_template = 'Write a concise summary of the following text:\n\n"{text}"\n\nCONCISE SUMMARY:'
            combine_prompt_template = 'Write a concise summary that combines the previous summaries:\n\n"{text}"\n\nFINAL CONCISE SUMMARY:'
        mp = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        cp = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        return mp, cp

    def _normalize_chain_output_to_text(self, out: Any) -> str:
        """
        Accept various chain outputs and return a plain string.
        Handles:
          - plain string
          - dict with 'output_text', 'summary', 'text', ...
          - other objects -> str(...)
        """
        if out is None:
            return ""
        # dict-like
        if isinstance(out, dict):
            for key in ("output_text", "summary", "text", "answer", "result"):
                if key in out and isinstance(out[key], str):
                    return out[key].strip()
            # possible nested structure with "output" etc.
            # try to find first string value
            for v in out.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return str(out)
        # string-like
        if isinstance(out, str):
            return out.strip()
        # if LangChain returns a custom object, try common attributes
        try:
            if hasattr(out, "output_text"):
                return str(getattr(out, "output_text")).strip()
        except Exception:
            pass
        return str(out)

    def summarize(self, url: str, summary_type: str = "detailed") -> Dict[str, Any]:
        art = self.fetch_article(url)
        if not art:
            return {"error": "Failed to fetch article."}

        docs = self.create_documents(art.text)
        if load_summarize_chain is None:
            return {"error": "Summarization chain component (load_summarize_chain) not available. Install langchain_classic or ensure your environment matches the classic stack."}

        mp, cp = self._build_prompts(summary_type)
        chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
        try:
            out = chain.invoke(docs)
        except Exception:
            out = chain.run(docs)

        summary_text = self._normalize_chain_output_to_text(out)
        return {
            "title": getattr(art, "title", ""),
            "authors": getattr(art, "authors", []),
            "publish_date": getattr(art, "publish_date", None),
            "summary": summary_text,
            "url": url,
            "input_doc_count": len(docs),
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


# Streamlit UI wrapper
class NewsUI:
    @staticmethod
    def render(api_key, provider, model, chunk_size=1000, chunk_overlap=200):
        st.header("📰 News Summarizer")
        input_mode = st.selectbox("Input mode", ["URL", "Paste text", "Upload file"])
        text = None
        url = None
        uploaded = None

        if input_mode == "URL":
            url = st.text_input("Article URL")
        elif input_mode == "Paste text":
            text = st.text_area("Paste article text", height=300)
        else:
            uploaded = st.file_uploader("Upload file (.txt / .pdf / .docx)", type=["txt","pdf","docx"])

        summary_style = st.selectbox("Summary style", ["detailed","concise"])
        if st.button("Generate Summary"):
            if provider == "openai" and not api_key and not os.getenv("OPENAI_API_KEY"):
                st.warning("Provide OpenAI API key (sidebar) or set OPENAI_API_KEY in environment.")
                return
            try:
                summarizer = NewsArticleSummarizer(api_key=api_key if provider=="openai" else None,
                                                    model_type=("openai" if provider=="openai" else "ollama"),
                                                    model_name=model,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap)
                if input_mode == "URL":
                    if not url:
                        st.warning("Please enter a URL.")
                        return
                    with st.spinner("Fetching & summarizing..."):
                        res = summarizer.summarize(url, summary_type=summary_style)
                    if res.get("error"):
                        st.error(res["error"])
                        return

                    st.subheader(res.get("title","Article"))
                    st.write("Authors:", ", ".join(res.get("authors") or []))
                    st.write("Published:", res.get("publish_date"))

                    # Summary display: use markdown and a text area for copy/paste
                    summary_text = res.get("summary", "")
                    st.markdown("### Summary")
                    # Markdown for basic formatting; use code block to preserve newlines if needed
                    st.markdown(summary_text)
                    # Provide full text area to easily copy
                    st.text_area("Full Summary (copyable)", summary_text, height=300)

                    # Safe download (encode to bytes)
                    try:
                        st.download_button("Download summary (.txt)", summary_text.encode("utf-8"), file_name="news_summary.txt")
                    except Exception:
                        # last-resort: pass string directly
                        st.download_button("Download summary (.txt)", summary_text, file_name="news_summary.txt")

                else:
                    if uploaded:
                        raw = uploaded.read()
                        try:
                            text = raw.decode("utf-8")
                        except Exception:
                            text = str(raw)
                    if not text:
                        st.warning("Please provide text by pasting or uploading a file.")
                        return
                    with st.spinner("Chunking and summarizing..."):
                        docs = summarizer.create_documents(text)
                        mp, cp = summarizer._build_prompts(summary_style)
                        chain = load_summarize_chain(llm=summarizer.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
                        try:
                            out = chain.invoke(docs)
                        except Exception:
                            out = chain.run(docs)

                        # Normalize output to string
                        if isinstance(out, dict):
                            # prioritize common keys
                            summary_text = out.get("output_text") or out.get("summary") or out.get("text") or str(out)
                        else:
                            summary_text = str(out)

                        st.markdown("### Summary")
                        st.markdown(summary_text)
                        st.text_area("Full Summary (copyable)", summary_text, height=300)
                        try:
                            st.download_button("Download summary (.txt)", summary_text.encode("utf-8"), file_name="news_summary.txt")
                        except Exception:
                            st.download_button("Download summary (.txt)", summary_text, file_name="news_summary.txt")

            except Exception as e:
                st.error(f"Error during summarization: {e}")
                if st.sidebar.checkbox("Show full traceback"):
                    st.code(traceback.format_exc())
