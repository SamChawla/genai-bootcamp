# modules/news.py
import os
from typing import Optional, List, Dict, Any
import streamlit as st
import traceback

# Lazy/optional imports — import when needed to avoid import-time failures
def _try_imports():
    global Article, RecursiveCharacterTextSplitter, PromptTemplate, load_summarize_chain, ChatOpenAI, ChatOllama, Document
    Article = None
    RecursiveCharacterTextSplitter = None
    PromptTemplate = None
    load_summarize_chain = None
    ChatOpenAI = None
    ChatOllama = None
    Document = None

    try:
        from newspaper import Article as _Article
        Article = _Article
    except Exception:
        Article = None

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter as _R
        from langchain.prompts import PromptTemplate as _P
        from langchain.chains.summarize import load_summarize_chain as _L
        from langchain.schema import Document as _Doc
        RecursiveCharacterTextSplitter = _R
        PromptTemplate = _P
        load_summarize_chain = _L
        Document = _Doc
    except Exception:
        RecursiveCharacterTextSplitter = None
        PromptTemplate = None
        load_summarize_chain = None
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

class NewsArticleSummarizer:
    def __init__(self, api_key: str = None, model_type: str = "openai", model_name: str = "gpt-4o-mini", chunk_size: int = 2000, chunk_overlap: int = 200):
        if model_type == "openai" and not (api_key or os.getenv("OPENAI_API_KEY")):
            raise ValueError("OPENAI_API_KEY is required for OpenAI models.")
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if ChatOpenAI is None:
                raise ImportError("langchain_openai.ChatOpenAI not available.")
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_type == "ollama":
            if ChatOllama is None:
                raise ImportError("ChatOllama not available.")
            self.llm = ChatOllama(model=model_name, temperature=0, timeout=120)
        else:
            raise ValueError("Unsupported model_type")

        if RecursiveCharacterTextSplitter is None:
            raise ImportError("LangChain text splitter not available.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

    def fetch_article(self, url: str) -> Optional[object]:
        if Article is None:
            raise ImportError("newspaper3k not installed.")
        try:
            art = Article(url)
            art.download()
            art.parse()
            return art
        except Exception:
            return None

    def create_documents(self, text: str) -> List[Any]:
        texts = self.text_splitter.split_text(text)
        return [Document(page_content=t) for t in texts]

    def _build_prompts(self, summary_type: str):
        if summary_type == "detailed":
            map_prompt_template = 'Write a detailed summary of the following text:\\n\\n"{text}"\\n\\nDETAILED SUMMARY:'
            combine_prompt_template = 'Write a detailed summary that combines the previous summaries:\\n\\n"{text}"\\n\\nFINAL DETAILED SUMMARY:'
        else:
            map_prompt_template = 'Write a concise summary of the following text:\\n\\n"{text}"\\n\\nCONCISE SUMMARY:'
            combine_prompt_template = 'Write a concise summary that combines the previous summaries:\\n\\n"{text}"\\n\\nFINAL CONCISE SUMMARY:'
        mp = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        cp = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        return mp, cp

    def summarize(self, url: str, summary_type: str = "detailed") -> Dict[str, Any]:
        art = self.fetch_article(url)
        if not art:
            return {"error": "Failed to fetch article."}

        docs = self.create_documents(art.text)
        if load_summarize_chain is None:
            raise ImportError("load_summarize_chain not available (langchain).")
        mp, cp = self._build_prompts(summary_type)
        chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
        try:
            out = chain.invoke(docs)
        except Exception:
            out = chain.run(docs)
        summary_text = out if isinstance(out, str) else str(out)
        return {
            "title": getattr(art, "title", ""),
            "authors": getattr(art, "authors", []),
            "publish_date": getattr(art, "publish_date", None),
            "summary": summary_text,
            "url": url,
            "input_doc_count": len(docs),
            "model_info": {"type": self.model_type, "name": self.model_name},
        }

# Streamlit wrapper UI (News)
class NewsUI:
    @staticmethod
    def render(api_key, provider, model, chunk_size=1000, chunk_overlap=200):
        st.header("News Summarizer")
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
                    else:
                        st.subheader(res.get("title","Article"))
                        st.write("Authors:", ", ".join(res.get("authors") or []))
                        st.write("Published:", res.get("publish_date"))
                        st.markdown("### Summary")
                        st.write(res.get("summary",""))
                        st.download_button("Download summary (.txt)", res.get("summary",""), file_name="news_summary.txt")
                else:
                    if uploaded:
                        raw = uploaded.read()
                        try:
                            text = raw.decode("utf-8")
                        except Exception:
                            text = str(raw)
                    if not text:
                        st.warning("Please paste text or upload a file.")
                        return
                    with st.spinner("Chunking and summarizing..."):
                        docs = summarizer.create_documents(text)
                        mp, cp = summarizer._build_prompts(summary_style)
                        from langchain.chains.summarize import load_summarize_chain
                        chain = load_summarize_chain(llm=summarizer.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
                        try:
                            out = chain.invoke(docs)
                        except Exception:
                            out = chain.run(docs)
                        summary_text = out if isinstance(out, str) else str(out)
                        st.markdown("### Summary")
                        st.write(summary_text)
                        st.download_button("Download summary (.txt)", summary_text, file_name="news_summary.txt")
            except Exception as e:
                st.error(f"Error: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())
