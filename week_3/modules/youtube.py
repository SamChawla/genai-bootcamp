# modules/youtube.py
import os
import uuid
import traceback
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Lazy imports (classic-first) ---
yt_dlp = None
whisper = None
RecursiveCharacterTextSplitter = None
ChatOpenAI = None
ChatOllama = None
OpenAIEmbeddings = None
Chroma = None
ConversationalRetrievalChain = None
ChatPromptTemplate = None
ConversationBufferMemory = None
load_summarize_chain = None
Document = None

def _try_imports():
    global yt_dlp, whisper, RecursiveCharacterTextSplitter, ChatOpenAI, ChatOllama
    global OpenAIEmbeddings, Chroma, ConversationalRetrievalChain, ChatPromptTemplate, ConversationBufferMemory
    global load_summarize_chain, Document

    try:
        import yt_dlp as _yd
        yt_dlp = _yd
    except Exception:
        yt_dlp = None

    try:
        import whisper as _ws
        whisper = _ws
    except Exception:
        whisper = None

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
        from langchain_openai import ChatOpenAI as _COpen
        ChatOpenAI = _COpen
    except Exception:
        try:
            from langchain.chat_models import ChatOpenAI as _COpen2
            ChatOpenAI = _COpen2
        except Exception:
            ChatOpenAI = None

    try:
        from langchain_community.chat_models import ChatOllama as _CO
        ChatOllama = _CO
    except Exception:
        ChatOllama = None

    try:
        from langchain_openai import OpenAIEmbeddings as _OE
        OpenAIEmbeddings = _OE
    except Exception:
        OpenAIEmbeddings = None

    try:
        from langchain_community.vectorstores import Chroma as _Chroma
        Chroma = _Chroma
    except Exception:
        Chroma = None

    try:
        from langchain_classic.chains import ConversationalRetrievalChain as _CR
        ConversationalRetrievalChain = _CR
    except Exception:
        ConversationalRetrievalChain = None

    try:
        from langchain_core.prompts import ChatPromptTemplate as _CPT
        ChatPromptTemplate = _CPT
    except Exception:
        ChatPromptTemplate = None

    try:
        from langchain_classic.memory import ConversationBufferMemory as _CBM
        ConversationBufferMemory = _CBM
    except Exception:
        ConversationBufferMemory = None

    try:
        from langchain_classic.chains.summarize import load_summarize_chain as _LS
        load_summarize_chain = _LS
    except Exception:
        load_summarize_chain = None

    try:
        from langchain_core.documents import Document as _Doc
        Document = _Doc
    except Exception:
        Document = None

_try_imports()

# --- Fallback simple splitter ---
class _SimpleSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_text(self, text: str) -> List[str]:
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
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for i in range(0, len(s), step):
                        chunks.append(s[i:i+step])
                    cur = ""
                else:
                    cur = s
        if cur:
            chunks.append(cur)
        return chunks

def _make_splitter(chunk_size=1000, chunk_overlap=100, separators=None):
    if RecursiveCharacterTextSplitter:
        try:
            if separators is None:
                return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        except Exception:
            try:
                return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception:
                pass
    return _SimpleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# --- Helpers ---
def _normalize_chain_output(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        for key in ("output_text", "summary", "text", "answer", "result"):
            if key in out and isinstance(out[key], str):
                return out[key].strip()
        # take first string value
        for v in out.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(out)
    # try common attributes
    try:
        if hasattr(out, "output_text"):
            return str(getattr(out, "output_text")).strip()
    except Exception:
        pass
    return str(out)

# -------------------------
# Embeddings / LLM wrappers
# -------------------------
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError("OpenAIEmbeddings not available.")
            self.embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "nomic":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()

class LLMModel:
    def __init__(self, model_type="openai", model_name="gpt-4"):
        self.model_type = model_type
        self.model_name = model_name
        if model_type == "openai":
            if ChatOpenAI is None:
                raise ImportError("ChatOpenAI not available.")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        elif model_type == "ollama":
            if ChatOllama is None:
                raise ImportError("ChatOllama not available.")
            self.llm = ChatOllama(model=model_name, temperature=0, timeout=120)
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")

# -------------------------
# Youtube Video Summarizer
# -------------------------
class YoutubeVideoSummarizer:
    def __init__(self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"):
        if whisper is None:
            raise ImportError("whisper not installed. Install openai-whisper to use the YouTube summarizer.")
        if yt_dlp is None:
            raise ImportError("yt-dlp not installed. Install yt-dlp to download YouTube videos.")
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict[str, str]:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str, str]:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
                video_title = info.get("title", "Unknown Title")
                return audio_path, video_title
        except Exception as e:
            # ffmpeg / ffprobe missing commonly causes yt_dlp to fail postprocessing
            msg = str(e)
            if "ffmpeg" in msg.lower() or "ffprobe" in msg.lower():
                raise RuntimeError("ffmpeg/ffprobe not found. Install ffmpeg or provide --ffmpeg-location to yt-dlp.")
            raise

    def transcribe_audio(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        return result.get("text", "")

    def create_documents(self, text: str, video_title: str):
        splitter = _make_splitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
        texts = splitter.split_text(text)
        docs = []
        for t in texts:
            if Document:
                try:
                    docs.append(Document(page_content=t, metadata={"source": video_title}))
                except Exception:
                    docs.append({"page_content": t, "metadata": {"source": video_title}})
            else:
                docs.append({"page_content": t, "metadata": {"source": video_title}})
        return docs

    def create_vector_store(self, documents: List[Any]):
        if Chroma is None:
            raise ImportError("Chroma vectorstore not installed.")
        return Chroma.from_documents(documents=documents, embedding=self.embedding_model.embedding_fn, collection_name=f"yt_{uuid.uuid4().hex}")

    def generate_summary(self, documents: List[Any]) -> str:
        if load_summarize_chain is None:
            raise ImportError("Summarize chain not available.")
        if ChatPromptTemplate is None:
            raise ImportError("ChatPromptTemplate not available.")
        mp = ChatPromptTemplate.from_template('Write a concise summary of the following transcript section:\n\n"{text}"\n\nCONCISE SUMMARY:')
        cp = ChatPromptTemplate.from_template('Write a detailed summary of the following video transcript sections:\n\n"{text}"\n\nDETAILED SUMMARY:')
        chain = load_summarize_chain(llm=self.llm_model.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
        try:
            out = chain.invoke(documents)
        except Exception:
            out = chain.run(documents)
        return _normalize_chain_output(out)

    def setup_qa_chain(self, vector_store):
        if ConversationalRetrievalChain is None:
            raise ImportError("ConversationalRetrievalChain not available.")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if ConversationBufferMemory else None
        return ConversationalRetrievalChain.from_llm(llm=self.llm_model.llm, retriever=vector_store.as_retriever(), memory=memory, verbose=True)

    def process_video(self, url: str) -> Dict[str, Any]:
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, title)
            # create vector store (may be memory heavy)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)
            try:
                os.remove(audio_path)
            except Exception:
                pass
            summary = self.generate_summary(documents)
            return {"summary": summary, "qa_chain": qa_chain, "title": title, "full_transcript": transcript}
        except Exception as e:
            return {"error": str(e)}

# -------------------------
# Streamlit UI wrapper
# -------------------------
class YouTubeUI:
    @staticmethod
    def render(api_key, provider, model, embedding_type="openai"):
        st.header("🎥 YouTube Summarizer")
        yt_url = st.text_input("YouTube URL")
        # model field is passed by app_streamlit; keep a select fallback for convenience
        yt_model_choice = st.text_input("LLM model for summarization", value=model)
        yt_embedding_choice = st.selectbox("Embeddings backend", ["openai", "chroma", "nomic"], index=0)
        process_btn = st.button("Process YouTube Video")

        if process_btn:
            if not yt_url:
                st.warning("Please enter a YouTube URL.")
                return
            try:
                with st.spinner("Downloading & transcribing (may take a minute)..."):
                    s = YoutubeVideoSummarizer(llm_type=("openai" if provider == "openai" else "ollama"),
                                               llm_model_name=yt_model_choice,
                                               embedding_type=(yt_embedding_choice or embedding_type))
                    res = s.process_video(yt_url)
                if res.get("error"):
                    st.error(res.get("error"))
                    # helpful hint for ffmpeg
                    if "ffmpeg" in res.get("error").lower() or "ffprobe" in res.get("error").lower():
                        st.info("ffmpeg/ffprobe are required by yt-dlp for audio extraction. Install them or set --ffmpeg-location for yt-dlp.")
                    return
                st.subheader(res.get("title","Video"))
                st.markdown("### Summary")
                st.write(res.get("summary",""))
                if st.checkbox("Show transcript"):
                    st.text_area("Transcript", res.get("full_transcript",""), height=400)
                st.session_state["yt_result"] = res
                st.success("Processed video — QA available below.")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())

        # QA UI
        if "yt_result" in st.session_state:
            st.markdown("---")
            st.subheader("Ask questions about the video")
            question = st.text_input("Question about the video")
            if st.button("Ask video"):
                qa_chain = st.session_state["yt_result"].get("qa_chain")
                if not qa_chain:
                    st.error("QA chain not available for this video.")
                    return
                with st.spinner("Retrieving and answering..."):
                    try:
                        try:
                            out = qa_chain({"question": question})
                        except Exception:
                            try:
                                out = qa_chain.invoke({"question": question})
                            except Exception:
                                out = qa_chain.run(question)
                        answer = out.get("answer") if isinstance(out, dict) else str(out)
                        st.markdown("### Answer")
                        st.write(answer)
                        # append to history
                        hist = st.session_state.get("yt_qa_history", [])
                        hist.append((question, answer))
                        st.session_state["yt_qa_history"] = hist
                    except Exception as e:
                        st.error(f"QA failed: {e}")
                        if st.sidebar.checkbox("Show traceback"):
                            st.code(traceback.format_exc())
