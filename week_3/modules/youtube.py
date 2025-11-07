# modules/youtube.py
import os
import uuid
import streamlit as st
import traceback
from typing import List, Dict, Any

# Lazy imports
def _try_imports():
    global yt_dlp, whisper, RecursiveCharacterTextSplitter, ChatOpenAI, ChatOllama
    global OpenAIEmbeddings, Chroma, ConversationalRetrievalChain, ChatPromptTemplate, ConversationBufferMemory, load_summarize_chain, Document
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
        from langchain.text_splitter import RecursiveCharacterTextSplitter as _R
        RecursiveCharacterTextSplitter = _R
    except Exception:
        RecursiveCharacterTextSplitter = None

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

    try:
        from langchain_openai import OpenAIEmbeddings as _OpenEmb
        OpenAIEmbeddings = _OpenEmb
    except Exception:
        OpenAIEmbeddings = None

    try:
        from langchain_community.vectorstores import Chroma as _Chroma
        Chroma = _Chroma
    except Exception:
        Chroma = None

    try:
        from langchain.chains import ConversationalRetrievalChain as _CR
        ConversationalRetrievalChain = _CR
    except Exception:
        ConversationalRetrievalChain = None

    try:
        from langchain_core.prompts import ChatPromptTemplate as _CPT
        ChatPromptTemplate = _CPT
    except Exception:
        ChatPromptTemplate = None

    try:
        from langchain.memory import ConversationBufferMemory as _CBM
        ConversationBufferMemory = _CBM
    except Exception:
        ConversationBufferMemory = None

    try:
        from langchain.chains.summarize import load_summarize_chain as _LS
        load_summarize_chain = _LS
    except Exception:
        load_summarize_chain = None

    try:
        from langchain_core.documents import Document as _Doc
        Document = _Doc
    except Exception:
        Document = None

_try_imports()

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError("OpenAI embeddings not available.")
            self.embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "nomic":
            # placeholder for nomic/hf embeddings
            from langchain.embeddings import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()
        else:
            from langchain.embeddings import HuggingFaceEmbeddings
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

class YoutubeVideoSummarizer:
    def __init__(self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"):
        if whisper is None:
            raise ImportError("whisper not installed. Install whisper to use YouTube summarizer.")
        if yt_dlp is None:
            raise ImportError("yt-dlp not installed. Install yt-dlp to download YouTube videos.")
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")

    def download_video(self, url: str) -> tuple[str, str]:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        result = self.whisper_model.transcribe(audio_path)
        return result.get("text", "")

    def create_documents(self, text: str, video_title: str):
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("Text splitter not available.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n","\n",". "," ",""])
        texts = splitter.split_text(text)
        docs = []
        for t in texts:
            # use simple dict-like document shape if langchain Document not available
            if Document:
                docs.append(Document(page_content=t, metadata={"source": video_title}))
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
        mp = ChatPromptTemplate.from_template('Write a concise summary of the following transcript section:\\n\\n"{text}"\\n\\nCONCISE SUMMARY:')
        cp = ChatPromptTemplate.from_template('Write a detailed summary of the following video transcript sections:\\n\\n"{text}"\\n\\nDETAILED SUMMARY:')
        chain = load_summarize_chain(llm=self.llm_model.llm, chain_type="map_reduce", map_prompt=mp, combine_prompt=cp, verbose=False)
        try:
            out = chain.invoke(documents)
        except Exception:
            out = chain.run(documents)
        return out if isinstance(out, str) else str(out)

    def setup_qa_chain(self, vector_store):
        if ConversationalRetrievalChain is None:
            raise ImportError("ConversationalRetrievalChain not available.")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=self.llm_model.llm, retriever=vector_store.as_retriever(), memory=memory, verbose=True)

    def process_video(self, url: str) -> Dict[str, Any]:
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, title)
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

# Streamlit wrapper
class YouTubeUI:
    @staticmethod
    def render(api_key, provider, model, embedding_type="openai"):
        st.header("YouTube Summarizer")
        yt_url = st.text_input("YouTube URL")
        yt_model_choice = st.text_input("LLM model for summarization", value=model)
        yt_embedding_choice = st.selectbox("Embeddings backend", ["openai","chroma","nomic"], index=0)
        process_btn = st.button("Process YouTube Video")

        if process_btn:
            if not yt_url:
                st.warning("Please enter a YouTube URL.")
                return
            try:
                with st.spinner("Downloading & transcribing (may take a minute)..."):
                    s = YoutubeVideoSummarizer(llm_type=("openai" if provider=="openai" else "ollama"),
                                               llm_model_name=yt_model_choice,
                                               embedding_type=yt_embedding_choice)
                    res = s.process_video(yt_url)
                if res.get("error"):
                    st.error(res.get("error"))
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
                    except Exception as e:
                        st.error(f"QA failed: {e}")
                        if st.sidebar.checkbox("Show traceback"):
                            st.code(traceback.format_exc())
