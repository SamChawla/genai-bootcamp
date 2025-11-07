# modules/voice.py
import os
import tempfile
import uuid
import streamlit as st
import traceback
from typing import List, Any

# Lazy imports
def _try_imports():
    global whisper, sd, sf, OpenAIEmbeddings, Chroma, ChatOpenAI, ConversationalRetrievalChain, ConversationBufferMemory, RecursiveCharacterTextSplitter
    whisper = None
    sd = None
    sf = None
    OpenAIEmbeddings = None
    Chroma = None
    ChatOpenAI = None
    ConversationalRetrievalChain = None
    ConversationBufferMemory = None
    RecursiveCharacterTextSplitter = None

    try:
        import whisper as _w
        whisper = _w
    except Exception:
        whisper = None

    try:
        import sounddevice as _sd
        sd = _sd
    except Exception:
        sd = None

    try:
        import soundfile as _sf
        sf = _sf
    except Exception:
        sf = None

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
        from langchain_openai import ChatOpenAI as _COpen
        ChatOpenAI = _COpen
    except Exception:
        ChatOpenAI = None

    try:
        from langchain.chains import ConversationalRetrievalChain as _CR
        ConversationalRetrievalChain = _CR
    except Exception:
        ConversationalRetrievalChain = None

    try:
        from langchain.memory import ConversationBufferMemory as _CBM
        ConversationBufferMemory = _CBM
    except Exception:
        ConversationBufferMemory = None

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter as _R
        RecursiveCharacterTextSplitter = _R
    except Exception:
        RecursiveCharacterTextSplitter = None

_try_imports()

# Document loader/processor
class DocumentProcessor:
    def __init__(self):
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("Text splitter not available.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n",". "," ",""])
        if OpenAIEmbeddings is None:
            # still allow if not installed; will error later on create_vector_store
            self.embeddings = None
        else:
            self.embeddings = OpenAIEmbeddings()

    def load_documents(self, directory: str) -> List[Any]:
        # Simple loader: read .txt files and ignore others if loader libs missing
        docs = []
        for root, _, files in os.walk(directory):
            for fn in files:
                fpath = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                try:
                    if ext == ".txt":
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        docs.append({"page_content": content, "metadata": {"source": fn}})
                    elif ext == ".pdf":
                        # try to use PyPDFLoader if available
                        try:
                            from langchain.document_loaders import PyPDFLoader
                            loader = PyPDFLoader(fpath)
                            docs.extend(loader.load())
                        except Exception:
                            # fallback: skip with warning in streamlit context
                            pass
                    elif ext in (".md", ".markdown"):
                        try:
                            from langchain.document_loaders import UnstructuredMarkdownLoader
                            loader = UnstructuredMarkdownLoader(fpath)
                            docs.extend(loader.load())
                        except Exception:
                            pass
                except Exception:
                    pass
        return docs

    def process_documents(self, documents: List[Any]):
        # documents could be langchain Document objects or dicts
        # we will attempt to split using our splitter
        out = []
        for d in documents:
            text = d.page_content if hasattr(d, "page_content") else d.get("page_content", "")
            chunks = self.text_splitter.split_text(text)
            for c in chunks:
                if hasattr(d, "metadata"):
                    md = d.metadata
                else:
                    md = d.get("metadata", {})
                # if langchain Document available we would create Document; else dict
                try:
                    from langchain_core.documents import Document as LC_Doc
                    out.append(LC_Doc(page_content=c, metadata=md))
                except Exception:
                    out.append({"page_content": c, "metadata": md})
        return out

    def create_vector_store(self, documents: List[Any], persist_directory: str):
        if Chroma is None:
            raise ImportError("Chroma vector store not installed.")
        # choose embedding function
        if self.embeddings is None:
            raise ImportError("Embeddings not initialized (install langchain_openai or other embeddings).")
        # create or load
        os.makedirs(persist_directory, exist_ok=True)
        if os.listdir(persist_directory):
            vs = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            vs = Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=persist_directory)
            try:
                vs.persist()
            except Exception:
                pass
        return vs

# TTS generator via ElevenLabs
class VoiceGenerator:
    def __init__(self, api_key: str = None):
        try:
            from elevenlabs import generate, set_api_key, voices
            self.generate_fn = generate
            set_api_key(api_key or os.getenv("ELEVEN_LABS_API_KEY"))
            self.available_voices = [v.name for v in voices()]
        except Exception:
            self.generate_fn = None
            self.available_voices = []
        self.default_voice = "Rachel" if "Rachel" in self.available_voices else (self.available_voices[0] if self.available_voices else None)

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        if self.generate_fn is None:
            raise ImportError("ElevenLabs client not available.")
        selected_voice = voice_name or self.default_voice
        try:
            audio_bytes = self.generate_fn(text=text, voice=selected_voice, model="eleven_multilingual_v2")
            # audio_bytes may be bytes
            if isinstance(audio_bytes, (bytes, bytearray)):
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp.write(audio_bytes)
                tmp.flush()
                tmp.close()
                return tmp.name
            # else try join if iterable
            if hasattr(audio_bytes, "__iter__"):
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                for chunk in audio_bytes:
                    tmp.write(chunk)
                tmp.flush()
                tmp.close()
                return tmp.name
            return None
        except Exception as e:
            raise

# Voice Assistant RAG
class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key: str = None):
        if whisper is None:
            raise ImportError("whisper not installed.")
        if ChatOpenAI is None:
            # voice assistant still can work with other LLMs if ChatOpenAI missing
            pass
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) if ChatOpenAI else None
        self.embeddings = OpenAIEmbeddings() if OpenAIEmbeddings else None
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_store(self, vector_store):
        self.vector_store = vector_store
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if ConversationBufferMemory else None
        if ConversationalRetrievalChain is None:
            raise ImportError("ConversationalRetrievalChain not available.")
        self.qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.vector_store.as_retriever(), memory=memory, verbose=True)

    def record_audio(self, duration=5):
        if sd is None:
            raise ImportError("sounddevice not available for recording.")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        # audio_array may be numpy array or path; accept both
        tmp = None
        try:
            if isinstance(audio_array, str) and os.path.exists(audio_array):
                # file path
                res = self.whisper_model.transcribe(audio_array)
                return res.get("text", "")
            else:
                # numpy array -> write to wav
                tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                import soundfile as sf
                sf.write(tmpf.name, audio_array, self.sample_rate)
                res = self.whisper_model.transcribe(tmpf.name)
                try:
                    os.unlink(tmpf.name)
                except Exception:
                    pass
                return res.get("text", "")
        except Exception as e:
            raise

    def generate_response(self, query: str):
        if self.qa_chain is None:
            raise RuntimeError("QA chain not initialized.")
        try:
            out = None
            try:
                out = self.qa_chain({"question": query})
            except Exception:
                try:
                    out = self.qa_chain.invoke({"question": query})
                except Exception:
                    out = self.qa_chain.run(query)
            if isinstance(out, dict):
                return out.get("answer") or out.get("output_text") or str(out)
            return str(out)
        except Exception:
            raise

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        return self.voice_generator.generate_voice_response(text, voice_name)

# UI wrapper
class VoiceUI:
    @staticmethod
    def render(api_key, provider, model, elevenlabs_key=None):
        st.header("Voice Assistant (RAG)")
        st.subheader("1) Setup Knowledge Base")
        uploaded_files = st.file_uploader("Upload documents (.pdf/.txt/.md)", accept_multiple_files=True, type=["pdf","txt","md"])
        persist_dir = st.text_input("Vector store persist directory", value="knowledge_base")

        if st.button("Process & Index Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
                return
            try:
                tmpdir = tempfile.mkdtemp()
                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as fh:
                        fh.write(f.getbuffer())
                proc = DocumentProcessor()
                docs = proc.load_documents(tmpdir)
                processed = proc.process_documents(docs)
                vs = proc.create_vector_store(processed, persist_dir)
                st.session_state.vector_store = vs
                st.success(f"Indexed {len(processed)} chunks into '{persist_dir}'")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())
            finally:
                try:
                    for fn in os.listdir(tmpdir):
                        os.remove(os.path.join(tmpdir, fn))
                    os.rmdir(tmpdir)
                except Exception:
                    pass

        st.markdown("---")
        st.subheader("2) Use Voice Assistant")
        if "vector_store" not in st.session_state:
            st.warning("Please index documents first.")
            return

        if "voice_assistant" not in st.session_state:
            try:
                st.session_state.voice_assistant = VoiceAssistantRAG(elevenlabs_key)
                st.session_state.voice_assistant.setup_vector_store(st.session_state.vector_store)
            except Exception as e:
                st.error(f"Failed to create voice assistant: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())
                return

        assistant = st.session_state.voice_assistant

        op = st.radio("Input mode", ["Record (microphone)", "Upload audio file", "Type text question"])
        if op == "Record (microphone)":
            duration = st.slider("Duration (sec)", 1, 20, 5)
            if st.button("Start Recording"):
                try:
                    rec = assistant.record_audio(duration)
                    st.session_state.last_audio = rec
                    st.success("Recorded audio available.")
                except Exception as e:
                    st.error(f"Recording failed: {e}")

        elif op == "Upload audio file":
            audio_file = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav","mp3","m4a"])
            if audio_file and st.button("Use uploaded audio"):
                tmp = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{audio_file.name}")
                with open(tmp, "wb") as fh:
                    fh.write(audio_file.getbuffer())
                st.session_state.last_audio_file = tmp
                st.success("Uploaded audio saved.")
        else:
            text_query = st.text_input("Type your question")

        if st.button("Get Answer"):
            try:
                if op == "Record (microphone)":
                    if "last_audio" not in st.session_state:
                        st.warning("Please record audio first.")
                        return
                    transcript = assistant.transcribe_audio(st.session_state.last_audio)
                    st.write("You said:", transcript)
                    answer = assistant.generate_response(transcript)
                    st.markdown("### Answer")
                    st.write(answer)
                    if elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY"):
                        audio_path = assistant.text_to_speech(answer)
                        if audio_path:
                            st.audio(audio_path)
                            try:
                                os.unlink(audio_path)
                            except Exception:
                                pass
                elif op == "Upload audio file":
                    if "last_audio_file" not in st.session_state:
                        st.warning("No uploaded audio file.")
                        return
                    if hasattr(assistant, "whisper_model"):
                        res = assistant.whisper_model.transcribe(st.session_state.last_audio_file)
                        transcript = res.get("text")
                    else:
                        import soundfile as sf
                        arr, sr = sf.read(st.session_state.last_audio_file)
                        transcript = assistant.transcribe_audio(arr)
                    st.write("You said:", transcript)
                    answer = assistant.generate_response(transcript)
                    st.write("Answer:", answer)
                    if elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY"):
                        ap = assistant.text_to_speech(answer)
                        if ap:
                            st.audio(ap)
                            try:
                                os.unlink(ap)
                            except Exception:
                                pass
                else:
                    if not text_query:
                        st.warning("Please type a question.")
                        return
                    answer = assistant.generate_response(text_query)
                    st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Voice-RAG error: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())
