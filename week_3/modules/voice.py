# modules/voice.py
import os
import tempfile
import uuid
import traceback
from typing import List, Any, Optional, Dict

# Lazy optional imports
try:
    import whisper as _whisper
except Exception:
    _whisper = None

try:
    import sounddevice as _sd
except Exception:
    _sd = None

try:
    import soundfile as _sf
except Exception:
    _sf = None

try:
    import numpy as _np
except Exception:
    _np = None

# LangChain / embeddings / vectorstore optional imports
try:
    from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings
except Exception:
    _OpenAIEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma as _Chroma
except Exception:
    _Chroma = None

try:
    from langchain_core.documents import Document as _LC_Doc
except Exception:
    _LC_Doc = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter
except Exception:
    _RecursiveCharacterTextSplitter = None

# Streamlit for UI bits
try:
    import streamlit as st
except Exception:
    st = None

# -------------------------
# DocumentProcessor
# -------------------------
class DocumentProcessor:
    """
    Minimal document loader/splitter/embedding helper.
    - Load plain .txt files reliably.
    - Try PDF/Markdown loaders if langchain_community loaders are available.
    - Provide robust create_vector_store with embedding checks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if _RecursiveCharacterTextSplitter is None:
            # fallback simple splitter
            self.text_splitter = None
        else:
            self.text_splitter = _RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
            )

        if _OpenAIEmbeddings is None:
            self.embeddings = None
        else:
            # Will use default environment OPENAI_API_KEY
            try:
                self.embeddings = _OpenAIEmbeddings()
            except Exception:
                self.embeddings = None

    def load_documents(self, directory: str) -> List[Any]:
        docs = []
        for root, _, files in os.walk(directory):
            for fn in files:
                fpath = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                try:
                    if ext == ".txt":
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        if content and content.strip():
                            docs.append({"page_content": content, "metadata": {"source": fn}})
                    elif ext == ".pdf":
                        # try PyPDFLoader if available
                        try:
                            from langchain_community.document_loaders import PyPDFLoader
                            loader = PyPDFLoader(fpath)
                            docs.extend(loader.load())
                        except Exception:
                            # skip PDF if loader not present
                            pass
                    elif ext in (".md", ".markdown"):
                        try:
                            from langchain_community.document_loaders import UnstructuredMarkdownLoader
                            loader = UnstructuredMarkdownLoader(fpath)
                            docs.extend(loader.load())
                        except Exception:
                            pass
                except Exception:
                    # ignore single-file errors
                    pass
        return docs

    def process_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into chunks. Returns list of chunks as langchain Document objects (if available)
        or plain dicts.
        """
        out = []
        if not documents:
            return out

        if self.text_splitter:
            for d in documents:
                text = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else None) or ""
                if not text:
                    continue
                chunks = self.text_splitter.split_text(text)
                for c in chunks:
                    md = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                    try:
                        if _LC_Doc:
                            out.append(_LC_Doc(page_content=c, metadata=md or {}))
                        else:
                            out.append({"page_content": c, "metadata": md or {}})
                    except Exception:
                        out.append({"page_content": c, "metadata": md or {}})
        else:
            # fallback: naive split by paragraphs
            for d in documents:
                text = getattr(d, "page_content", None) or (d.get("page_content") if isinstance(d, dict) else None) or ""
                if not text:
                    continue
                for para in text.split("\n\n"):
                    p = para.strip()
                    if not p:
                        continue
                    md = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
                    try:
                        if _LC_Doc:
                            out.append(_LC_Doc(page_content=p, metadata=md or {}))
                        else:
                            out.append({"page_content": p, "metadata": md or {}})
                    except Exception:
                        out.append({"page_content": p, "metadata": md or {}})
        return out

    def _docs_to_texts(self, documents: List[Any]) -> List[str]:
        texts = []
        for d in documents:
            text = ""
            if hasattr(d, "page_content"):
                text = getattr(d, "page_content") or ""
            elif isinstance(d, dict):
                text = d.get("page_content", "") or ""
            else:
                text = str(d)
            if text and text.strip():
                texts.append(text.strip())
        return texts

    def create_vector_store(self, documents: List[Any], persist_directory: str):
        # Validations
        if not documents:
            raise ValueError("No documents provided to create_vector_store(). Ensure your loader returned content.")
        texts = self._docs_to_texts(documents)
        if not texts:
            raise ValueError("Document chunks are empty after splitting. Check your document loader and splitter.")

        # Ensure embeddings object or fallback
        if self.embeddings is None:
            # try local HF fallback
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings()
                if st:
                    st.warning("OpenAIEmbeddings not configured — falling back to HuggingFaceEmbeddings for local testing.")
            except Exception:
                raise ImportError(
                    "Embeddings not available. Install langchain_openai (and set OPENAI_API_KEY) or install a local embedding provider (langchain_community HuggingFaceEmbeddings)."
                )

        # test embedding generation
        try:
            sample = texts[:2]
            emb_vecs = None
            if hasattr(self.embeddings, "embed_documents"):
                emb_vecs = self.embeddings.embed_documents(sample)
            elif hasattr(self.embeddings, "embed_query"):
                emb_vecs = [self.embeddings.embed_query(t) for t in sample]
            else:
                # try callable
                if callable(self.embeddings):
                    emb_vecs = self.embeddings(sample)
            if not emb_vecs or len(emb_vecs) == 0 or (isinstance(emb_vecs, list) and len(emb_vecs[0]) == 0):
                raise RuntimeError("Embeddings returned empty vectors for sample texts.")
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed. Check your embedding provider and API keys. Details: {e}")

        # create or load Chroma
        if _Chroma is None:
            raise ImportError("Chroma vectorstore not installed. Install langchain-community and chromadb.")
        os.makedirs(persist_directory, exist_ok=True)
        try:
            if os.listdir(persist_directory):
                vs = _Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            else:
                vs = _Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=persist_directory)
                try:
                    vs.persist()
                except Exception:
                    pass
            return vs
        except Exception as e:
            raise RuntimeError(f"Failed creating/loading vector store: {e}")

# -------------------------
# VoiceGenerator
# -------------------------
class VoiceGenerator:
    """
    Robust ElevenLabs wrapper that supports multiple SDK shapes.
    - Initialize with api_key (or environment ELEVEN_LABS_API_KEY).
    - load_voices() returns a list of available voice *names*.
    - generate_voice_response(text, voice_name_or_id) -> path to audio file (.mp3)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.available_voices: Dict[str, str] = {}
        self.default_voice: Optional[str] = None
        self._init_error: Optional[str] = None
        self._tts_call = None  # function(text, voice_id_or_name) -> bytes/iterable/file-like
        self.generate_fn = None

        resolved_key = (api_key or os.getenv("ELEVEN_LABS_API_KEY") or "").strip()
        if not resolved_key:
            self._init_error = "No ElevenLabs API key provided (env ELEVEN_LABS_API_KEY or UI)."
            return

        # Try modern ElevenLabs client shape first
        try:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=resolved_key)

            def _call_modern(text: str, voice: Optional[str]):
                # Try the variety of common modern signatures defensively
                # 1) client.text_to_speech.convert(voice_id=..., text=..., model_id=...)
                try:
                    if hasattr(client, "text_to_speech") and hasattr(client.text_to_speech, "convert"):
                        return client.text_to_speech.convert(voice_id=voice, text=text, model_id="eleven_turbo_v2_5")
                except TypeError:
                    try:
                        return client.text_to_speech.convert(voice=voice, text=text, model_id="eleven_turbo_v2_5")
                    except TypeError:
                        pass
                except Exception:
                    raise

                # 2) client.text_to_speech.generate or similar
                try:
                    if hasattr(client.text_to_speech, "generate"):
                        return client.text_to_speech.generate(voice_id=voice, input=text)
                except TypeError:
                    try:
                        return client.text_to_speech.generate(voice=voice, text=text)
                    except Exception:
                        pass

                # 3) client.generate(...) fallback
                try:
                    if hasattr(client, "generate"):
                        return client.generate(text=text, voice=voice)
                except TypeError:
                    try:
                        return client.generate(text, voice)
                    except Exception:
                        pass

                raise TypeError("No supported ElevenLabs TTS method signature found on ElevenLabs client.")

            # list voices if possible
            vlist = []
            try:
                if hasattr(client, "voices") and hasattr(client.voices, "get_all"):
                    resp = client.voices.get_all()
                    vlist = getattr(resp, "voices", resp) or []
                elif hasattr(client, "voices") and callable(getattr(client, "voices")):
                    vlist = client.voices() or []
                else:
                    vlist = []
            except Exception as e:
                # record but continue - _init_error kept for diagnostics
                self._init_error = f"Error listing voices (modern client): {e}"

            for v in vlist or []:
                vid = getattr(v, "voice_id", None) or getattr(v, "id", None) or getattr(v, "voiceId", None)
                name = getattr(v, "name", None) or getattr(v, "label", None) or (str(vid) if vid is not None else None)
                if not vid and isinstance(v, dict):
                    vid = v.get("voice_id") or v.get("id") or v.get("voiceId")
                    name = name or v.get("name") or v.get("label")
                if vid and name and name not in self.available_voices:
                    self.available_voices[name] = vid

            self._tts_call = _call_modern
            self.generate_fn = _call_modern

        except Exception as modern_exc:
            # try legacy path
            try:
                from elevenlabs import generate as legacy_generate, set_api_key as legacy_set_key, voices as legacy_voices
                try:
                    legacy_set_key(resolved_key)
                except Exception:
                    # ignore set_key errors; still try
                    pass

                def _call_legacy(text: str, voice: Optional[str]):
                    try:
                        return legacy_generate(text=text, voice=voice)
                    except TypeError:
                        try:
                            return legacy_generate(text, voice)
                        except Exception:
                            pass
                    try:
                        return legacy_generate(text=text, voice_name=voice)
                    except Exception:
                        pass
                    raise TypeError("Legacy ElevenLabs generate() has unexpected signature in this environment.")

                vlist = []
                try:
                    vlist = legacy_voices() if callable(legacy_voices) else list(legacy_voices)
                except Exception as e:
                    self._init_error = f"Error listing voices (legacy client): {e}"
                    vlist = []

                for v in vlist or []:
                    if isinstance(v, dict):
                        vid = v.get("voice_id") or v.get("id")
                        name = v.get("name") or vid
                    else:
                        vid = getattr(v, "voice_id", None) or getattr(v, "id", None) or None
                        name = getattr(v, "name", None) or str(vid)
                    if vid and name and name not in self.available_voices:
                        self.available_voices[name] = vid

                self._tts_call = _call_legacy
                self.generate_fn = _call_legacy

            except Exception as legacy_exc:
                self._init_error = f"ElevenLabs init failed (modern error: {modern_exc}; legacy error: {legacy_exc})"
                self._tts_call = None
                self.generate_fn = None
                self.available_voices = {}
                self.default_voice = None
                return

        if self.available_voices:
            self.default_voice = next(iter(self.available_voices.keys()))
        if not getattr(self, "_init_error", None):
            self._init_error = None

    def load_voices(self) -> List[str]:
        return list(self.available_voices.keys())

    def generate_voice_response(self, text: str, voice_name: Optional[str] = None) -> Optional[str]:
        if not self._tts_call:
            raise ImportError("ElevenLabs TTS client not configured or unsupported SDK shape. See _init_error for details.")

        # Resolve a voice id; prefer mapping name->id, allow id passed directly
        selected_id = None
        if voice_name:
            selected_id = self.available_voices.get(voice_name)
            if selected_id is None and voice_name in set(self.available_voices.values()):
                selected_id = voice_name

        if not selected_id and self.available_voices:
            selected_id = next(iter(self.available_voices.values()))

        if not selected_id:
            raise RuntimeError("No ElevenLabs voice id available. Ensure your account has at least one voice or provide a valid voice id.")

        # call TTS wrapper
        try:
            payload = self._tts_call(text, selected_id)
        except Exception as e:
            raise RuntimeError(f"ElevenLabs TTS failed during call: {e}. Init error: {getattr(self,'_init_error', None)}")

        # normalize payload
        try:
            if isinstance(payload, (bytes, bytearray)):
                audio_bytes = bytes(payload)
            elif hasattr(payload, "__iter__") and not isinstance(payload, (str, bytes, bytearray)):
                audio_bytes = b"".join([chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk) for chunk in payload])
            elif hasattr(payload, "read"):
                audio_bytes = payload.read()
                if isinstance(audio_bytes, str):
                    raise RuntimeError("TTS returned text instead of binary audio.")
                if isinstance(audio_bytes, memoryview):
                    audio_bytes = audio_bytes.tobytes()
            else:
                raise RuntimeError(f"Unexpected TTS payload type: {type(payload)}")
        except Exception as e:
            raise RuntimeError(f"Failed to normalize audio payload from ElevenLabs: {e}")

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            return tmp.name
        except Exception as e:
            raise RuntimeError(f"Failed to write TTS audio to temp file: {e}")

# -------------------------
# VoiceAssistantRAG
# -------------------------
class VoiceAssistantRAG:
    """
    Voice Assistant backed by Whisper (local) + LLM via LangChain wrappers (ChatOpenAI or other).
    Provides start_recording(), stop_recording(), transcribe_audio(), generate_response(), text_to_speech().
    """

    def __init__(self, elevenlabs_api_key: Optional[str] = None, whisper_model_name: str = "base"):
        # whisper model
        if _whisper is None:
            raise ImportError("openai-whisper is required for transcription. Install with: pip install openai-whisper")
        try:
            self.whisper_model = _whisper.load_model(whisper_model_name)
        except Exception:
            # fallback: try loading base if specified model missing
            self.whisper_model = _whisper.load_model("base")

        # embeddings / llm initialization is optional; VoiceAssistant will still do transcription
        try:
            from langchain_openai import ChatOpenAI as _ChatOpenAI
            self.llm = _ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        except Exception:
            self.llm = None

        try:
            from langchain_openai import OpenAIEmbeddings as _OE
            self.embeddings = _OE() if _OE else None
        except Exception:
            self.embeddings = None

        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100

        # recording related
        self._recording_stream = None
        self._recording_queue = None
        self._recording_chunks = None

        # TTS generator
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    # -----------------------
    # Recording helpers
    # -----------------------
    def start_recording(self):
        if _sd is None:
            raise ImportError("sounddevice is required for recording (pip install sounddevice).")

        # reset buffers
        import queue as _queue
        self._recording_queue = _queue.Queue()
        self._recording_chunks = []

        def _callback(indata, frames, time, status):
            if status:
                # keep a lightweight log
                print("SoundDevice status:", status)
            # Put a copy to avoid reused memory issues
            try:
                self._recording_queue.put(indata.copy())
            except Exception:
                self._recording_chunks.append(indata.copy())

        try:
            self._recording_stream = _sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=_callback)
            self._recording_stream.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start recording input stream: {e}")

    def stop_recording(self) -> str:
        if _sd is None:
            raise ImportError("sounddevice is required for recording (pip install sounddevice).")
        if not getattr(self, "_recording_stream", None):
            raise RuntimeError("Recording not started.")

        try:
            self._recording_stream.stop()
            self._recording_stream.close()
        except Exception:
            pass
        self._recording_stream = None

        frames = []
        try:
            while not self._recording_queue.empty():
                frames.append(self._recording_queue.get_nowait())
        except Exception:
            pass

        if getattr(self, "_recording_chunks", None):
            frames.extend(self._recording_chunks)

        if not frames:
            raise RuntimeError("No audio captured (frames empty).")

        try:
            import numpy as _np
        except Exception:
            raise RuntimeError("numpy is required for audio processing. Install numpy.")

        arr = _np.concatenate(frames, axis=0)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]

        # convert to float32 normalized
        if arr.dtype.kind in ("i", "u"):
            try:
                arr = arr.astype("float32") / float(_np.iinfo(arr.dtype).max)
            except Exception:
                arr = arr.astype("float32")
        else:
            arr = arr.astype("float32")

        maxv = float(_np.max(_np.abs(arr))) if arr.size else 0.0
        if maxv > 1.0:
            arr = arr / maxv

        # write to wav
        if _sf is None:
            raise RuntimeError("soundfile is required to save recording. Install with: pip install soundfile")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            _sf.write(tmp_path, arr, self.sample_rate, subtype="PCM_16")
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"Failed to write WAV: {e}")

        return tmp_path

    # -----------------------
    # Transcription & QA
    # -----------------------
    @staticmethod
    def _normalize_acronyms(text: str) -> str:
        if not text:
            return text
        t = text.strip().lower()
        if t in {"sequel", "s q l", "s. q. l.", "s q l.", "s.q.l", "ess cue el", "ess cue ell"}:
            return "SQL"
        tokens = t.replace(".", " ").replace(",", " ").split()
        if tokens in (["s", "q", "l"], ["s", "q", "el"]):
            return "SQL"
        if t == "gps":
            return "SQL"
        if len(t) <= 5 and all(ch.isalpha() or ch == " " for ch in t):
            if t.isalpha() and t.upper() == t or t in {"sql"}:
                return t.upper()
        return text.strip()

    def transcribe_audio(self, audio_array_or_path) -> str:
        try:
            if isinstance(audio_array_or_path, str) and os.path.exists(audio_array_or_path):
                audio_path = audio_array_or_path
            else:
                if _np is None:
                    raise RuntimeError("numpy is required to write temp audio file.")
                arr = audio_array_or_path
                if not isinstance(arr, _np.ndarray):
                    arr = _np.array(arr, dtype="float32")
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                maxv = float(_np.max(_np.abs(arr))) if arr.size else 0.0
                if maxv > 1.0:
                    arr = arr / maxv
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                if _sf is None:
                    raise RuntimeError("soundfile is required for audio I/O. Install soundfile.")
                _sf.write(tmp_path, arr, self.sample_rate, subtype="PCM_16")
                audio_path = tmp_path

            model = getattr(self, "whisper_model", None)
            if model is None:
                raise ImportError("Whisper model not loaded.")

            # Primary attempt
            try:
                kwargs = {"language": "en", "temperature": 0.0}
                try:
                    kwargs["beam_size"] = 5
                except Exception:
                    pass
                result = model.transcribe(audio_path, **kwargs)
            except TypeError:
                result = model.transcribe(audio_path, language="en", temperature=0.0)

            text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

            # If text very short, attempt medium fallback (best-effort)
            if len(text.split()) <= 3:
                try:
                    # only try if medium not already loaded
                    mname = getattr(model, "model_name", "") or ""
                    if not any(k in mname.lower() for k in ["medium", "large"]):
                        try:
                            med = _whisper.load_model("medium")
                            med_res = med.transcribe(audio_path, language="en", temperature=0.0, beam_size=5)
                            med_text = med_res.get("text", "").strip()
                            if med_text and med_text != text:
                                text = med_text
                        except Exception:
                            pass
                except Exception:
                    pass

            normalized = self._normalize_acronyms(text)

            # cleanup if we wrote temp file
            if not (isinstance(audio_array_or_path, str) and os.path.exists(audio_array_or_path)):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
            return normalized

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def setup_vector_store(self, vector_store):
        self.vector_store = vector_store
        # Build conversational retrieval chain if langchain available
        try:
            from langchain_classic.chains import ConversationalRetrievalChain as _CR
            from langchain_classic.memory import ConversationBufferMemory as _CBM
            memory = _CBM(memory_key="chat_history", return_messages=True)
            if not hasattr(self.vector_store, "as_retriever"):
                raise RuntimeError("Vector store doesn't provide retriever interface.")
            self.qa_chain = _CR.from_llm(llm=self.llm, retriever=self.vector_store.as_retriever(), memory=memory, verbose=True)
        except Exception:
            # gracefully degrade: QA chain may not be available
            self.qa_chain = None

    def generate_response(self, query: str) -> str:
        if not self.qa_chain:
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
        except Exception as e:
            raise RuntimeError(f"RAG query failed: {e}")

    def text_to_speech(self, text: str, voice_name: Optional[str] = None) -> Optional[str]:
        if not self.voice_generator or not self.voice_generator._tts_call:
            raise ImportError(
                "ElevenLabs TTS client not configured. To enable TTS, install 'elevenlabs' and set ELEVEN_LABS_API_KEY in your environment or pass key in UI."
            )
        # delegate to voice_generator (it resolves IDs and falls back to available voice)
        return self.voice_generator.generate_voice_response(text, voice_name)

# -------------------------
# Voice UI (Streamlit)
# -------------------------
class VoiceUI:
    
    def _safe_tts_playback(assistant, text, voice_choice, elevenlabs_key):
        """
        Attempts TTS playback. Suppresses the specific 'no voice id' error from appearing
        as a Streamlit error message. Other exceptions are shown.
        """
        # If no key, skip quietly
        if not (elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY")):
            # optional non-intrusive note (no st.error)
            return

        try:
            # Normalize voice choice passed (None means use default in generator)
            selected = None
            if voice_choice and voice_choice != "(none)":
                selected = voice_choice
            audio_out = None
            try:
                audio_out = assistant.text_to_speech(text, selected)
            except TypeError:
                # some older signatures might require single arg
                audio_out = assistant.text_to_speech(text)

            if audio_out:
                st.audio(audio_out)
                try:
                    os.unlink(audio_out)
                except Exception:
                    pass
        except Exception as e:
            msg = str(e).lower()
            # Detect common "no voice id" / "voice not found" patterns and suppress them
            no_voice_patterns = [
                "no elevenlabs voice id available",
                "voice_not_found",
                "voice not found",
                "a voice with the voice_id none",
                "voice_id none",
                "voice id none",
                "no voice id",
            ]
            if any(p in msg for p in no_voice_patterns):
                # Quietly skip showing an error. If debug toggled, show a sidebar info.
                if st.sidebar.checkbox("Show debug info"):
                    st.sidebar.info("TTS skipped: no ElevenLabs voice available in the configured account.")
                # otherwise stay silent (no st.error)
                return
            # Otherwise show the error (unexpected)
            st.error(f"TTS failed: {e}")

    
    
    @staticmethod
    def render(api_key, provider, model, elevenlabs_key=None):
        if st is None:
            raise RuntimeError("Streamlit (st) is required for UI rendering.")

        st.header("Voice Assistant (RAG)")
        st.subheader("1) Setup Knowledge Base")
        uploaded_files = st.file_uploader("Upload documents (.pdf/.txt/.md)", accept_multiple_files=True, type=["pdf", "txt", "md"])
        persist_dir = st.text_input("Vector store persist directory", value="knowledge_base")

        if st.button("Process & Index Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                tmpdir = tempfile.mkdtemp()
                try:
                    for f in uploaded_files:
                        path = os.path.join(tmpdir, f.name)
                        with open(path, "wb") as fh:
                            fh.write(f.getbuffer())
                    proc = DocumentProcessor()
                    docs = proc.load_documents(tmpdir)
                    if not docs:
                        st.error("No loadable documents found. Supported: .txt, .pdf, .md")
                    else:
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

        # (re)create assistant when necessary or when key changed
        recreate = False
        if "voice_assistant" not in st.session_state:
            recreate = True
        else:
            # if elevenlabs_key provided in UI changed vs cached assistant, recreate
            cached_key = getattr(st.session_state.get("voice_assistant"), "voice_generator", None)
            # simpler approach: user can clear assistant if needed
        if recreate:
            try:
                st.session_state.voice_assistant = VoiceAssistantRAG(elevenlabs_key)
                st.session_state.voice_assistant.setup_vector_store(st.session_state.vector_store)
            except Exception as e:
                st.error(f"Failed to create voice assistant: {e}")
                if st.sidebar.checkbox("Show traceback"):
                    st.code(traceback.format_exc())
                return

        assistant: VoiceAssistantRAG = st.session_state.voice_assistant

        # debug info toggle
        if st.sidebar.checkbox("Show debug info"):
            st.sidebar.write("ELEVEN_LABS_API_KEY present:", bool(elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY")))
            vg = getattr(assistant, "voice_generator", None)
            tts_ready = bool(vg and getattr(vg, "_tts_call", None))
            st.sidebar.write("TTS client available:", tts_ready)
            if vg:
                st.sidebar.write("TTS init error:", getattr(vg, "_init_error", None))
                st.sidebar.write("TTS voices count:", len(vg.load_voices()))
                st.sidebar.write("TTS voices mapping (first 10):", dict(list(vg.available_voices.items())[:10]))

        op = st.radio("Input mode", ["Record (manual Start/Stop)", "Upload audio file", "Type text question"])

        if op == "Record (manual Start/Stop)":
            # TTS voice selection (if available)
            voice_choice = None
            try:
                avail = assistant.voice_generator.load_voices() if getattr(assistant, "voice_generator", None) else []
            except Exception:
                avail = []
            if avail:
                voice_choice = st.selectbox("TTS voice (optional)", ["(none)"] + avail, index=0)
            else:
                st.info("No ElevenLabs voices available. Provide valid key in the sidebar or add a voice in your ElevenLabs account. TTS will be skipped.")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Start Recording", key="start_record"):
                    try:
                        if not hasattr(assistant, "start_recording"):
                            st.error("Recording not supported by assistant implementation.")
                        else:
                            assistant.start_recording()
                            st.session_state.is_recording = True
                            st.success("Recording started. Click Stop Recording when done.")
                    except Exception as e:
                        st.error(f"Start recording failed: {e}")

            with col2:
                if st.button("Stop Recording", key="stop_record"):
                    if not st.session_state.get("is_recording", False):
                        st.warning("Recording not started.")
                    else:
                        try:
                            wav_path = assistant.stop_recording()
                            if isinstance(wav_path, str) and os.path.exists(wav_path):
                                st.session_state.last_audio_file = wav_path
                                st.success(f"Recording saved: {os.path.basename(wav_path)}")
                            else:
                                st.session_state.last_audio = wav_path
                                st.success("Recording captured (in-memory).")
                            st.session_state.is_recording = False
                        except Exception as e:
                            st.error(f"Stop recording failed: {e}")

            if st.session_state.get("last_audio_file") or st.session_state.get("last_audio"):
                st.markdown("**Recorded audio ready**")
                if st.session_state.get("last_audio_file"):
                    st.audio(st.session_state.last_audio_file)
                else:
                    try:
                        arr = st.session_state.get("last_audio")
                        if isinstance(arr, (list, tuple)):
                            arr = _np.array(arr, dtype="float32")
                        tmpplay = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        _sf.write(tmpplay.name, arr, assistant.sample_rate, subtype="PCM_16")
                        st.audio(tmpplay.name)
                        st.session_state._temp_playback = tmpplay.name
                    except Exception:
                        pass

                if st.button("Transcribe & Ask"):
                    try:
                        if st.session_state.get("last_audio_file"):
                            source = st.session_state.last_audio_file
                            transcript = assistant.transcribe_audio(source)
                        else:
                            transcript = assistant.transcribe_audio(st.session_state.last_audio)
                        st.write("You said:", transcript)
                        answer = assistant.generate_response(transcript)
                        st.markdown("### Answer")
                        st.write(answer)

                        # TTS playback (if configured)
                        try:
                            if avail and (voice_choice and voice_choice != "(none)"):
                                tts_voice = voice_choice
                            else:
                                tts_voice = None
                            audio_out = None
                            try:
                                audio_out = assistant.text_to_speech(answer, tts_voice) if (elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY")) else None
                            except ImportError as ie:
                                st.info(str(ie))
                            if audio_out:
                                st.audio(audio_out)
                                try:
                                    os.unlink(audio_out)
                                except Exception:
                                    pass
                            else:
                                if not (elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY")):
                                    st.info("ElevenLabs key not provided — TTS skipped.")
                                elif not avail:
                                    st.info("No ElevenLabs voices available in account — TTS skipped.")
                                else:
                                    st.info("TTS did not produce audio (check ElevenLabs client or account).")
                        except Exception as e:
                            st.error(f"TTS failed: {e}")

                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        if st.sidebar.checkbox("Show traceback"):
                            st.code(traceback.format_exc())

        elif op == "Upload audio file":
            audio_file = st.file_uploader("Upload audio (.wav/.mp3/.m4a)", type=["wav", "mp3", "m4a"])
            if audio_file and st.button("Use uploaded audio"):
                tmp = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{audio_file.name}")
                with open(tmp, "wb") as fh:
                    fh.write(audio_file.getbuffer())
                st.session_state.last_audio_file = tmp
                st.success("Uploaded audio saved.")

            if st.session_state.get("last_audio_file"):
                st.audio(st.session_state.last_audio_file)
                if st.button("Transcribe & Ask Uploaded"):
                    try:
                        if hasattr(assistant, "whisper_model") and assistant.whisper_model:
                            res = assistant.whisper_model.transcribe(st.session_state.last_audio_file)
                            transcript = res.get("text", "") if isinstance(res, dict) else str(res)
                        else:
                            arr, sr = _sf.read(st.session_state.last_audio_file)
                            transcript = assistant.transcribe_audio(arr)
                        st.write("You said:", transcript)
                        answer = assistant.generate_response(transcript)
                        st.write("Answer:", answer)
                        try:
                            VoiceUI._safe_tts_playback(assistant, answer, tts_voice, elevenlabs_key)
                        except ImportError as ie:
                            st.info(str(ie))
                        except Exception as e:
                            st.error(f"TTS failed: {e}")
                    except Exception as e:
                        st.error(f"Uploaded audio processing failed: {e}")
                        if st.sidebar.checkbox("Show traceback"):
                            st.code(traceback.format_exc())

        else:  # Type text question
            text_query = st.text_input("Type your question")
            if st.button("Ask (text)"):
                if not text_query or not text_query.strip():
                    st.warning("Please type a question.")
                else:
                    try:
                        answer = assistant.generate_response(text_query)
                        st.write("Answer:", answer)
                        try:
                            audio_out = assistant.text_to_speech(answer) if (elevenlabs_key or os.getenv("ELEVEN_LABS_API_KEY")) else None
                            if audio_out:
                                st.audio(audio_out)
                                try:
                                    os.unlink(audio_out)
                                except Exception:
                                    pass
                        except ImportError as ie:
                            st.info(str(ie))
                        except Exception as e:
                            st.error(f"TTS failed: {e}")
                    except Exception as e:
                        st.error(f"Text query failed: {e}")
                        if st.sidebar.checkbox("Show traceback"):
                            st.code(traceback.format_exc())
