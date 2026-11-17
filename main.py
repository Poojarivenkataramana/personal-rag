import hashlib
import io
import json
import os
import re

import chromadb
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# OCR is optional: if these packages are missing, the app still works without it.
try:
    import pypdfium2 as pdfium
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------- settings ----------------
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # tried in order
GROQ_URL = "https://api.groq.com/openai/v1"

CHUNK_SIZE = 1000       # target characters per chunk
CHUNK_OVERLAP = 150     # characters repeated between neighbouring chunks
CANDIDATES = 15         # chunks gathered by hybrid search before reranking
TOP_K = 5               # chunks finally sent to the model
MEMORY_MESSAGES = 6     # recent messages the model remembers
MIN_PAGE_CHARS = 30     # PDF pages with less text than this are treated as scanned
HISTORY_FILE = "chat_history.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's documents. "
    "The latest user message contains context passages from the documents. "
    "Answer using ONLY that context. If the answer isn't there, say you couldn't "
    "find it in the documents. Earlier messages are only for understanding what the "
    "user is referring to. Be clear and concise."
)

st.set_page_config(page_title="Personal RAG", page_icon="📚")


# ---------------- cached resources ----------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection("docs", embedding_function=embedder)


@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def get_ocr():
    return RapidOCR()


@st.cache_resource
def build_bm25(key: str, _documents):
    # `key` changes whenever the stored chunks change, which rebuilds the index
    return BM25Okapi([tokenize(d) for d in _documents])


collection = get_collection()


def default_api_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


# ---------------- reading sources ----------------
def ocr_image(img) -> str:
    result, _ = get_ocr()(np.array(img.convert("RGB")))
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def read_pdf(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages, pdf_doc = [], None
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if len(text.strip()) < MIN_PAGE_CHARS and OCR_AVAILABLE:  # scanned page
            if pdf_doc is None:
                pdf_doc = pdfium.PdfDocument(data)
            text = ocr_image(pdf_doc[i].render(scale=2).to_pil())
        pages.append((i + 1, text))
    return pages


def read_upload(uploaded) -> list[tuple[int, str]]:
    """Returns [(page_number, text)]. Page 0 means the source has no pages."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        return read_pdf(data)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return [(0, "\n\n".join(p.text for p in doc.paragraphs))]
    if name.endswith((".png", ".jpg", ".jpeg")):
        if not OCR_AVAILABLE:
            raise RuntimeError("Image OCR needs: pip install rapidocr-onnxruntime pypdfium2")
        return [(0, ocr_image(Image.open(io.BytesIO(data))))]
    return [(0, data.decode("utf-8", errors="ignore"))]


def read_url(url: str) -> list[tuple[int, str]]:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (personal-rag)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    parts = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"])
    ]
    return [(0, "\n\n".join(p for p in parts if p))]


# ---------------- smart chunking ----------------
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_units(text: str) -> list[str]:
    """Break text into paragraph-sized units; headings stay attached to what follows."""
    units, heading = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        looks_like_heading = len(para) < 80 and not para.endswith((".", "!", "?", ":", ","))
        if looks_like_heading:
            heading = f"{heading} {para}".strip()
            continue
        if heading:
            para, heading = f"{heading}\n{para}", ""
        if len(para) <= CHUNK_SIZE:
            units.append(para)
            continue
        for sentence in SENTENCE_SPLIT.split(para):  # long paragraph -> sentences
            while len(sentence) > CHUNK_SIZE:  # a single giant sentence: hard cut
                units.append(sentence[:CHUNK_SIZE])
                sentence = sentence[CHUNK_SIZE:]
            if sentence:
                units.append(sentence)
    if heading:
        units.append(heading)
    return units


def chunk_text(text: str) -> list[str]:
    chunks, current = [], ""
    for unit in split_units(text):
        if current and len(current) + len(unit) + 1 > CHUNK_SIZE:
            chunks.append(current)
            tail = current[-CHUNK_OVERLAP:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail  # start at a word
            current = f"{tail}\n{unit}"
        else:
            current = f"{current}\n{unit}" if current else unit
    if current.strip():
        chunks.append(current)
    return chunks


# ---------------- indexing + managing documents ----------------
def add_document(source: str, pages: list[tuple[int, str]]) -> int:
    collection.delete(where={"source": source})  # replace any older version
    file_id = hashlib.md5(source.encode()).hexdigest()[:10]
    ids, docs, metas = [], [], []
    for page, text in pages:
        for chunk in chunk_text(text):
            ids.append(f"{file_id}-{len(ids)}")
            docs.append(chunk)
            metas.append({"source": source, "page": page, "chunk": len(metas)})
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def list_documents() -> dict[str, dict]:
    data = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for m in data["metadatas"]:
        d = docs.setdefault(m["source"], {"chunks": 0, "pages": 0})
        d["chunks"] += 1
        d["pages"] = max(d["pages"], m.get("page", 0))
    return docs


# ---------------- search: hybrid (meaning + keywords), then rerank ----------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def retrieve(query: str, advanced: bool):
    if not advanced:
        res = collection.query(query_texts=[query], n_results=TOP_K)
        return res["documents"][0], res["metadatas"][0]

    data = collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    lookup = {i: (d, m) for i, d, m in zip(ids, docs, metas)}

    # 1) meaning-based search
    vec = collection.query(query_texts=[query], n_results=min(CANDIDATES, len(ids)))
    vec_ids = vec["ids"][0]

    # 2) keyword search (BM25)
    key = hashlib.md5("".join(sorted(ids)).encode()).hexdigest()
    scores = build_bm25(key, docs).get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:CANDIDATES]
    kw_ids = [ids[i] for i in order if scores[i] > 0]

    # 3) merge both rankings (reciprocal rank fusion)
    fused: dict[str, float] = {}
    for ranking in (vec_ids, kw_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    candidates = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]

    # 4) rerank the candidates with a cross-encoder
    try:
        rerank_scores = get_reranker().predict([(query, lookup[c][0]) for c in candidates])
        candidates = [c for _, c in sorted(zip(rerank_scores, candidates), reverse=True)]
    except Exception:
        pass  # if the reranker can't load, keep the merged order
    top = candidates[:TOP_K]
    return [lookup[c][0] for c in top], [lookup[c][1] for c in top]


def source_label(m) -> str:
    return m["source"] + (f", page {m['page']}" if m.get("page", 0) else "")


def format_sources(metas) -> str:
    pages: dict[str, set[int]] = {}
    for m in metas:
        pages.setdefault(m["source"], set())
        if m.get("page", 0):
            pages[m["source"]].add(m["page"])
    parts = []
    for source, pgs in sorted(pages.items()):
        parts.append(
            f"{source} (p. {', '.join(str(p) for p in sorted(pgs))})" if pgs else source
        )
    return ", ".join(parts)


def build_prompt(question: str, chunks, metas) -> str:
    context = "\n\n".join(f"[Source: {source_label(m)}]\n{c}" for c, m in zip(chunks, metas))
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


# ---------------- LLM calls ----------------
def create_completion(client: OpenAI, messages, stream: bool = False):
    last_error = None
    for model in MODELS:
        try:
            return client.chat.completions.create(model=model, messages=messages, stream=stream)
        except Exception as e:  # try the next model
            last_error = e
    raise last_error


def condense_question(client: OpenAI, history, question: str) -> str:
    """Turn a follow-up like 'explain that simpler' into a standalone question."""
    if not history:
        return question
    transcript = "\n".join(f"{role}: {text}" for role, text in history[-MEMORY_MESSAGES:])
    prompt = (
        "Rewrite the follow-up question as a standalone question that makes sense "
        "without the conversation. Keep it short. If it is already standalone, return "
        "it unchanged. Return ONLY the question.\n\n"
        f"Conversation:\n{transcript}\n\nFollow-up question: {question}"
    )
    try:
        resp = create_completion(client, [{"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip() or question
    except Exception:
        return question


def stream_answer(client: OpenAI, history, prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, text in history[-MEMORY_MESSAGES:]:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": prompt})
    for part in create_completion(client, messages, stream=True):
        if part.choices and part.choices[0].delta.content:
            yield part.choices[0].delta.content


# ---------------- saved chat ----------------
def load_history() -> list[tuple[str, str]]:
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return [tuple(x) for x in json.load(f)]
    except Exception:
        return []


def save_history(history) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass


def history_as_markdown(history) -> str:
    return "\n\n".join(
        f"**{'You' if role == 'user' else 'Assistant'}:** {text}" for role, text in history
    )


if "history" not in st.session_state:
    st.session_state.history = load_history()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Groq API key")
    api_key = st.text_input("Paste your key", default_api_key(), type="password")

    advanced = st.toggle(
        "Advanced search (hybrid + rerank)",
        value=True,
        help="Slower but finds better passages. Turn off for plain meaning-based search.",
    )

    st.divider()
    st.header("Add documents")
    types = ["pdf", "docx", "txt", "md"] + (["png", "jpg", "jpeg"] if OCR_AVAILABLE else [])
    files = st.file_uploader("Upload files", type=types, accept_multiple_files=True)
    if st.button("Index files", disabled=not files):
        for f in files:
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    n = add_document(f.name, read_upload(f))
                if n:
                    st.success(f"{f.name}: {n} chunks")
                else:
                    hint = "" if OCR_AVAILABLE else " (scanned? OCR isn't installed)"
                    st.warning(f"{f.name}: no text found{hint}")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    if not OCR_AVAILABLE:
        st.caption("OCR not installed: scanned PDFs and images are skipped.")

    url = st.text_input("Or add a web page", placeholder="https://...")
    if st.button("Index web page", disabled=not url.strip()):
        try:
            with st.spinner("Fetching page..."):
                n = add_document(url.strip(), read_url(url.strip()))
            st.success(f"{n} chunks added") if n else st.warning("No text found on that page")
        except Exception as e:
            st.error(f"Could not read that page: {e}")

    st.divider()
    st.header("Your files")
    documents = list_documents()
    if not documents:
        st.caption("Nothing indexed yet.")
    for name, info in documents.items():
        left, right = st.columns([5, 1])
        short = name if len(name) <= 32 else name[:29] + "..."
        extra = f"{info['pages']} pages · " if info["pages"] else ""
        left.markdown(f"**{short}**  \n{extra}{info['chunks']} chunks")
        if right.button("✕", key=f"del-{name}", help=f"Delete {name}"):
            collection.delete(where={"source": name})
            st.rerun()

    st.divider()
    st.header("Chat")
    if st.button("New chat"):
        st.session_state.history = []
        save_history([])
        st.rerun()
    st.download_button(
        "Download chat",
        data=history_as_markdown(st.session_state.history),
        file_name="chat.md",
        mime="text/markdown",
        disabled=not st.session_state.history,
    )

# ---------------- chat ----------------
st.title("📚 Personal RAG")
st.caption("Add documents on the left, then ask questions about them.")

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Ask something about your documents")
if question:
    prior = list(st.session_state.history)
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if collection.count() == 0:
            answer = "Please add and index some documents first."
            st.markdown(answer)
        elif not api_key:
            answer = "Please paste your Groq API key in the sidebar."
            st.markdown(answer)
        else:
            try:
                client = OpenAI(base_url=GROQ_URL, api_key=api_key)
                search_query = condense_question(client, prior, question)
                if search_query != question:
                    st.caption(f"Searching for: {search_query}")
                with st.spinner("Searching your documents..."):
                    chunks, metas = retrieve(search_query, advanced)
                prompt = build_prompt(question, chunks, metas)
                answer = st.write_stream(stream_answer(client, prior, prompt))
                st.caption("Sources: " + format_sources(metas))
                with st.expander("Retrieved passages"):
                    for c, m in zip(chunks, metas):
                        st.markdown(f"**{source_label(m)}**")
                        st.text(c)
            except Exception as e:
                answer = f"Could not get an answer: {e}"
                st.error(answer)
    st.session_state.history.append(("assistant", answer))
    save_history(st.session_state.history)
