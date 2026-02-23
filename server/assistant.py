#!/usr/bin/env python3
import argparse
import difflib
import json
import math
import os
import re
import sys
import time
import html
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

CONFIG_DEFAULTS = {
    "vault_path": "/path/to/YourVault",
    "papers_path": "/path/to/YourPapers",
    "index_dir": "index",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "max_context_chunks": 6,
    "max_file_mb": 20,
    "max_response_tokens": 128,
    "read_only": False,
    "mode": "safe",
    "allow_edit": True,
    "allow_terminal": False,
    "allow_python": False,
    "allow_self_improve": False,
    "enable_web": False,
    "auto_sync": True,
    "sync_interval_sec": 120,
    "sync_idle_sec": 300,
    "model_context_tokens": 4096,
    "title_boost": 1.5,
    "mmr_lambda": 0.7,
    "mmr_candidates": 30,
    "max_chunks_per_file": 1,
    "chat_backend": "llamacpp",  # llamacpp or ollama
    "api_base": "http://127.0.0.1:1234/v1",
    "api_key": "",
    "ollama_url": "http://localhost:11434",
    "model": "obsidian-qwen2.5-3b",
    "retrieval_backend": "bm25",  # bm25 only (fast, stable)
}

SYSTEM_PROMPT = (
    "You are an Obsidian vault assistant for a physicist named Bev. You provide summaries, help retrieve and recall information, "
    "and can look things up online when enabled. You also help with programming and computer subjects. "
    "Use only the provided context and cite sources as [1], [2], etc. If the context is insufficient, "
    "say so and ask a clarifying question. Never claim to have edited or deleted any files. "
    "Do not edit or add anything without explicit permission; if the user asks to modify notes, "
    "propose changes and ask for approval. You are encouraged to suggest additional helpful next steps. "
    "Be warm, concise, and personable in tone. For social questions (e.g., “how are you?”), respond briefly and friendly "
    "before offering help."
)

VAULT_EXTS = {".md", ".markdown", ".txt", ".csv", ".tsv"}
PAPER_EXTS = {".pdf", ".md", ".txt"}
TABLE_EXTS = {".csv", ".tsv"}
EDIT_EXTS = {".md", ".markdown", ".txt"}
EXCLUDE_DIRS = {".git", ".obsidian", "node_modules", "__pycache__"}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
GREET_RE = re.compile(r"^(hi|hello|hey|yo|thanks|thank you|ok|okay|test)$", re.I)
SUMMARY_RE = re.compile(r"\\b(su[m]{1,2}mar(?:ize|ise|y)|summrize|summry)\\b", re.I)
LATEST_HINT_RE = re.compile(r"\\b(latest|newest|most recent|last|recent)\\b", re.I)
NOTE_WORD_RE = re.compile(r"\\b(note|file|doc|document)\\b", re.I)
LATEST_NOTE_RE = re.compile(r"\\b(latest|newest|most recent|last|recent)\\b.*\\b(note|file|doc|document)\\b", re.I)
NOTE_PATH_RE = re.compile(r"(?:su[m]{1,2}mar(?:ize|ise|y)|summrize|summry).*?(?:note|file|doc|document)\\s*[:=]?\\s*(.+)", re.I)
NOTE_PATH_EXT_RE = re.compile(r"(?:\"([^\"]+\\.(?:md|markdown|txt))\"|'([^']+\\.(?:md|markdown|txt))'|([^\\s]+\\.(?:md|markdown|txt)))", re.I)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CONFIG_PATH = HERE / "config.json"


@dataclass
class Config:
    vault_path: Path
    papers_path: Path
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    max_context_chunks: int
    max_file_mb: int
    max_response_tokens: int
    read_only: bool
    mode: str
    allow_edit: bool
    allow_terminal: bool
    allow_python: bool
    allow_self_improve: bool
    enable_web: bool
    auto_sync: bool
    sync_interval_sec: int
    sync_idle_sec: int
    model_context_tokens: int
    title_boost: float
    mmr_lambda: float
    mmr_candidates: int
    max_chunks_per_file: int
    chat_backend: str
    api_base: str
    api_key: str
    ollama_url: str
    model: str
    retrieval_backend: str


def _resolve_path(value: str, base: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(path: Path) -> Config:
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    merged = {**CONFIG_DEFAULTS, **data}
    return Config(
        vault_path=_resolve_path(merged["vault_path"], REPO_ROOT),
        papers_path=_resolve_path(merged["papers_path"], REPO_ROOT),
        index_dir=_resolve_path(merged["index_dir"], REPO_ROOT),
        chunk_size=int(merged["chunk_size"]),
        chunk_overlap=int(merged["chunk_overlap"]),
        max_context_chunks=int(merged["max_context_chunks"]),
        max_file_mb=int(merged["max_file_mb"]),
        max_response_tokens=int(merged["max_response_tokens"]),
        read_only=bool(merged["read_only"]),
        mode=str(merged["mode"]),
        allow_edit=bool(merged["allow_edit"]),
        allow_terminal=bool(merged["allow_terminal"]),
        allow_python=bool(merged["allow_python"]),
        allow_self_improve=bool(merged["allow_self_improve"]),
        enable_web=bool(merged["enable_web"]),
        auto_sync=bool(merged["auto_sync"]),
        sync_interval_sec=int(merged["sync_interval_sec"]),
        sync_idle_sec=int(merged["sync_idle_sec"]),
        model_context_tokens=int(merged["model_context_tokens"]),
        title_boost=float(merged["title_boost"]),
        mmr_lambda=float(merged["mmr_lambda"]),
        mmr_candidates=int(merged["mmr_candidates"]),
        max_chunks_per_file=int(merged["max_chunks_per_file"]),
        chat_backend=str(merged["chat_backend"]),
        api_base=str(merged["api_base"]),
        api_key=str(merged["api_key"]),
        ollama_url=str(merged["ollama_url"]),
        model=str(merged["model"]),
        retrieval_backend=str(merged["retrieval_backend"]),
    )


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 300, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e


def _max_input_chars(cfg: Config) -> int:
    # conservative estimate to avoid context_length_exceeded
    tokens = max(int(cfg.model_context_tokens) - int(cfg.max_response_tokens) - 1024, 512)
    return max(tokens * 3, 1500)


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _fetch_url(url: str, timeout: int = 20) -> str:
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\\s+", " ", text).strip()


def _clean_ddg_url(url: str) -> str:
    if "duckduckgo.com/l/?" in url:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    return url


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    html_text = _fetch_url(url)

    results: List[Dict[str, str]] = []
    link_re = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
    snippet_re = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.I | re.S)

    links = link_re.findall(html_text)
    snippets = [ _strip_html(s) for s in snippet_re.findall(html_text) ]
    for i, (href, title_html) in enumerate(links):
        if len(results) >= max_results:
            break
        title = _strip_html(title_html)
        href = _clean_ddg_url(href)
        snippet = snippets[i] if i < len(snippets) else ""
        if not title or not href:
            continue
        results.append({"title": title, "url": href, "snippet": snippet})
    return results


def _openai_chat(
    messages: List[Dict[str, str]], model: str, api_base: str, api_key: str, max_tokens: int
) -> str:
    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = _post_json(url, payload, headers=headers)
    choices = resp.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "")


def _ollama_chat(messages: List[Dict[str, str]], model: str, ollama_url: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": max_tokens},
    }
    resp = _post_json(f"{ollama_url.rstrip('/')}/api/chat", payload)
    msg = resp.get("message", {})
    return msg.get("content", "")


def _chat(cfg: Config, messages: List[Dict[str, str]]) -> str:
    if cfg.chat_backend == "ollama":
        return _ollama_chat(messages, cfg.model, cfg.ollama_url, cfg.max_response_tokens)
    return _openai_chat(messages, cfg.model, cfg.api_base, cfg.api_key, cfg.max_response_tokens)


def _iter_files(root: Path, exts: set, max_mb: int) -> Iterable[Path]:
    if not root.exists():
        return []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if exts and path.suffix.lower() not in exts:
                continue
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
            except OSError:
                continue
            if size_mb > max_mb:
                continue
            yield path


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _files_meta_path(cfg: Config) -> Path:
    return cfg.index_dir / "files.json"


def _load_files_meta(cfg: Config) -> Dict[str, Dict[str, Any]]:
    path = _files_meta_path(cfg)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_files_meta(cfg: Config, meta: Dict[str, Dict[str, Any]]) -> None:
    path = _files_meta_path(cfg)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf not installed. Install it to read PDFs.")
        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            step = max_chars - overlap if max_chars > overlap else max_chars
            for i in range(0, len(p), step):
                chunk = p[i : i + max_chars]
                if chunk:
                    chunks.append(chunk)
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip()
            continue
        if buf:
            chunks.append(buf)
            if overlap > 0 and len(buf) > overlap:
                buf = (buf[-overlap:] + "\n\n" + p).strip()
                if len(buf) > max_chars:
                    chunks.append(buf[:max_chars])
                    buf = ""
            else:
                buf = p
        else:
            step = max_chars - overlap if max_chars > overlap else max_chars
            for i in range(0, len(p), step):
                chunks.append(p[i : i + max_chars])
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _build_bm25(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs: List[Dict[str, Any]] = []
    df: Dict[str, int] = {}
    total_len = 0
    for c in chunks:
        tokens = _tokenize(c["text"])
        title_tokens = _tokenize(Path(c["path"]).stem)
        total_len += len(tokens)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf.keys():
            df[t] = df.get(t, 0) + 1
        docs.append({
            "path": c["path"],
            "source": c["source"],
            "text": c["text"],
            "tf": tf,
            "len": len(tokens),
            "title_tokens": title_tokens,
        })
    avgdl = total_len / max(len(docs), 1)
    return {"docs": docs, "df": df, "avgdl": avgdl}


def _find_latest_note(cfg: Config) -> Optional[Path]:
    latest: Optional[Tuple[int, Path]] = None
    for path in _iter_files(cfg.vault_path, VAULT_EXTS, cfg.max_file_mb):
        try:
            mtime = int(path.stat().st_mtime_ns)
        except OSError:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, path)
    return latest[1] if latest else None


def _resolve_note_path_from_query(cfg: Config, query: str) -> Optional[Path]:
    raw = ""
    m = NOTE_PATH_RE.search(query)
    if m:
        raw = m.group(1).strip().strip("\"' ")
    if not raw:
        m2 = NOTE_PATH_EXT_RE.search(query)
        if m2:
            raw = next((g for g in m2.groups() if g), "").strip().strip("\"' ")
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (cfg.vault_path / p).resolve()
    if not str(p).startswith(str(cfg.vault_path.resolve())):
        return None
    if p.suffix.lower() not in EDIT_EXTS:
        return None
    return p if p.exists() else None


def _summarize_text(cfg: Config, text: str, instruction: str) -> str:
    max_chars = _max_input_chars(cfg)
    text = _truncate_text(text, max_chars)
    prompt = (
        "Summarize the following note. If asked for a specific part, only summarize that part. "
        "Be concise and actionable.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Note:\n{text}"
    )
    messages = [
        {"role": "system", "content": "You are a careful summarizer."},
        {"role": "user", "content": prompt},
    ]
    try:
        return _chat(cfg, messages)
    except RuntimeError as e:
        msg = str(e)
        if "context_length_exceeded" not in msg and "maximum context length" not in msg:
            raise
        # retry with more aggressive truncation
        text = _truncate_text(text, max(max_chars // 2, 1000))
        prompt = (
            "Summarize the following note. If asked for a specific part, only summarize that part. "
            "Be concise and actionable.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Note:\n{text}"
        )
        messages = [
            {"role": "system", "content": "You are a careful summarizer."},
            {"role": "user", "content": prompt},
        ]
        return _chat(cfg, messages)


def _iter_index_files(cfg: Config) -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    for path in _iter_files(cfg.vault_path, VAULT_EXTS, cfg.max_file_mb):
        files.append((path, "vault"))
    for path in _iter_files(cfg.papers_path, PAPER_EXTS, cfg.max_file_mb):
        files.append((path, "papers"))
    return files


def build_index(cfg: Config) -> None:
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[Dict[str, Any]] = []
    files_meta: Dict[str, Dict[str, Any]] = {}

    def add_files(root: Path, exts: set, source: str) -> None:
        for path in _iter_files(root, exts, cfg.max_file_mb):
            try:
                fp = _file_fingerprint(path)
            except Exception:
                continue
            files_meta[str(path)] = {**fp, "source": source}
            try:
                text = _read_text(path)
            except Exception as e:
                print(f"[skip] {path} ({e})")
                continue
            text = text.strip()
            if not text:
                continue
            for chunk in _chunk_text(text, cfg.chunk_size, cfg.chunk_overlap):
                if not chunk.strip():
                    continue
                chunks.append({
                    "path": str(path),
                    "source": source,
                    "text": chunk,
                })

    print("Indexing vault...")
    add_files(cfg.vault_path, VAULT_EXTS, "vault")
    print("Indexing papers...")
    add_files(cfg.papers_path, PAPER_EXTS, "papers")

    if not chunks:
        raise RuntimeError("No chunks created. Check your paths and file types.")

    print(f"Building BM25 index for {len(chunks)} chunks...")
    bm25 = _build_bm25(chunks)

    index_path = cfg.index_dir / "bm25.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(bm25, f, ensure_ascii=True)

    meta_path = cfg.index_dir / "meta.json"
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg.model,
        "retrieval_backend": cfg.retrieval_backend,
        "num_chunks": len(chunks),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    _save_files_meta(cfg, files_meta)


def load_index(cfg: Config) -> Dict[str, Any]:
    index_path = cfg.index_dir / "bm25.json"
    if not index_path.exists():
        raise RuntimeError("Index not found")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sync_index(cfg: Config) -> Dict[str, Any]:
    try:
        idx = load_index(cfg)
    except Exception:
        build_index(cfg)
        return load_index(cfg)

    files_meta = _load_files_meta(cfg)
    current_files = _iter_index_files(cfg)
    current_meta: Dict[str, Dict[str, Any]] = {}
    changed = False

    for path, source in current_files:
        try:
            fp = _file_fingerprint(path)
        except Exception:
            continue
        fp["source"] = source
        current_meta[str(path)] = fp
        old = files_meta.get(str(path))
        if not old or old.get("mtime_ns") != fp["mtime_ns"] or old.get("size") != fp["size"]:
            changed = True

    if set(files_meta.keys()) != set(current_meta.keys()):
        changed = True

    if not changed:
        return idx

    build_index(cfg)
    return load_index(cfg)


def detect_changes(cfg: Config) -> bool:
    files_meta = _load_files_meta(cfg)
    current_files = _iter_index_files(cfg)
    current_meta: Dict[str, Dict[str, Any]] = {}

    for path, source in current_files:
        try:
            fp = _file_fingerprint(path)
        except Exception:
            continue
        fp["source"] = source
        current_meta[str(path)] = fp
        old = files_meta.get(str(path))
        if not old or old.get("mtime_ns") != fp["mtime_ns"] or old.get("size") != fp["size"]:
            return True

    if set(files_meta.keys()) != set(current_meta.keys()):
        return True

    return False


def retrieve(cfg: Config, idx: Dict[str, Any], query: str) -> Tuple[str, List[Dict[str, str]]]:
    docs = idx.get("docs", [])
    df = idx.get("df", {})
    avgdl = float(idx.get("avgdl", 1.0))
    if not docs:
        return "", []
    query_terms = _tokenize(query)
    if not query_terms:
        return "", []

    N = len(docs)
    k1 = 1.5
    b = 0.75
    scores: List[Tuple[float, int]] = []
    for i, doc in enumerate(docs):
        dl = max(int(doc.get("len", 0)), 1)
        tf = doc.get("tf", {})
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            df_t = df.get(term, 0)
            idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * (f * (k1 + 1) / denom)
        # title/path boost
        title_tokens = set(doc.get("title_tokens", []))
        title_bonus = 0.0
        if title_tokens:
            for term in query_terms:
                if term in title_tokens:
                    df_t = df.get(term, 0)
                    idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
                    title_bonus += cfg.title_boost * idf
        total = score + title_bonus
        if total > 0:
            scores.append((total, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    candidates = scores[: max(cfg.mmr_candidates, cfg.max_context_chunks)]

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # Precompute token sets for candidates
    token_sets: Dict[int, set] = {}
    for _, idx_doc in candidates:
        token_sets[idx_doc] = set(docs[idx_doc].get("tf", {}).keys())

    selected: List[int] = []
    selected_scores: Dict[int, float] = {}
    per_file: Dict[str, int] = {}
    lambda_ = float(cfg.mmr_lambda)
    while len(selected) < cfg.max_context_chunks and candidates:
        best = None
        best_score = -1e9
        for score, idx_doc in candidates:
            path = docs[idx_doc].get("path", "")
            if cfg.max_chunks_per_file > 0 and per_file.get(path, 0) >= cfg.max_chunks_per_file:
                continue
            if not selected:
                mmr = score
            else:
                max_sim = 0.0
                for s in selected:
                    max_sim = max(max_sim, jaccard(token_sets.get(idx_doc, set()), token_sets.get(s, set())))
                mmr = lambda_ * score - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best = idx_doc
                selected_scores[idx_doc] = score
        if best is None:
            break
        selected.append(best)
        path = docs[best].get("path", "")
        per_file[path] = per_file.get(path, 0) + 1
        candidates = [(s, i) for (s, i) in candidates if i != best]

    context_blocks = []
    sources = []
    for rank, idx_doc in enumerate(selected, start=1):
        doc = docs[idx_doc]
        context_blocks.append(f"[{rank}] {doc['text']}")
        sources.append({"label": f"[{rank}]", "path": doc["path"]})
    context = "\n\n".join(context_blocks)
    return context, sources


def search_index(cfg: Config, idx: Dict[str, Any], query: str, max_hits: int = 8, snippet_chars: int = 220) -> List[Dict[str, str]]:
    docs = idx.get("docs", [])
    df = idx.get("df", {})
    avgdl = float(idx.get("avgdl", 1.0))
    if not docs:
        return []
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    N = len(docs)
    k1 = 1.5
    b = 0.75
    scores: List[Tuple[float, int]] = []
    for i, doc in enumerate(docs):
        dl = max(int(doc.get("len", 0)), 1)
        tf = doc.get("tf", {})
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            df_t = df.get(term, 0)
            idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * (f * (k1 + 1) / denom)
        title_tokens = set(doc.get("title_tokens", []))
        title_bonus = 0.0
        if title_tokens:
            for term in query_terms:
                if term in title_tokens:
                    df_t = df.get(term, 0)
                    idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
                    title_bonus += cfg.title_boost * idf
        total = score + title_bonus
        if total > 0:
            scores.append((total, i))
    scores.sort(reverse=True, key=lambda x: x[0])

    seen_paths = set()
    results: List[Dict[str, str]] = []
    for rank, (_, idx_doc) in enumerate(scores, start=1):
        doc = docs[idx_doc]
        path = doc.get("path")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        text = doc.get("text", "")
        snippet = text[:snippet_chars].replace("\n", " ").strip()
        results.append({
            "label": f"[{len(results)+1}]",
            "path": path,
            "snippet": snippet,
        })
        if len(results) >= max_hits:
            break
    return results


def answer_question(
    cfg: Config,
    idx: Dict[str, Any],
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    extra_context: Optional[str] = None,
    extra_sources: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, str]], str]:
    context, sources = retrieve(cfg, idx, question)
    if GREET_RE.match(question.strip()):
        # avoid wasting context on greetings
        context = ""
        sources = []
    max_chars = _max_input_chars(cfg)
    extra_context = _truncate_text(extra_context or "", max_chars // 3) if extra_context else None
    remaining = max_chars - (len(extra_context) if extra_context else 0)
    context = _truncate_text(context, max(remaining, 0))
    sys_msg = SYSTEM_PROMPT
    if not context:
        sys_msg += " There is no indexed context available."
    def build_messages(ctx: str, extra: Optional[str]) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": sys_msg}]
        if extra:
            msgs.append({"role": "system", "content": extra})
        if ctx:
            msgs.append({"role": "system", "content": f"Context:\n{ctx}"})
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": question})
        return msgs

    used_context = context
    used_extra = extra_context
    try:
        reply = _chat(cfg, build_messages(used_context, used_extra))
    except RuntimeError as e:
        msg = str(e)
        if "context_length_exceeded" not in msg and "maximum context length" not in msg:
            raise
        # retry with progressively smaller context
        for factor in (0.5, 0.25, 0.1):
            used_context = _truncate_text(context, int(len(context) * factor)) if context else ""
            used_extra = _truncate_text(extra_context, int(len(extra_context or "") * factor)) if extra_context else None
            try:
                reply = _chat(cfg, build_messages(used_context, used_extra))
                break
            except RuntimeError as e2:
                msg2 = str(e2)
                if "context_length_exceeded" not in msg2 and "maximum context length" not in msg2:
                    raise
        else:
            # final fallback: no context
            used_context = ""
            used_extra = None
            reply = _chat(cfg, build_messages(used_context, used_extra))

    if extra_sources and used_extra:
        sources = extra_sources + sources
    if not used_context:
        sources = []
    return reply, sources, used_context


def propose_edit(cfg: Config, path: str, instruction: str) -> Tuple[str, str]:
    p = Path(path)
    if not p.is_absolute():
        p = (cfg.vault_path / p).resolve()
    if not str(p).startswith(str(cfg.vault_path.resolve())):
        raise RuntimeError("Path is outside vault.")
    if p.suffix.lower() not in EDIT_EXTS:
        raise RuntimeError("Unsupported file type.")
    original = p.read_text(encoding="utf-8", errors="ignore")
    prompt = (
        "You are editing a single markdown note. "
        "Return the full updated content only, no commentary.\n\n"
        "Instruction:\n"
        f"{instruction}\n\n"
        "Original:\n"
        f"{original}\n"
    )
    messages = [
        {"role": "system", "content": "You are a precise editor. Output only the updated file content."},
        {"role": "user", "content": prompt},
    ]
    updated = _chat(cfg, messages)
    if not updated.strip():
        raise RuntimeError("Model returned empty edit.")
    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=str(p),
            tofile=str(p),
            lineterm="",
        )
    )
    return updated, diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "sync", "ask"], help="Command to run")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--question", default="")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.command == "index":
        build_index(cfg)
        return
    if args.command == "sync":
        sync_index(cfg)
        return
    if args.command == "ask":
        if not args.question:
            print("Provide --question")
            return
        idx = load_index(cfg)
        reply, sources, _ = answer_question(cfg, idx, args.question)
        print(reply)
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"{s['label']} {s['path']}")
        return


if __name__ == "__main__":
    main()
