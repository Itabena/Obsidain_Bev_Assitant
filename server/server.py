#!/usr/bin/env python3
import json
import threading
import time
import uuid
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from assistant import (
    DEFAULT_CONFIG_PATH,
    answer_question,
    search_index,
    load_config,
    load_index,
    propose_edit,
    sync_index,
    web_search,
    detect_changes,
    _find_latest_note,
    _resolve_note_path_from_query,
    _summarize_text,
    _read_text,
    _chat,
    LATEST_NOTE_RE,
    SUMMARY_RE,
    NOTE_WORD_RE,
    LATEST_HINT_RE,
    EDIT_EXTS,
)

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
PROFILE_PATH = HERE / "profile.json"

PROFILE_LOCK = threading.Lock()
PROFILE: Optional[Dict[str, Any]] = None

cfg = load_config(DEFAULT_CONFIG_PATH)

INDEX_LOCK = threading.Lock()
INDEX_DATA: Optional[Dict[str, Any]] = None

PENDING_EDITS: Dict[str, Dict[str, Any]] = {}

RESPONSE_TOKENS_RE = re.compile(
    r"(response|output).*(length|size|tokens|max)|max\\s*(tokens|response)|response\\s*tokens",
    re.I,
)
CONTEXT_RE = re.compile(r"(context|ctx).*(length|size|chunks|max)|max\\s*context|context\\s*chunks", re.I)
NUMBER_RE = re.compile(r"(\\d+)")
LATEST_NOTE_QUERY_RE = re.compile(
    r"(latest|most recent|last|newest).*(note|file|document)",
    re.I,
)
FIND_NOTES_RE = re.compile(
    r"(find|search|list|do i have|is there).*(notes?|note|files?|file|documents?)",
    re.I,
)
ACTIVE_NOTE_RE = re.compile(r"(current|open|active).*(note|file|doc|document)|current note|open note|active note", re.I)
SUBJECT_RE = re.compile(r"\\b(subject|topic|title|about)\\b", re.I)
POS_FEEDBACK_RE = re.compile(r"\\b(thanks|thank you|great|awesome|perfect|works now|nice|love it)\\b", re.I)
NEG_FEEDBACK_RE = re.compile(r"\\b(not helpful|doesn't work|does not work|wrong|useless|lie|broken|still not)\\b", re.I)


def _generate_chat_title(history: Any) -> Optional[str]:
    cleaned = _sanitize_history(history, max_messages=12, max_chars=600)
    if not cleaned:
        return None
    lines = []
    for item in cleaned:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = item.get("content", "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220] + "..."
        lines.append(f"{role}: {content}")
    prompt = (
        "Create a short 3-7 word title for this conversation. "
        "Return only the title without quotes or punctuation at the end.\n\n"
        + "\n".join(lines)
    )
    messages = [
        {"role": "system", "content": "You generate concise chat titles."},
        {"role": "user", "content": prompt},
    ]
    title = _chat(cfg, messages).strip()
    title = re.sub(r"^title\\s*:\\s*", "", title, flags=re.I).strip()
    title = title.strip("\"'“”")
    title = re.sub(r"[\\r\\n]+", " ", title).strip()
    title = re.sub(r"[.\\-–—]+$", "", title).strip()
    if not title:
        return None
    return title[:80]


def _resolve_active_note(cfg: Any, raw_path: str) -> Optional[Path]:
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = (cfg.vault_path / p).resolve()
    if not str(p).startswith(str(cfg.vault_path.resolve())):
        return None
    if p.suffix.lower() not in EDIT_EXTS:
        return None
    return p if p.exists() else None


def _extract_subject(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or "(untitled)"
        return line[:200]
    return "(empty note)"


def _load_profile() -> Dict[str, Any]:
    global PROFILE
    with PROFILE_LOCK:
        if PROFILE is not None:
            return PROFILE
        if PROFILE_PATH.exists():
            try:
                PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                return PROFILE
            except Exception:
                PROFILE = None
        PROFILE = {
            "feedback": {
                "helpful": 0,
                "not_helpful": 0,
                "last": None,
                "last_at": None,
                "notes": [],
            }
        }
        return PROFILE


def _save_profile(profile: Dict[str, Any]) -> None:
    with PROFILE_LOCK:
        PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=True, indent=2), encoding="utf-8")


def _update_feedback(helpful: bool, comment: Optional[str] = None) -> Dict[str, Any]:
    profile = _load_profile()
    fb = profile.setdefault("feedback", {})
    fb.setdefault("helpful", 0)
    fb.setdefault("not_helpful", 0)
    fb.setdefault("notes", [])
    if helpful:
        fb["helpful"] += 1
        fb["last"] = "helpful"
    else:
        fb["not_helpful"] += 1
        fb["last"] = "not_helpful"
    fb["last_at"] = datetime.now().isoformat(timespec="seconds")
    if comment:
        fb["notes"].append(comment)
        fb["notes"] = fb["notes"][-10:]
    _save_profile(profile)
    return profile


def _profile_context() -> Optional[str]:
    profile = _load_profile()
    fb = profile.get("feedback", {})
    helpful = int(fb.get("helpful", 0) or 0)
    not_helpful = int(fb.get("not_helpful", 0) or 0)
    total = helpful + not_helpful
    if total <= 0:
        return None
    last = fb.get("last") or "unknown"
    guidance = ""
    if last == "not_helpful":
        guidance = "Adjust by being more concise, asking a clarifying question, and verifying assumptions."
    else:
        guidance = "Keep the current style, remain concise, and confirm key steps when needed."
    return (
        f"User feedback summary: helpful={helpful}, not helpful={not_helpful}, last={last}. {guidance}"
    )


def _save_config(values: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        try:
            data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(values)
    DEFAULT_CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _apply_mode(mode: str) -> Dict[str, Any]:
    mode = mode.lower()
    if mode == "power":
        return {
            "mode": "power",
            "allow_edit": True,
            "allow_terminal": True,
            "allow_python": True,
            "allow_self_improve": False,
            "read_only": False,
        }
    if mode == "full":
        return {
            "mode": "full",
            "allow_edit": True,
            "allow_terminal": True,
            "allow_python": True,
            "allow_self_improve": True,
            "read_only": False,
        }
    return {
        "mode": "safe",
        "allow_edit": True,
        "allow_terminal": False,
        "allow_python": False,
        "allow_self_improve": False,
        "read_only": False,
    }


def _apply_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    if "mode" in updates:
        updates = {**updates, **_apply_mode(str(updates["mode"]))}

    allow_edit = bool(updates.get("allow_edit", cfg.allow_edit))
    read_only = bool(updates.get("read_only", cfg.read_only))

    if read_only:
        allow_edit = False
    else:
        if "allow_edit" in updates:
            read_only = not allow_edit

    cfg.mode = str(updates.get("mode", cfg.mode))
    cfg.allow_edit = allow_edit
    cfg.allow_terminal = bool(updates.get("allow_terminal", cfg.allow_terminal))
    cfg.allow_python = bool(updates.get("allow_python", cfg.allow_python))
    cfg.allow_self_improve = bool(updates.get("allow_self_improve", cfg.allow_self_improve))
    cfg.read_only = read_only

    cfg.max_response_tokens = int(updates.get("max_response_tokens", cfg.max_response_tokens))
    cfg.max_context_chunks = int(updates.get("max_context_chunks", cfg.max_context_chunks))

    values = {
        "mode": cfg.mode,
        "allow_edit": cfg.allow_edit,
        "allow_terminal": cfg.allow_terminal,
        "allow_python": cfg.allow_python,
        "allow_self_improve": cfg.allow_self_improve,
        "read_only": cfg.read_only,
        "max_response_tokens": cfg.max_response_tokens,
        "max_context_chunks": cfg.max_context_chunks,
    }
    _save_config(values)
    return values


def _handle_settings_command(message: str) -> Optional[str]:
    lower = message.lower()
    if "show settings" in lower or "current settings" in lower or "assistant settings" in lower:
        return (
            f"Current settings:\\n"
            f"- max_response_tokens: {cfg.max_response_tokens}\\n"
            f"- max_context_chunks: {cfg.max_context_chunks}"
        )

    number_match = NUMBER_RE.search(message)
    if not number_match:
        return None
    value = int(number_match.group(1))

    if RESPONSE_TOKENS_RE.search(message):
        clamped = max(64, min(value, 2048))
        cfg.max_response_tokens = clamped
        _save_config({"max_response_tokens": clamped})
        if clamped != value:
            return f"Set max_response_tokens to {clamped} (clamped from {value})."
        return f"Set max_response_tokens to {clamped}."

    if CONTEXT_RE.search(message):
        clamped = max(1, min(value, 20))
        cfg.max_context_chunks = clamped
        _save_config({"max_context_chunks": clamped})
        if clamped != value:
            return f"Set max_context_chunks to {clamped} (clamped from {value})."
        return f"Set max_context_chunks to {clamped}."

    return None


def _load_index_safe() -> Dict[str, Any]:
    global INDEX_DATA
    with INDEX_LOCK:
        if INDEX_DATA is None:
            INDEX_DATA = load_index(cfg)
        return INDEX_DATA


def _sync_index_safe() -> None:
    global INDEX_DATA
    with INDEX_LOCK:
        INDEX_DATA = sync_index(cfg)


def _auto_sync_loop() -> None:
    if not cfg.auto_sync:
        return
    pending = False
    last_change = 0.0
    while True:
        try:
            if detect_changes(cfg):
                pending = True
                last_change = time.time()
            if pending:
                idle_for = time.time() - last_change
                if idle_for >= max(cfg.sync_idle_sec, cfg.sync_interval_sec):
                    _sync_index_safe()
                    pending = False
            else:
                # Ensure index is loaded at least once
                _load_index_safe()
        except Exception as e:
            print(f"[auto-sync] {e}")
        time.sleep(max(cfg.sync_interval_sec, 10))


def _sanitize_history(history: Any, max_messages: int = 10, max_chars: int = 2000) -> List[Dict[str, str]]:
    if not isinstance(history, list):
        return []
    cleaned: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        if len(content) > max_chars:
            content = content[:max_chars]
        cleaned.append({"role": role, "content": content})
    if len(cleaned) > max_messages:
        cleaned = cleaned[-max_messages:]
    return cleaned


class Handler(BaseHTTPRequestHandler):
    server_version = "ObsidianAssistant/0.2"

    def _set_headers(self, code=200, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
            self._set_headers(200, "text/html")
            self.wfile.write(html.encode("utf-8"))
            return
        if self.path == "/settings":
            self._set_headers(200)
            self.wfile.write(
                json.dumps(
                    {
                        "mode": cfg.mode,
                        "allow_edit": cfg.allow_edit,
                        "allow_terminal": cfg.allow_terminal,
                        "allow_python": cfg.allow_python,
                        "allow_self_improve": cfg.allow_self_improve,
                        "read_only": cfg.read_only,
                        "max_response_tokens": cfg.max_response_tokens,
                        "max_context_chunks": cfg.max_context_chunks,
                        "enable_web": cfg.enable_web,
                    }
                ).encode("utf-8")
            )
            return
        self._set_headers(404)
        self.wfile.write(b"Not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        if self.path == "/feedback":
            helpful = bool(payload.get("helpful", False))
            comment = payload.get("comment")
            profile = _update_feedback(helpful, comment if isinstance(comment, str) else None)
            self._set_headers(200)
            self.wfile.write(json.dumps({"ok": True, "feedback": profile.get("feedback", {})}).encode("utf-8"))
            return

        if self.path == "/title":
            history = payload.get("messages", [])
            try:
                title = _generate_chat_title(history)
                if not title:
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"title": ""}).encode("utf-8"))
                    return
                self._set_headers(200)
                self.wfile.write(json.dumps({"title": title}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/chat":
            message = payload.get("message", "")
            use_web = bool(payload.get("use_web", False))
            history = _sanitize_history(payload.get("history", []))
            active_note = payload.get("active_note", "")
            try:
                if NEG_FEEDBACK_RE.search(message) and not POS_FEEDBACK_RE.search(message):
                    _update_feedback(False, message)
                elif POS_FEEDBACK_RE.search(message) and not NEG_FEEDBACK_RE.search(message):
                    _update_feedback(True, message)

                settings_reply = _handle_settings_command(message)
                if settings_reply is not None:
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"reply": settings_reply, "sources": []}).encode("utf-8"))
                    return

                if ACTIVE_NOTE_RE.search(message):
                    active_path = _resolve_active_note(cfg, str(active_note))
                    if active_path is None:
                        reply = (
                            "I couldn't resolve the currently open note. "
                            "Please make sure a note is active in Obsidian or provide the note path."
                        )
                        sources = []
                    else:
                        text = _read_text(active_path)
                        if SUBJECT_RE.search(message):
                            subject = _extract_subject(text)
                            reply = f"Current note: {active_path}\\nSubject: {subject}"
                        else:
                            reply = _summarize_text(cfg, text, message)
                        sources = [{"label": "[1]", "path": str(active_path)}]
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"reply": reply, "sources": sources}).encode("utf-8"))
                    return

                if LATEST_NOTE_QUERY_RE.search(message):
                    note_path = _find_latest_note(cfg)
                    if note_path is None:
                        reply = "I couldn't find any notes in your vault."
                        sources = []
                    else:
                        mtime = datetime.fromtimestamp(note_path.stat().st_mtime)
                        reply = f"Latest updated note: {note_path}\\nLast modified: {mtime.isoformat(sep=' ', timespec='seconds')}"
                        sources = [{"label": "[1]", "path": str(note_path)}]
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"reply": reply, "sources": sources}).encode("utf-8"))
                    return

                if FIND_NOTES_RE.search(message):
                    idx = _load_index_safe()
                    results = search_index(cfg, idx, message, max_hits=8)
                    if not results:
                        reply = (
                            "I couldn't find matching notes in the index. "
                            "Try reindexing or give a more specific keyword."
                        )
                        sources = []
                    else:
                        lines = ["Matching notes:"]
                        for item in results:
                            line = f"{item['label']} {item['path']}"
                            if item.get("snippet"):
                                line += f" — {item['snippet']}"
                            lines.append(line)
                        reply = "\n".join(lines)
                        sources = [{"label": r["label"], "path": r["path"]} for r in results]
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"reply": reply, "sources": sources}).encode("utf-8"))
                    return

                summary_word = SUMMARY_RE.search(message) is not None
                note_hint = NOTE_WORD_RE.search(message) is not None
                latest_hint = LATEST_HINT_RE.search(message) is not None or LATEST_NOTE_RE.search(message) is not None

                # Special-case summary requests about notes
                note_path = _resolve_note_path_from_query(cfg, message)
                should_summarize_note = summary_word and (note_hint or latest_hint or note_path is not None)

                if should_summarize_note:
                    if note_path is None:
                        note_path = _find_latest_note(cfg)
                    if note_path is not None:
                        text = note_path.read_text(encoding="utf-8", errors="ignore")
                        lower_msg = message.lower()
                        if "last paragraph" in lower_msg or "final paragraph" in lower_msg:
                            parts = [p for p in text.split("\n\n") if p.strip()]
                            text = parts[-1] if parts else text
                        reply = _summarize_text(cfg, text, message)
                        sources = [{"label": "[1]", "path": str(note_path)}]
                    else:
                        reply = (
                            "I couldn't find a note to summarize. "
                            "Tell me the note path (e.g., \"summarize note: folder/note.md\") "
                            "or ask \"summarize latest note\"."
                        )
                        sources = []
                else:
                    extra_context = None
                    extra_sources = []
                    if use_web:
                        if cfg.enable_web:
                            results = web_search(message, max_results=5)
                            if results:
                                lines = ["Web results:"]
                                for i, item in enumerate(results, start=1):
                                    line = f"[W{i}] {item['title']} - {item['url']}"
                                    if item.get("snippet"):
                                        line += f" | {item['snippet']}"
                                    lines.append(line)
                                    extra_sources.append({"label": f"[W{i}]", "path": item["url"]})
                                extra_context = "\n".join(lines)
                        else:
                            extra_context = "Web search requested but disabled in config."
                    profile_hint = _profile_context()
                    if profile_hint:
                        extra_context = f"{profile_hint}\n\n{extra_context}" if extra_context else profile_hint
                    idx = _load_index_safe()
                    reply, sources, _ = answer_question(
                        cfg,
                        idx,
                        message,
                        history=history if history else None,
                        extra_context=extra_context,
                        extra_sources=extra_sources,
                    )
                if use_web and not cfg.enable_web:
                    reply += "\n\n[Web search is disabled in config.]"
                self._set_headers(200)
                self.wfile.write(json.dumps({"reply": reply, "sources": sources}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/edit/propose":
            if not cfg.allow_edit:
                self._set_headers(403)
                self.wfile.write(json.dumps({"error": "Edit mode disabled in settings."}).encode("utf-8"))
                return
            path = payload.get("path", "")
            instruction = payload.get("instruction", "")
            try:
                updated, diff = propose_edit(cfg, path, instruction)
                token = str(uuid.uuid4())
                PENDING_EDITS[token] = {"path": path, "updated": updated}
                self._set_headers(200)
                self.wfile.write(json.dumps({"token": token, "diff": diff, "path": path}).encode("utf-8"))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/edit/apply":
            token = payload.get("token", "")
            if cfg.read_only:
                self._set_headers(403)
                self.wfile.write(json.dumps({"error": "Read-only mode enabled."}).encode("utf-8"))
                return
            edit = PENDING_EDITS.pop(token, None)
            if not edit:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "Edit token not found"}).encode("utf-8"))
                return
            try:
                path = Path(edit["path"])
                if not path.is_absolute():
                    path = (cfg.vault_path / path).resolve()
                path.write_text(edit["updated"], encoding="utf-8")
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "applied"}).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if self.path == "/edit/reject":
            token = payload.get("token", "")
            PENDING_EDITS.pop(token, None)
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "rejected"}).encode("utf-8"))
            return

        if self.path == "/settings":
            try:
                updated = _apply_settings(payload if isinstance(payload, dict) else {})
                self._set_headers(200)
                self.wfile.write(json.dumps(updated).encode("utf-8"))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Unknown route"}).encode("utf-8"))


def main():
    host = "127.0.0.1"
    port = 8000
    server = HTTPServer((host, port), Handler)
    print(f"Listening on http://{host}:{port}")
    if cfg.auto_sync:
        t = threading.Thread(target=_auto_sync_loop, daemon=True)
        t.start()
    server.serve_forever()


if __name__ == "__main__":
    main()
