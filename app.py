# -------------------------------------------------------------
# Project   : Study 3 - AI Tutor-Based Phishing Awareness
# Purpose   : Bilingual (Arabic/English) web platform for
#             phishing awareness training and assessment
#             designed for Saudi healthcare employees.
# Tech Stack: Python 3.9, Streamlit, Multi-Provider AI (Groq/Claude/OpenAI/Gemini)
# AI Models : Groq LLaMA 3.3-70b | Claude claude-sonnet-4-6 | GPT-4o | Gemini 1.5 Pro
# Admin     : Hidden Admin Panel at /?admin=true (password protected)
#             Compare 4 AI providers with 8-metric scoring system.
# -------------------------------------------------------------
# App Flow:
#   HOME -> LEARNING (6 AI-generated phishing examples)
#        -> COMPLETE -> ASSESSMENT (10 questions)
#        -> RESULTS -> PERFORMANCE REPORT
# =============================================================
#
# EN — QUICK CODE MAP (where to look for what):
#   • ROLE_MAP                  → job-role display names → (description, context, role_type)
#   • PHISHING_SCENARIOS        → legacy/unused list, kept for reference only
#   • FORCED_SCENARIOS          → the actual scenario pool per role_type
#                                 (admin/clinical/it have 10 each, "other" has 6
#                                 tied 1:1 to OTHER_JOB_PROFILES)
#   • OTHER_JOB_PROFILES        → the 6 sub-job profiles used when role="Other"
#                                 (now also carries name_en/name_ar for greeting binding)
#   • RECIPIENT_POOLS           → fixed (name, email) pairs per role_type, used to
#                                 keep the email greeting and the "to" address consistent
#   • get_session_random_order  → shared helper: shuffles indices once per session
#                                 so the same "example slot" doesn't always get the
#                                 same scenario/recipient across different sessions
#   • build_prompt               → builds the prompt for the LEARNING phase (6 examples)
#   • build_assess_prompt        → builds the prompt for the ASSESSMENT phase (10 questions)
#   • call_ai / call_groq        → sends the prompt to whichever provider is active
#                                  (groq/anthropic/openai/gemini) and normalizes the reply
#   • parse_json_response        → robust JSON parsing with repair fallbacks
#   • page_results                → renders "مراجعة الإجابات" (review answers) — has the
#                                  <bdi> bidi-isolation fix for mixed Arabic/English text
#   • load_persistent_provider /
#     save_persistent_provider /
#     set_active_provider        → keeps the admin's chosen AI provider saved to disk
#                                  (provider_config.json) so it survives logout/new sessions
#
# AR — خريطة سريعة للكود (وين تدورين على كل شي):
#   • ROLE_MAP                  → أسماء الأدوار الوظيفية ← (الوصف، السياق، نوع الدور)
#   • PHISHING_SCENARIOS        → قائمة قديمة غير مستخدمة، باقية للمرجعية فقط
#   • FORCED_SCENARIOS          → مسبح السيناريوهات الفعلي لكل دور
#                                 (إداري/سريري/تقني فيها 10 لكل واحد، و"Other" فيها 6
#                                 مرتبطة 1:1 مع OTHER_JOB_PROFILES)
#   • OTHER_JOB_PROFILES        → الـ6 بروفايلات الفرعية المستخدمة لما الدور = "Other"
#                                 (الآن فيها كمان name_en/name_ar لربط اسم التحية)
#   • RECIPIENT_POOLS           → أزواج (اسم، بريد) ثابتة لكل دور، نستخدمها حتى تتطابق
#                                 التحية بالبريد مع عنوان "to" دايمًا
#   • get_session_random_order  → دالة مشتركة: تخلط الفهارس مرة واحدة لكل جلسة، حتى
#                                 "خانة المثال" نفسها ما تطلع لها نفس السيناريو/المستلم
#                                 بكل الجلسات
#   • build_prompt               → يبني تعليمات مرحلة التعلم (الـ6 أمثلة)
#   • build_assess_prompt        → يبني تعليمات مرحلة الاختبار (الـ10 أسئلة)
#   • call_ai / call_groq        → يرسل التعليمات لأي بيئة نشطة حاليًا
#                                  (groq/anthropic/openai/gemini) ويوحّد شكل الرد
#   • parse_json_response        → تحليل JSON قوي مع محاولات تصحيح تلقائية
#   • page_results                → يعرض صفحة "مراجعة الإجابات" — فيها إصلاح اتجاه
#                                  النص (<bdi>) للنصوص المختلطة عربي/إنجليزي
#   • load_persistent_provider /
#     save_persistent_provider /
#     set_active_provider        → يحفظ اختيار الأدمن لمزوّد الذكاء الاصطناعي على القرص
#                                  (provider_config.json) حتى يفضل بعد تسجيل خروج/جلسة جديدة
# =============================================================

import streamlit as st
import json
import requests
import os
import re
import html as html_lib
import random
import urllib.parse
import ast

st.set_page_config(
    page_title="AI Phishing Awareness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto"
)

# =============================================================
# Persistent AI Provider Config (survives logout / new session)
# -------------------------------------------------------------
# ai_provider used to live only in st.session_state, which is
# reset on every logout/login or new session. We now store the
# admin's chosen provider in a small JSON file on disk so it
# stays the single source of truth across all sessions.
# =============================================================
_PROVIDER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provider_config.json")
_VALID_PROVIDERS = {"groq", "anthropic", "openai", "gemini"}

def load_persistent_provider(default="openai"):
    try:
        with open(_PROVIDER_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        pk = data.get("ai_provider", default)
        if pk in _VALID_PROVIDERS:
            return pk
    except Exception:
        pass
    return default

def save_persistent_provider(pk):
    if pk not in _VALID_PROVIDERS:
        return
    try:
        with open(_PROVIDER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"ai_provider": pk}, f)
    except Exception:
        # If the filesystem is read-only (some hosting setups), fail silently
        # and keep relying on session_state for the current session only.
        pass

def set_active_provider(pk):
    """Single entrypoint to change the active AI provider.
    Updates session_state AND persists to disk so it survives
    logout/login and new sessions/devices."""
    st.session_state["ai_provider"] = pk
    save_persistent_provider(pk)

# =============================================================
# PERSISTENT RESEARCH DATA — survives refresh/new sessions
# -------------------------------------------------------------
# Two JSON files on disk next to the app:
#  - runs.json     : one record per FULL CYCLE (6 learning + 10
#                     assessment) the researcher rated, holistically.
#  - metrics.json   : raw auto-tracked API call stats (speed, JSON
#                     success, errors, uniqueness) — accumulated
#                     across every call regardless of session.
# Both are best-effort: if the filesystem is read-only on a given
# host, writes fail silently and behaviour falls back to the old
# session-only behaviour (no crash either way).
# =============================================================
_RUNS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.json")
_METRICS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.json")

def load_runs():
    local = []
    try:
        with open(_RUNS_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            local = data
    except Exception:
        local = []
    try:
        return merge_local_and_gsheet_runs(local)
    except Exception:
        return local

def save_run(record):
    """Append one holistic run-rating record and persist to disk."""
    runs = load_runs()
    runs.append(record)
    try:
        with open(_RUNS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        push_run_to_gsheet(record)
    except Exception:
        pass
    return runs

def delete_all_runs():
    try:
        with open(_RUNS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
    except Exception:
        pass

def load_metrics_file():
    try:
        with open(_METRICS_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def save_metrics_file(metrics_dict):
    try:
        with open(_METRICS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =============================================================
# GOOGLE SHEETS SYNC — durable copy independent of this app's
# container/host, so research data survives redeploys/reboots.
# -------------------------------------------------------------
# Fully best-effort and optional: if gspread isn't installed, or
# st.secrets["gcp_service_account"] / st.secrets["GSHEET_ID"]
# aren't configured yet, every function below quietly no-ops.
# Nothing here can ever crash the app or block local saving —
# the local JSON files remain the primary, always-on storage;
# this is purely an extra durable copy on top.
#
# ONE-TIME SETUP (done outside this code, in Google Cloud /
# Google Sheets):
#   1. Google Cloud Console → create a Service Account → create
#      a JSON key for it → copy its "client_email".
#   2. Create a new Google Sheet (any name) under the
#      researcher's own Google account.
#   3. Share that Sheet with the service account's client_email,
#      giving it "Editor" access.
#   4. In Streamlit Cloud → App settings → Secrets, add:
#        GSHEET_ID = "<the sheet's ID from its URL>"
#        [gcp_service_account]
#        type = "service_account"
#        ... (every field from the downloaded JSON key, as TOML)
#   5. Add "gspread" and "google-auth" to requirements.txt.
# That's it — every future "Save Ratings" / metrics update will
# also land in the Sheet automatically, with three tabs:
#   - "Cycle Ratings" : one row per manually-rated cycle
#   - "Auto Metrics"  : one row per periodic metrics snapshot
# =============================================================
_GSHEET_CLIENT_CACHE = {"client": None, "tried": False}

def _get_gsheet_client():
    """Lazily build (and cache) an authorized gspread client from
    st.secrets. Returns None — silently — if the library isn't
    installed or secrets aren't configured, so callers never need
    to special-case "not set up yet" themselves."""
    if _GSHEET_CLIENT_CACHE["tried"]:
        return _GSHEET_CLIENT_CACHE["client"]
    _GSHEET_CLIENT_CACHE["tried"] = True
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        _GSHEET_CLIENT_CACHE["client"] = client
    except Exception:
        _GSHEET_CLIENT_CACHE["client"] = None
    return _GSHEET_CLIENT_CACHE["client"]

def _get_gsheet_id():
    try:
        sid = st.secrets.get("GSHEET_ID")
        return sid.strip() if isinstance(sid, str) and sid.strip() else None
    except Exception:
        return None

def gsheet_setup_status():
    """Human-readable status for the admin panel — never raises."""
    try:
        import gspread  # noqa: F401
    except Exception:
        return False, "مكتبة gspread غير مثبتة (أضيفيها لـ requirements.txt)" if st.session_state.get("language") == "Arabic" else "gspread library not installed (add it to requirements.txt)"
    if not _get_gsheet_id():
        return False, "GSHEET_ID غير موجود بالـ Secrets" if st.session_state.get("language") == "Arabic" else "GSHEET_ID missing from Secrets"
    try:
        has_creds = "gcp_service_account" in st.secrets
    except Exception:
        has_creds = False
    if not has_creds:
        return False, "بيانات اعتماد gcp_service_account غير موجودة بالـ Secrets" if st.session_state.get("language") == "Arabic" else "gcp_service_account credentials missing from Secrets"
    client = _get_gsheet_client()
    if client is None:
        return False, "فشل الاتصال — تأكدي من صحة بيانات الاعتماد ومشاركة الشيت مع الحساب الآلي" if st.session_state.get("language") == "Arabic" else "Connection failed — check credentials and that the Sheet is shared with the service account"
    return True, "متصل وشغال ✅" if st.session_state.get("language") == "Arabic" else "Connected and working ✅"

def _get_or_create_worksheet(sheet, tab_name, headers):
    try:
        ws = sheet.worksheet(tab_name)
    except Exception:
        ws = sheet.add_worksheet(title=tab_name, rows=1000, cols=max(10, len(headers)))
        ws.append_row(headers)
        return ws
    try:
        existing_header = ws.row_values(1)
        if not existing_header:
            ws.append_row(headers)
    except Exception:
        pass
    return ws

def push_run_to_gsheet(record):
    """Append one manually-rated cycle to the 'Cycle Ratings' tab.
    Best-effort: any failure (offline, not configured, quota, etc.)
    is swallowed so the researcher's local save never fails because
    of this extra step."""
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    if not client or not sheet_id:
        return
    try:
        sheet = client.open_by_key(sheet_id)
        headers = ["timestamp", "provider", "language", "overall",
                   "auto_difficulty", "auto_arabic", "auto_quality", "auto_medical",
                   "n_auto_emails", "avg_speed", "json_rate", "error_rate",
                   "diversity", "note"]
        ws = _get_or_create_worksheet(sheet, "Cycle Ratings", headers)
        ws.append_row([record.get(h, "") for h in headers], value_input_option="USER_ENTERED")
    except Exception:
        pass

def push_metrics_snapshot_to_gsheet(provider, m):
    """Append one timestamped snapshot row per provider to the
    'Auto Metrics' tab, using the SAME raw shape _record_metric keeps
    in st.session_state['metrics'][provider] (speed list, json_ok/
    json_fail counts, errors, calls, hashes) — summarized here into
    the same percentages shown on the Score Card."""
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    if not client or not sheet_id:
        return
    try:
        sheet = client.open_by_key(sheet_id)
        headers = ["timestamp", "provider", "calls", "avg_speed_s",
                   "json_success_rate_pct", "error_rate_pct", "unique_responses"]
        ws = _get_or_create_worksheet(sheet, "Auto Metrics", headers)
        m = m or {}
        speeds = m.get("speed") or []
        avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else ""
        json_total = (m.get("json_ok", 0) + m.get("json_fail", 0))
        json_rate = round(100 * m.get("json_ok", 0) / json_total, 1) if json_total else ""
        calls = m.get("calls", 0)
        err_rate = round(100 * m.get("errors", 0) / calls, 1) if calls else ""
        unique = len(m.get("hashes") or [])
        ws.append_row([
            __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            provider, calls, avg_speed, json_rate, err_rate, unique,
        ], value_input_option="USER_ENTERED")
    except Exception:
        pass

def pull_latest_auto_metrics_from_gsheet():
    """Read the 'Auto Metrics' tab and return the LATEST snapshot row per
    provider (since each save appends a new timestamped row rather than
    updating in place). Used as a durable fallback for the Score Card's
    live performance boxes (Avg Speed / JSON% / Error% / Unique Responses)
    when the local in-session 'metrics' dict is empty for that provider —
    e.g. right after the app container restarts and metrics.json is wiped,
    even though real generation activity already happened and was synced."""
    cache = st.session_state.get("_gsheet_auto_metrics_cache")
    now = __import__("time").time()
    if cache and (now - cache.get("ts", 0) < 60):
        return cache["latest"]
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    latest = {}
    if not client or not sheet_id:
        return latest
    try:
        sheet = client.open_by_key(sheet_id)
        ws = sheet.worksheet("Auto Metrics")
        records = ws.get_all_records()
        for rec in records:
            p = str(rec.get("provider", "")).strip()
            if not p:
                continue
            latest[p] = rec  # later rows overwrite earlier ones -> last wins
    except Exception:
        latest = {}
    st.session_state["_gsheet_auto_metrics_cache"] = {"ts": now, "latest": latest}
    return latest

def pull_runs_from_gsheet():
    """Read every row back from the 'Cycle Ratings' tab and reshape it into
    the exact same record dicts load_runs()/save_run() use locally. This is
    what lets the Score Card / Manual Ratings counts / Excel export keep
    working even after the app's own container restarts and wipes
    runs.json — Google Sheets is the durable source of truth, the local
    file is just a fast first-read cache. Cached for 60s per session so
    we don't re-fetch from the Sheets API on every single rerun."""
    cache = st.session_state.get("_gsheet_runs_cache")
    now = __import__("time").time()
    if cache and (now - cache.get("ts", 0) < 60):
        return cache["rows"]
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    if not client or not sheet_id:
        return []
    rows = []
    try:
        sheet = client.open_by_key(sheet_id)
        ws = sheet.worksheet("Cycle Ratings")
        records = ws.get_all_records()
        numeric_fields = ["overall", "auto_difficulty", "auto_arabic", "auto_quality",
                           "auto_medical", "n_auto_emails", "avg_speed", "json_rate",
                           "error_rate"]
        for rec in records:
            row = dict(rec)
            for f in numeric_fields:
                v = row.get(f)
                if v in ("", None):
                    row[f] = None
                else:
                    try:
                        row[f] = float(v) if f in ("avg_speed",) else int(float(v))
                    except (ValueError, TypeError):
                        row[f] = None
            rows.append(row)
    except Exception:
        rows = []
    st.session_state["_gsheet_runs_cache"] = {"ts": now, "rows": rows}
    return rows

def merge_local_and_gsheet_runs(local_runs):
    """Combine local runs.json entries with whatever's in the durable
    Google Sheet copy, deduplicating by (timestamp, provider, language)
    so the same cycle saved locally AND already synced to the sheet
    doesn't get double-counted."""
    sheet_runs = pull_runs_from_gsheet()
    if not sheet_runs:
        return local_runs
    seen = {(r.get("timestamp"), r.get("provider"), r.get("language")) for r in local_runs}
    merged = list(local_runs)
    for r in sheet_runs:
        key = (r.get("timestamp"), r.get("provider"), r.get("language"))
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged

# =============================================================
# SYSTEMATIC ROTATION PLAN — 10 cycles, balanced role x difficulty
# -------------------------------------------------------------
# Purely informational for the researcher: she runs cycles 1-10
# manually following this table (role + difficulty to pick on the
# main app) so that, across 10 cycles, all 4 roles and 3 difficulty
# levels get reasonably balanced coverage instead of relying on
# pure chance with a small sample size.
# =============================================================
ROTATION_PLAN = [
    {"cycle": 1,  "role_en": "Clinical",   "role_ar": "سريري",  "difficulty": "easy",   "language": "English"},
    {"cycle": 2,  "role_en": "Admin",      "role_ar": "إداري",  "difficulty": "medium", "language": "English"},
    {"cycle": 3,  "role_en": "IT",         "role_ar": "تقني",   "difficulty": "hard",   "language": "English"},
    {"cycle": 4,  "role_en": "Other",      "role_ar": "أخرى",  "difficulty": "easy",   "language": "English"},
    {"cycle": 5,  "role_en": "Clinical",   "role_ar": "سريري",  "difficulty": "medium", "language": "English"},
    {"cycle": 6,  "role_en": "Admin",      "role_ar": "إداري",  "difficulty": "hard",   "language": "Arabic"},
    {"cycle": 7,  "role_en": "IT",         "role_ar": "تقني",   "difficulty": "easy",   "language": "Arabic"},
    {"cycle": 8,  "role_en": "Other",      "role_ar": "أخرى",  "difficulty": "medium", "language": "Arabic"},
    {"cycle": 9,  "role_en": "Clinical",   "role_ar": "سريري",  "difficulty": "hard",   "language": "Arabic"},
    {"cycle": 10, "role_en": "Admin",      "role_ar": "إداري",  "difficulty": "easy",   "language": "Arabic"},
]

# =============================================================
# AUTOMATIC EVALUATION — difficulty conformance, Arabic quality,
# general quality, medical relevance. All computed from the
# generated text itself, no human judgement involved.
# =============================================================
_DIRECT_PASSWORD_RE = re.compile(r"\b(password|otp|one[- ]time code|reply with your)\b|كلمة\s*(السر|المرور)|الرمز\s*المؤقت", re.I)
_DIRECT_THREAT_RE    = re.compile(r"\b(suspend(ed)?|terminat(e|ed|ion)|legal action|account.*delet|deactivat)\b|تعليق|إيقاف|إنهاء|حذف\s*الحساب", re.I)
_IMMEDIATE_URGENCY_RE = re.compile(r"\b(immediately|right now|within\s*(1|2|3)\s*hour|asap)\b|فورًا|الآن|خلال\s*ساعت", re.I)
_WINDOW_URGENCY_RE    = re.compile(r"\b(24|48|72)\s*hours?\b|٢٤|٤٨|٧٢\s*ساعة", re.I)
_GENERIC_GREETING_RE  = re.compile(r"^(dear (staff|team|healthcare professional|doctor|valued)|dear sir|عزيزي الموظف|إلى من يهمه)", re.I)
_PERSONAL_GREETING_RE = re.compile(r"dear dr\.?\s+[a-z\u0600-\u06ff]+\s+[a-z\u0600-\u06ff]+|عزيزي\s+د\.?\s*[\u0600-\u06ff]+\s+[\u0600-\u06ff]+", re.I)
_MEDICAL_KEYWORDS = [
    "patient","hospital","clinical","emr","medical","pharmacy","lab","radiology",
    "doctor","dr.","nurse","ministry of health","moh","healthcare","health record",
    "مريض","المستشفى","سريري","نظام طبي","صيدلية","مخبر","أشعة","طبيب","ممرض",
    "وزارة الصحة","صحي","سجل صحي","طبية","عيادة",
]
_PLACEHOLDER_LEFTOVER_RE = re.compile(r"\[QR(?:\s*Code)?\s*:?|suspicious_link\s*:|suspicious_text\s*:", re.I)

def check_difficulty_conformance(result, difficulty, is_phishing=True):
    """Score 0-100: how well the generated email matches the 9 textual
    rules that were supposed to drive generation for this difficulty
    level. Same 9 dimensions used as generation instructions, now used
    as a post-hoc automatic check instead of a manual slider."""
    if not isinstance(result, dict):
        return None
    body = str((result.get("body") or ""))
    subject = str((result.get("subject") or ""))
    frm = str((result.get("from") or ""))
    indicators = result.get("indicators", []) if isinstance(result.get("indicators"), list) else []
    text = f"{subject} {body}"

    domain_match = re.search(r"@([\w.-]+)>?", frm)
    domain = (domain_match.group(1) if domain_match else "").lower()
    domain_obvious = any(w in domain for w in ADVANCED_BANNED_DOMAIN_WORDS)

    checks = []  # each: True = conforms to expectation for this difficulty
    if not is_phishing:
        # Legitimate emails: just check they avoid red flags regardless of level.
        checks.append(not _DIRECT_PASSWORD_RE.search(text))
        checks.append(not _DIRECT_THREAT_RE.search(text))
        checks.append(not domain_obvious)
        return round(sum(checks) / len(checks) * 100) if checks else None

    if difficulty == "easy":
        checks.append(bool(_GENERIC_GREETING_RE.search(body.strip())))
        checks.append(domain_obvious)
        checks.append(bool(_DIRECT_PASSWORD_RE.search(text)))
        checks.append(bool(_DIRECT_THREAT_RE.search(text)))
        checks.append(bool(_IMMEDIATE_URGENCY_RE.search(text)))
        checks.append(len(indicators) >= 3)
    elif difficulty == "hard":
        checks.append(bool(_PERSONAL_GREETING_RE.search(body.strip())))
        checks.append(not domain_obvious)
        checks.append(not _DIRECT_PASSWORD_RE.search(text))
        checks.append(not _DIRECT_THREAT_RE.search(text))
        checks.append(not _IMMEDIATE_URGENCY_RE.search(text))
        checks.append(len(indicators) <= 3)
    else:  # medium
        checks.append(not _GENERIC_GREETING_RE.search(body.strip()))
        checks.append(not _DIRECT_PASSWORD_RE.search(text))
        checks.append(not _DIRECT_THREAT_RE.search(text))
        checks.append(bool(_WINDOW_URGENCY_RE.search(text)) or not _IMMEDIATE_URGENCY_RE.search(text))
        checks.append(len(indicators) <= 3)

    caps_words = re.findall(r"\b[A-Z]{4,}\b", body)
    excl_count = body.count("!")
    if difficulty == "easy":
        checks.append((len(caps_words) + excl_count) >= 1)
    else:
        checks.append((len(caps_words) + excl_count) == 0)

    return round(sum(checks) / len(checks) * 100) if checks else None

def check_arabic_quality(result, is_ar):
    """Score 0-100: cheap proxy for Arabic text cleanliness — leftover
    Latin words, broken/garbled characters, obviously truncated text."""
    if not is_ar or not isinstance(result, dict):
        return None
    body = str((result.get("body") or ""))
    if not body.strip():
        return 0
    words = body.split()
    if not words:
        return 0
    latin_words = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
    latin_ratio = len(latin_words) / len(words)
    score = 100
    score -= min(60, int(latin_ratio * 200))            # too many stray English words
    if re.search(r"\ufffd|\\u[0-9a-fA-F]{4}", body):     # mojibake / unescaped unicode
        score -= 30
    if not re.search(r"[.!؟،]\s*$", body.strip()):       # doesn't end cleanly
        score -= 10
    if re.search(r"(\b\w+\b)(\s+\1){2,}", body):          # same word repeated 3+ times in a row
        score -= 20
    return max(0, min(100, score))

def check_general_quality(result):
    """Score 0-100: structural completeness proxy — required fields
    present, reasonable length, enough indicators, no leftover template
    artifacts that should have been rendered (e.g. raw '[QR:' text)."""
    if not isinstance(result, dict):
        return 0
    score = 100
    for field in ["from", "subject", "body"]:
        if not str(result.get(field, "")).strip():
            score -= 25
    body = str((result.get("body") or ""))
    wc = len(body.split())
    if wc < 15:
        score -= 25
    elif wc > 220:
        score -= 10
    indicators = result.get("indicators", []) if isinstance(result.get("indicators"), list) else []
    if len(indicators) < 2:
        score -= 20
    if _PLACEHOLDER_LEFTOVER_RE.search(body):
        score -= 25
    return max(0, min(100, score))

def check_medical_relevance(result):
    """Score 0/100 (with partial credit) — does this email actually sit
    in a healthcare/hospital context, based on keyword presence."""
    if not isinstance(result, dict):
        return None
    text = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body"]).lower()
    hits = sum(1 for kw in _MEDICAL_KEYWORDS if kw in text)
    if hits == 0:
        return 0
    if hits == 1:
        return 60
    return 100

# =============================================================
# PERSISTENT AUTO-EVAL LOG + PENDING-CYCLE BUCKET
# -------------------------------------------------------------
# auto_eval.json   : permanent raw log, one row per generated email,
#                    with its 4 automatic scores — used for long-term
#                    analysis / CSV export regardless of cycles.
# pending_cycle.json: per (provider, language) bucket of scores
#                    collected SINCE the last saved cycle rating, so
#                    that when the researcher finishes a full cycle
#                    (6 learning + 10 assessment) and rates it, we can
#                    snapshot the average automatic scores for THAT
#                    specific cycle into the run record.
# =============================================================
_AUTO_EVAL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_eval.json")
_PENDING_CYCLE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_cycle.json")
_PENDING_PERF_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_perf.json")

def _load_json_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _save_json_list(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =============================================================
# FULL EXCEL EXPORT — one workbook, several sheets:
#   - Summary: one row per (provider, language) with all 9 metrics
#   - One sheet PER PROVIDER (Groq/Claude/OpenAI/Gemini) with its own
#     10-cycle detailed breakdown — this is the part that keeps the
#     four providers clearly separated instead of mixed in one table.
#   - Raw_Email_Log: every individually auto-scored email (not just
#     cycle averages), useful for deeper statistical analysis.
#   - Rotation_Plan: the systematic 10-cycle role/difficulty/language plan.
# Built fully in-memory and offered via st.download_button, so once the
# researcher downloads it, the file lives on her own computer — totally
# independent of the app's server storage from that point on.
# =============================================================
def build_excel_export():
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    import io as _io

    runs = load_runs()
    auto_log = _load_json_list(_AUTO_EVAL_FILE_PATH)

    PROV_ORDER_X = ["groq", "anthropic", "openai", "gemini"]
    PROV_LABELS_X = {"groq": "Groq", "anthropic": "Claude", "openai": "OpenAI", "gemini": "Gemini"}

    HEADER_FILL = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
    HEADER_FONT = Font(bold=True, color="FFFFFF")

    def style_header(ws, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"

    def autosize(ws, ncols, width=16):
        for c in range(1, ncols + 1):
            ws.column_dimensions[get_column_letter(c)].width = width

    wb = Workbook()

    # ---- Sheet 1: Summary across all 4 providers x 2 languages ----
    ws = wb.active
    ws.title = "Summary"
    headers = ["Provider", "Language", "Cycles", "Avg Speed (s)", "JSON %", "Error %",
               "Difficulty %", "Arabic %", "Quality %", "Medical %", "Overall (avg/5)"]
    ws.append(headers)
    for p in PROV_ORDER_X:
        for lang in ["English", "Arabic"]:
            p_runs = [r for r in runs if r.get("provider") == p and r.get("language") == lang]
            def avgf(field):
                vals = [r.get(field) for r in p_runs if r.get(field) is not None]
                return round(sum(vals) / len(vals), 2) if vals else None
            ws.append([
                PROV_LABELS_X[p], lang, len(p_runs),
                avgf("avg_speed"), avgf("json_rate"), avgf("error_rate"),
                avgf("auto_difficulty"), avgf("auto_arabic"), avgf("auto_quality"),
                avgf("auto_medical"), avgf("overall"),
            ])
    style_header(ws, len(headers))
    autosize(ws, len(headers))

    # ---- One sheet per provider: its own 10-cycle breakdown ----
    cycle_headers = ["#", "Timestamp", "Language", "Avg Speed (s)", "JSON %", "Error %",
                      "Diversity", "Difficulty %", "Arabic %", "Quality %", "Medical %",
                      "Overall /5", "Note"]
    for p in PROV_ORDER_X:
        ws_p = wb.create_sheet(PROV_LABELS_X[p])
        ws_p.append(cycle_headers)
        p_runs_en = [r for r in runs if r.get("provider") == p and r.get("language") == "English"]
        p_runs_ar = [r for r in runs if r.get("provider") == p and r.get("language") == "Arabic"]
        for i, r in enumerate(p_runs_en + p_runs_ar, 1):
            ws_p.append([
                i, r.get("timestamp"), r.get("language"), r.get("avg_speed"),
                r.get("json_rate"), r.get("error_rate"), r.get("diversity"),
                r.get("auto_difficulty"), r.get("auto_arabic"), r.get("auto_quality"),
                r.get("auto_medical"), r.get("overall"), r.get("note"),
            ])
        style_header(ws_p, len(cycle_headers))
        autosize(ws_p, len(cycle_headers))

    # ---- Raw per-email auto-evaluation log (all providers together, filterable) ----
    ws_raw = wb.create_sheet("Raw_Email_Log")
    raw_headers = ["Timestamp", "Provider", "Language", "Difficulty Level",
                   "Difficulty Score %", "Arabic Score %", "Quality Score %", "Medical Score %"]
    ws_raw.append(raw_headers)
    for rec in auto_log:
        ws_raw.append([
            rec.get("timestamp"), PROV_LABELS_X.get(rec.get("provider"), rec.get("provider")),
            rec.get("language"), rec.get("difficulty"),
            rec.get("difficulty_score"), rec.get("arabic_score"),
            rec.get("quality_score"), rec.get("medical_score"),
        ])
    style_header(ws_raw, len(raw_headers))
    autosize(ws_raw, len(raw_headers))

    # ---- Rotation plan reference ----
    ws_rot = wb.create_sheet("Rotation_Plan")
    rot_headers = ["Cycle #", "Role", "Difficulty", "Language"]
    ws_rot.append(rot_headers)
    for plan in ROTATION_PLAN:
        ws_rot.append([plan["cycle"], plan["role_en"], plan["difficulty"].capitalize(), plan["language"]])
    style_header(ws_rot, len(rot_headers))
    autosize(ws_rot, len(rot_headers), width=14)

    buf = _io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def _load_json_dict(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def _save_json_dict(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_pending_buckets():
    return _load_json_dict(_PENDING_CYCLE_FILE_PATH)

def _save_pending_buckets(buckets):
    _save_json_dict(_PENDING_CYCLE_FILE_PATH, buckets)

def snapshot_and_clear_pending_perf(provider, language):
    """Average the per-cycle performance bucket (speed/JSON/error/
    diversity) collected since the last saved cycle, for THIS provider
    and language, then clear it so the next cycle starts fresh — mirrors
    snapshot_and_clear_pending_cycle() but for the 4 performance metrics."""
    buckets = _load_json_dict(_PENDING_PERF_FILE_PATH)
    key = f"{provider}__{language}"
    items = buckets.get(key, [])

    speeds = [it["speed"] for it in items if it.get("speed") is not None]
    json_ok = sum(1 for it in items if it.get("json_success") is True)
    json_fail = sum(1 for it in items if it.get("json_success") is False)
    errors = sum(1 for it in items if it.get("is_error"))
    calls = len(items)
    hashes = list({it["hash"] for it in items if it.get("hash")})

    snap = {
        "n_calls": calls,
        "avg_speed": round(sum(speeds)/len(speeds), 2) if speeds else None,
        "json_rate": round(json_ok/(json_ok+json_fail)*100) if (json_ok+json_fail) > 0 else None,
        "error_rate": round(errors/calls*100) if calls > 0 else None,
        "diversity": f"{len(hashes)}/{calls}" if calls > 0 else None,
    }
    if key in buckets:
        del buckets[key]
        _save_json_dict(_PENDING_PERF_FILE_PATH, buckets)
    return snap

def evaluate_and_log_auto_scores(result, difficulty, language, is_phishing=True):
    """Called right after a learning/assessment email is generated.
    Computes the 4 automatic scores and logs them both permanently
    (auto_eval.json) and into the current pending-cycle bucket so the
    next saved manual rating can snapshot this cycle's averages."""
    if not isinstance(result, dict) or "error" in result:
        return
    provider = st.session_state.get("ai_provider", "groq")
    is_ar = (language == "Arabic")
    rec = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "language": language,
        "difficulty": difficulty,
        "difficulty_score": check_difficulty_conformance(result, difficulty, is_phishing),
        "arabic_score": check_arabic_quality(result, is_ar),
        "quality_score": check_general_quality(result),
        "medical_score": check_medical_relevance(result),
    }
    log = _load_json_list(_AUTO_EVAL_FILE_PATH)
    log.append(rec)
    _save_json_list(_AUTO_EVAL_FILE_PATH, log)

    buckets = _load_pending_buckets()
    key = f"{provider}__{language}"
    buckets.setdefault(key, []).append(rec)
    _save_pending_buckets(buckets)

def snapshot_and_clear_pending_cycle(provider, language):
    """Average the pending bucket for this provider+language into one
    snapshot (used when the researcher saves a holistic cycle rating),
    then clear that bucket so the next cycle starts fresh."""
    buckets = _load_pending_buckets()
    key = f"{provider}__{language}"
    items = buckets.get(key, [])

    def _avg(field):
        vals = [it[field] for it in items if it.get(field) is not None]
        return round(sum(vals)/len(vals), 1) if vals else None

    snapshot = {
        "n_emails": len(items),
        "difficulty_score": _avg("difficulty_score"),
        "arabic_score": _avg("arabic_score"),
        "quality_score": _avg("quality_score"),
        "medical_score": _avg("medical_score"),
    }
    if key in buckets:
        del buckets[key]
        _save_pending_buckets(buckets)
    return snapshot

for k, v in [("language","English"),("page","home"),("role",""),
              ("example_index",0),("emails",{}),("difficulty","medium"),
              ("user_name",""),("user_email",""),
              ("ai_provider", load_persistent_provider("openai")),
              ("admin_authenticated",False),
              ("metrics", load_metrics_file()),  # {provider: {speed:[], json_ok:int, json_fail:int, errors:int, calls:int, hashes:[]}} — loaded from disk so it survives refresh
              ("manual_ratings",{}),  # legacy in-session structure, kept for backward compatibility
             ]:
    if k not in st.session_state:
        st.session_state[k] = v

_nav = (st.query_params.get("nav") or "")
if _nav in ("login", "register"):
    st.session_state["login_mode"] = _nav
    st.session_state["page"] = "login"
    _lang = (st.query_params.get("lang") or "")
    if _lang in ("Arabic", "English"):
        st.session_state["language"] = _lang
    st.query_params.clear()

def set_language(lang):
    st.session_state["language"] = lang
    st.session_state["lang_explicitly_chosen"] = True

def t(en, ar):
    return ar if st.session_state["language"] == "Arabic" else en

def _safe_error_text(err, language):
    """Never show raw API/debug dicts to trainees. Always return a short,
    human-readable sentence, and stash whatever technical detail we have in
    the hidden debug log (visible only to the admin panel / developer)."""
    is_ar = (language == "Arabic")
    try:
        _store_debug("ui_display", err)
    except Exception:
        pass
    msg = err.get("message") if isinstance(err, dict) else str(err)
    msg = str(msg or "")
    # If, despite everything, a low-level/raw message slipped through (very
    # long, or looks like a Python dict/JSON dump), replace it with a clean
    # generic sentence instead of exposing internals.
    looks_raw = len(msg) > 450 or msg.strip().startswith("{") or "Parsed keys/values" in msg
    if looks_raw or not msg.strip():
        return ("تعذّر توليد هذا المحتوى حالياً. يرجى الضغط على (حاول مرة أخرى)."
                if is_ar else
                "This content couldn't be generated right now. Please tap Try Again.")
    return msg

def go_to_learning(role):
    st.session_state["role"]          = role
    st.session_state["page"]          = "learning"
    st.session_state["example_index"] = 0
    st.session_state["emails"]        = {}

def clean_foreign_only(text):
    if not text: return text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff]', '', text)
    text = re.sub(r'[\u0400-\u04ff]', '', text)
    text = re.sub(r'[\u0100-\u017f]', '', text)
    text = re.sub(r'[\u1ea0-\u1ef9]', '', text)
    text = re.sub(r'[\u0900-\u097f]', '', text)
    text = re.sub(r'[\u0e00-\u0e7f]', '', text)
    text = re.sub(r'[\u10a0-\u10ff\u0530-\u058f\u05d0-\u05ff]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text

_ALLOWED_LATIN_RE = re.compile(
    r'^(https?://[^\s]+|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}|\.(?:pdf|xlsx|docx|txt|csv|zip|exe)|[0-9]+)$',
    re.IGNORECASE
)

def remove_foreign_latin_words(text):
    if not text: return text
    arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
    total_chars  = len(re.sub(r'\s', '', text))
    if total_chars == 0 or arabic_chars / total_chars < 0.25:
        return text
    def keep_token(tok):
        if _ALLOWED_LATIN_RE.match(tok): return tok
        if re.search(r'[\u0600-\u06ff]', tok): return tok
        if re.match(r'^[\u060c\u061b\u061f،؛؟!.,;:\-\u2013\u2014()\[\]{}\'"]+$', tok): return tok
        if re.match(r'^[a-zA-Z\u00c0-\u024f]+$', tok): return ''
        return tok
    tokens  = re.split(r'(\s+)', text)
    cleaned = ''.join(keep_token(t) for t in tokens)
    cleaned = re.sub(r'[a-zA-Z]{1,}[-_](?=[\u0600-\u06ff])', '', cleaned)
    cleaned = re.sub(r'(?<=[\u0600-\u06ff])[-_]?[a-zA-Z]{1,}', '', cleaned)
    cleaned = re.sub(r'[a-zA-Z]{1,}(?=[\u0600-\u06ff])', '', cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    cleaned = re.sub(r'\s([،؛،,.;:])', r'\1', cleaned)
    return cleaned

def clean_email_field(addr):
    if not addr: return addr
    addr = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0400-\u04ff\u0100-\u017f]', '', addr)
    addr = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', addr)
    return addr.strip()

def extract_to_email(to_val):
    if not to_val: return 'employee@hospital.org'
    m = re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', to_val)
    return m.group(0) if m else 'employee@hospital.org'

def fix_json_newlines(s):
    result, in_string, i = [], False, 0
    while i < len(s):
        c = s[i]
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            in_string = not in_string
        if in_string and c == '\n':   result.append('\\n')
        elif in_string and c == '\r': result.append('\\r')
        elif in_string and c == '\t': result.append('\\t')
        else:                         result.append(c)
        i += 1
    return ''.join(result)

ROLE_MAP = {
    "سريري": (
        "ممرض أو طبيب يعمل في مستشفى",
        "السجلات الطبية وجداول العمل السريرية والأنظمة الطبية وبيانات المرضى",
        "clinical"
    ),
    "إداري / إدارة": (
        "موظف إداري في مستشفى (سكرتارية طبية، استقبال، إدارة ملفات المرضى، التأمين الصحي، الفوترة الطبية)",
        "ملفات المرضى وحجز المواعيد والتأمين الصحي والفوترة الطبية وسكرتارية الأطباء وإدارة المستشفى والمشتريات الطبية",
        "admin"
    ),
    "تقنية المعلومات / المعلوماتية": (
        "متخصص تقنية معلومات في مستشفى",
        "الشبكة والخوادم وتحديثات البرامج والدعم التقني والأمن السيبراني",
        "it"
    ),
    "Clinical": (
        "a nurse or doctor in a hospital",
        "patient records, EMR systems, clinical schedules, medical data",
        "clinical"
    ),
    "Admin / Management": (
        "a healthcare administrative staff (medical secretary, receptionist, patient records manager, insurance coordinator, billing specialist)",
        "patient files, appointment scheduling, medical insurance, hospital billing, doctor's secretary, hospital management, medical procurement",
        "admin"
    ),
    "IT / Informatics": (
        "an IT specialist in a hospital",
        "VPN, network, servers, software updates, IT helpdesk, security systems",
        "it"
    ),
    "Other": (
        "a general hospital employee in Saudi Arabia",
        "any hospital department — clinical, administrative, or technical",
        "other"
    ),
    "أخرى": (
        "موظف عام في مستشفى سعودي",
        "أي قسم في المستشفى — سريري أو إداري أو تقني",
        "other"
    ),
}

EN_NAMES = {
    "clinical": [
        "dr.sarah.almutairi@hospital.org",
        "dr.ahmed.alotaibi@hospital.org",
        "n.noura.alshamri@hospital.org",
        "dr.fahad.aldosari@hospital.org",
        "n.mona.alharbi@hospital.org",
        "dr.khalid.alanazi@hospital.org",
    ],
    "admin": [
        "m.reem.alsabiei@hospital.org",
        "m.abdullah.alqahtani@hospital.org",
        "m.hind.alrashidi@hospital.org",
        "m.sultan.alghamdi@hospital.org",
        "m.dalal.alzahrani@hospital.org",
        "m.omar.albaqami@hospital.org",
    ],
    "it": [
        "t.mohammed.alshahri@hospital.org",
        "t.rania.almalki@hospital.org",
        "t.yusuf.aljuhani@hospital.org",
        "t.lama.alumari@hospital.org",
        "t.bandar.althubaiti@hospital.org",
        "t.nadia.alsalmi@hospital.org",
    ],
    "other": [
        "s.khalid.alharbi@hospital.org",
        "s.sara.alqahtani@hospital.org",
        "s.faisal.alzahrani@hospital.org",
        "s.nora.alotaibi@hospital.org",
        "s.ahmed.alshamri@hospital.org",
        "s.hessa.aldosari@hospital.org",
    ],
}





def get_recipient(role, index, language, phase="learn"):
    # EN: THIS is the real root cause of the recurring "Dear Nurse John"
    # name-mismatch bug. The model was correctly told in the prompt to use
    # a specific (name, email) pair — but AFTER generation, this function
    # used to silently overwrite the "to" field with a DIFFERENT, totally
    # unrelated email picked from EN_NAMES below. The greeting name inside
    # the body (bound to the prompt's pair) and the final "to" address
    # (overwritten here) ended up belonging to two different people.
    #
    # Fix: read from the SAME RECIPIENT_POOLS + the SAME session-randomized
    # order key that build_prompt/build_assess_prompt used to pick the pair
    # for this exact role_type + index + phase, so the override always
    # matches what the model was actually told to write in the greeting.
    #
    # AR: هذا هو السبب الحقيقي لخلل "Dear Nurse John" المتكرر. الموديل كان
    # يُطلب منه صراحة استخدام زوج (اسم، بريد) معيّن بالتعليمات — لكن بعد
    # التوليد، هذي الدالة كانت "تستبدل" حقل "to" بصمت بعنوان مختلف كليًا
    # من قائمة EN_NAMES تحت، فيصير اسم التحية بالنص (المرتبط بالزوج الأصلي)
    # وعنوان "to" النهائي (المُستبدل هنا) يخصان شخصين مختلفين تمامًا.
    #
    # الحل: نقرأ من نفس RECIPIENT_POOLS وبنفس مفتاح الترتيب العشوائي للجلسة
    # اللي استخدمته build_prompt/build_assess_prompt لاختيار الزوج لنفس
    # role_type + index + المرحلة، حتى الاستبدال هنا يطابق دائمًا ما طُلب
    # من الموديل فعليًا يكتبه بالتحية.
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        # EN: in practice, role_type "other" is handled entirely by
        # generate_other_email/generate_other_assess_email (static
        # templates) and never reaches this function. This is just a
        # safe fallback so we never crash if that ever changes.
        # AR: عمليًا دور "Other" تتم معالجته بالكامل بدوال القوالب الثابتة
        # المنفصلة ومايصل لهذي الدالة أبدًا. هذا احتياط آمن بس لو تغيّر هذا مستقبلًا.
        cached = st.session_state.get(f"_other_recipient_{index}")
        if cached:
            return cached
        legacy = EN_NAMES.get("other", EN_NAMES["clinical"])
        return legacy[index % len(legacy)]
    pool = RECIPIENT_POOLS.get(role_type)
    if not pool:
        legacy = EN_NAMES.get(role_type, EN_NAMES["clinical"])
        return legacy[index % len(legacy)]
    order_key = f"recipient_order_{role_type}" if phase == "learn" else f"assess_recipient_order_{role_type}"
    order = get_session_random_order(len(pool), order_key)
    return pool[order[index % len(order)]]["email"]

PHISHING_SCENARIOS = [
    {"key":"link",    "en_type":"Credential Harvesting Link",              "ar_type":"رابط سرقة بيانات الدخول",             "has_attachment":False, "attachment_ext":"", "has_link":True},
    {"key":"pdf",     "en_type":"Malicious PDF Attachment",                "ar_type":"مرفق PDF خبيث",                        "has_attachment":True,  "attachment_ext":".pdf", "has_link":False},
    {"key":"xlsx",    "en_type":"Malicious Excel Attachment",              "ar_type":"مرفق Excel خبيث",                     "has_attachment":True,  "attachment_ext":".xlsx","has_link":False},
    {"key":"docx",    "en_type":"Malicious Word Document - Enable Macros", "ar_type":"مرفق Word مزيف - تفعيل الماكرو",      "has_attachment":True,  "attachment_ext":".docx","has_link":False},
    {"key":"hr_link", "en_type":"Fake HR Announcement with Link",          "ar_type":"إعلان موارد بشرية مزيف برابط",        "has_attachment":False, "attachment_ext":"", "has_link":True},
    {"key":"exec",    "en_type":"Executive Impersonation - Urgent Request","ar_type":"انتحال هوية المدير - طلب عاجل",       "has_attachment":False, "attachment_ext":"", "has_link":False},
]

def get_shuffled_scenario_order():
    if "scenario_order" not in st.session_state:
        order = list(range(len(PHISHING_SCENARIOS)))
        random.shuffle(order)
        st.session_state["scenario_order"] = order
    return st.session_state["scenario_order"]

# =============================================================
# EN: SESSION-RANDOMIZED ORDER HELPER (variety fix)
# ---------------------------------------------------------------
# Problem we found during testing: "Example #3 of 6" was always
# mapped to the SAME scenario (e.g. "MOH protocol update") no
# matter how many times a user logged in again, because the code
# picked scenarios with `forced[index % len(forced)]` — a fixed,
# deterministic mapping. Across two different sessions, slot #3
# came out nearly identical.
#
# Fix: instead of a fixed mapping, we shuffle the list of indices
# ONCE per session and cache that shuffled order in session_state.
# Slot #3 now points to a DIFFERENT scenario each new session,
# while still guaranteeing no two of the 6 examples in the SAME
# session repeat each other (since it's a permutation, not a
# random pick-with-replacement).
#
# AR: دالة لعشوائية الترتيب لكل جلسة (إصلاح مشكلة التنوع)
# ---------------------------------------------------------------
# المشكلة اللي لاحظناها بالاختبار: "مثال 3 من 6" كان دايمًا يطلع
# له نفس السيناريو (مثل "تحديث بروتوكول وزارة الصحة") بغض النظر
# عن عدد مرات تسجيل الدخول، لأن الكود القديم كان يحدد السيناريو
# بمعادلة ثابتة `forced[index % len(forced)]` — يعني ربط دائم.
#
# الحل: نخلط ترتيب الفهارس مرة واحدة فقط لكل جلسة ونحفظه في
# session_state. الآن "مثال 3" يطلع له سيناريو مختلف كل جلسة جديدة،
# مع ضمان عدم تكرار أي سيناريو مرتين داخل نفس الجلسة (لأنه تبديل
# ترتيب "permutation"، مش اختيار عشوائي قد يتكرر).
# =============================================================
def get_session_random_order(pool_size, session_key):
    if session_key not in st.session_state:
        order = list(range(pool_size))
        random.shuffle(order)
        st.session_state[session_key] = order
    return st.session_state[session_key]

# =============================================================
# =============================================================
# FIX 7: FORCED SCENARIO DIVERSITY PER INDEX
# كل مثال له سيناريو محدد مسبقاً — يمنع التكرار نهائياً
# =============================================================
FORCED_SCENARIOS = {
    "admin": [
        {"en": "SCENARIO: Fake supplier invoice — impersonate a medical equipment supplier requesting urgent payment. IMPORTANT: vary every time — change supplier name (MedSupply Co./Al-Rashid Medical/Gulf Medical Supplies), equipment type (surgical instruments/radiology equipment/lab supplies/ICU monitors), invoice amount (SAR 75,000-200,000), and PDF filename each time.",
         "ar": "السيناريو: فاتورة مورد مزيفة — انتحل هوية مورد معدات طبية يطلب دفع 150,000 ريال بشكل عاجل. استخدم مرفق PDF باسم invoice.pdf."},
        {"en": "SCENARIO: Fake health insurance portal — re-verify coverage details via suspicious link. IMPORTANT: vary the insurance provider name (Tawuniya/Bupa Arabia/MedGulf/AXA), specific claim type (annual renewal/coverage update/reimbursement), and suspicious link URL each time.",
         "ar": "السيناريو: بوابة تأمين صحي مزيفة — ادّعِ أن نظام تأمين الموظفين يحتاج تحديث بيانات التغطية عبر رابط مشبوه. بدون مرفق."},
        {"en": "SCENARIO: Fake payroll system — update bank account details to avoid delayed salary. IMPORTANT: vary the urgency deadline (end of month/before 15th/within 48 hours), bank detail type requested (IBAN/account number/branch code), and sender name each time.",
         "ar": "السيناريو: نظام رواتب مزيف — ادّعِ أن نظام صرف الرواتب يحتاج تحديث بيانات الحساب البنكي لتجنب تأخر الراتب. بدون مرفق."},
        {"en": "SCENARIO: CEO/director impersonation — impersonate the hospital CEO urgently requesting the admin manager to process a financial transfer or share sensitive payroll data immediately. Pure social engineering, NO link, NO attachment.",
         "ar": "السيناريو: انتحال هوية المدير التنفيذي — تظاهر بأنك المدير التنفيذي وتطلب من المدير الإداري تحويل مالي عاجل أو مشاركة بيانات الرواتب. هندسة اجتماعية فقط."},
        {"en": "SCENARIO: Fake patient appointment system migration — claim the appointment booking system is being upgraded and staff must verify login credentials through a suspicious link. NO attachment.",
         "ar": "السيناريو: ترحيل مزيف لنظام حجز المواعيد — ادّعِ أن النظام يُرقَّى ويطلب التحقق من بيانات الدخول عبر رابط مشبوه."},
        {"en": "SCENARIO: Fake medical procurement portal — impersonate procurement system admin claiming a critical supplier contract expires this week and must be renewed via a suspicious link. NO attachment.",
         "ar": "السيناريو: بوابة مشتريات طبية مزيفة — انتحل هوية مسؤول المشتريات وادّعِ أن عقد مورد حيوي ينتهي هذا الأسبوع ويجب تجديده عبر رابط مشبوه."},
        # EN: 4 extra admin scenarios added so the scenario pool has more
        # raw material to randomly draw from each session (variety fix).
        # AR: 4 سيناريوهات إدارية إضافية لتوسيع مساحة الاختيار العشوائي كل جلسة.
        {"en": "SCENARIO: Fake annual leave/vacation balance system — claim the employee's leave balance system needs urgent re-verification via a suspicious link before year-end. NO attachment.",
         "ar": "السيناريو: نظام رصيد إجازات مزيف — ادّعِ أن نظام الإجازات يحتاج إعادة تحقق عاجلة عبر رابط مشبوه قبل نهاية السنة."},
        {"en": "SCENARIO: Fake employee benefits enrollment — claim annual benefits enrollment (housing allowance/transportation) is closing soon and requires immediate confirmation via a suspicious portal. NO attachment.",
         "ar": "السيناريو: تسجيل مزايا موظفين مزيف — ادّعِ أن التسجيل في المزايا السنوية (بدل سكن/نقل) يقفل قريبًا ويحتاج تأكيد فوري عبر بوابة مشبوهة."},
        {"en": "SCENARIO: Malicious vendor contract Word document — send a fake supplier or vendor contract renewal as a Word attachment requiring macro enablement to view terms.",
         "ar": "السيناريو: مستند Word خبيث لعقد مورد — أرسل تجديد عقد مورد مزيف كمرفق Word يطلب تفعيل الماكرو لعرض الشروط."},
        {"en": "SCENARIO: Fake hospital management system migration — claim the administrative records system is migrating to a new platform and staff must re-confirm their login via a suspicious link. NO attachment.",
         "ar": "السيناريو: ترحيل مزيف لنظام إدارة المستشفى — ادّعِ أن نظام السجلات الإدارية ينتقل لمنصة جديدة ويطلب إعادة تأكيد الدخول عبر رابط مشبوه."},
    ],
    "clinical": [
        {"en": "SCENARIO: Fake EMR system credential harvest — claim the hospital EMR system requires urgent re-verification of login credentials through a suspicious link. IMPORTANT: vary the details every time — change the sender name, the specific system name (EMR/Patient Portal/Clinical System), the suspicious link URL, and the spelling mistake used (choose ONE from: credintials/urgant/acces/imediatly — never reuse recived or procedue).",
         "ar": "السيناريو: سرقة بيانات نظام السجلات الطبية. مهم: غيّر التفاصيل في كل مرة — اسم المرسل، اسم النظام، الرابط المشبوه، والخطأ الإملائي (اختر من: تسجيـل/عاجلة/وصلت — لا تكرر نفس الخطأ)."},
        {"en": "SCENARIO: Malicious patient data PDF — send a fake urgent patient lab results update as a PDF attachment. IMPORTANT: vary every time — change the patient department (ICU/oncology/cardiology/radiology/pediatrics), the PDF filename, the doctor name, and the spelling mistake (choose ONE from: recieved/attachement/critcal/paicent — never repeat the same error).",
         "ar": "السيناريو: مرفق PDF خبيث لنتائج مختبر. مهم: غيّر القسم (ICU/أورام/قلب/أطفال) واسم الملف والطبيب والخطأ الإملائي في كل مرة."},
        {"en": "SCENARIO: Fake MOH clinical protocol — impersonate MOH sending urgent clinical guidance. IMPORTANT: vary the protocol topic every time (infection control/COVID-19 update/vaccination campaign/MRSA alert/antimicrobial resistance) and use a different suspicious link URL each time.",
         "ar": "السيناريو: بروتوكول سريري مزيف من وزارة الصحة. مهم: غيّر موضوع البروتوكول (مكافحة العدوى/كوفيد/تطعيمات/MRSA) والرابط في كل مرة."},
        {"en": "SCENARIO: Medical director impersonation — impersonate the medical director urgently requesting patient data or system access. IMPORTANT: vary the director name, department (Surgery/Internal Medicine/Emergency/ICU), and specific request each time. Pure social engineering — no link needed.",
         "ar": "السيناريو: انتحال هوية المدير الطبي. مهم: غيّر اسم المدير والتخصص (جراحة/طوارئ/باطنية) والطلب المحدد في كل مرة."},
        {"en": "SCENARIO: Fake clinical staff schedule Excel — send a malicious Excel file with updated duty roster. IMPORTANT: vary the time period (next month/Ramadan schedule/Q2 roster/holiday coverage), Excel filename, and head nurse name each time.",
         "ar": "السيناريو: جدول مناوبات مزيف كمرفق Excel. مهم: غيّر الفترة الزمنية (رمضان/الربع الثاني/الإجازات) واسم الملف في كل مرة."},
        {"en": "SCENARIO: Fake pharmacy or medical system update — claim the pharmacy dispensing system or drug management portal requires urgent login verification. IMPORTANT: vary the system name (Pharmacy System/Drug Dispensing Portal/Medication Management/Blood Bank System) and suspicious link each time.",
         "ar": "السيناريو: تحديث مزيف لنظام الصيدلية أو بنك الدم. مهم: غيّر اسم النظام (صيدلية/بنك الدم/إدارة الدواء) والرابط في كل مرة."},
        # EN: 4 extra clinical scenarios added so "example slot #N" doesn't
        # keep landing on the same scenario across different sessions.
        # AR: 4 سيناريوهات سريرية إضافية حتى ماتفضل "خانة المثال رقم N"
        # تطلع لها نفس السيناريو بكل الجلسات.
        {"en": "SCENARIO: Fake nurse scheduling/shift swap system — claim the shift-swap request system needs urgent credential re-verification via a suspicious link. IMPORTANT: vary the ward name and link each time.",
         "ar": "السيناريو: نظام تبديل المناوبات مزيف — ادّعِ أن نظام طلبات تبديل المناوبات يحتاج إعادة تحقق عاجلة عبر رابط مشبوه. مهم: غيّر اسم القسم والرابط في كل مرة."},
        {"en": "SCENARIO: Fake critical patient vitals alert PDF — send an urgent fake patient vitals/monitoring alert as a PDF attachment requiring immediate review. IMPORTANT: vary the ward (ICU/CCU/ER), patient reference, and PDF filename each time.",
         "ar": "السيناريو: تنبيه حيوي حرج لمريض كمرفق PDF مزيف — أرسل تنبيهًا عاجلاً مزيفًا لمراقبة مريض كمرفق PDF يطلب مراجعة فورية. مهم: غيّر القسم ورقم الحالة واسم الملف في كل مرة."},
        {"en": "SCENARIO: Fake telemedicine/remote consultation platform — claim the hospital's telemedicine system requires urgent credential re-verification via a suspicious link before today's remote consultations. NO attachment.",
         "ar": "السيناريو: منصة استشارات طبية عن بعد مزيفة — ادّعِ أن نظام الاستشارات عن بعد يحتاج إعادة تحقق عاجلة عبر رابط مشبوه قبل استشارات اليوم."},
        {"en": "SCENARIO: Fake blood bank or transfusion system alert — claim an urgent blood bank inventory system update requires immediate login via a suspicious portal. NO attachment.",
         "ar": "السيناريو: تنبيه نظام بنك دم مزيف — ادّعِ أن تحديث نظام مخزون بنك الدم يحتاج تسجيل دخول فوري عبر بوابة مشبوهة."},
    ],
    "it": [
        {"en": "SCENARIO: Fake VPN credential update — claim the hospital VPN gateway requires urgent re-authentication. IMPORTANT: vary the VPN system name (Cisco AnyConnect/FortiClient/Pulse Secure), the suspicious portal URL, and the urgency reason each time.", "ar": "السيناريو: تحديث مزيف لبيانات الـ VPN. مهم: غيّر اسم النظام (Cisco/FortiClient) والرابط والسبب في كل مرة."},
        {"en": "SCENARIO: Fake SSL certificate expiry — claim the hospital website or portal SSL certificate has expired. IMPORTANT: vary the affected system (hospital website/patient portal/EMR login/staff intranet), renewal deadline, and suspicious link each time.", "ar": "السيناريو: تنبيه مزيف بانتهاء شهادة SSL. مهم: غيّر النظام المتأثر (موقع/بوابة/EMR) والموعد والرابط في كل مرة."},
        {"en": "SCENARIO: Fake IT helpdesk remote access — impersonate IT helpdesk claiming a critical server issue requires remote access credentials immediately.", "ar": "السيناريو: مكتب مساعدة مزيف يطلب بيانات الوصول عن بُعد لحل مشكلة خادم حرجة."},
        {"en": "SCENARIO: CIO impersonation — impersonate the Chief Information Officer urgently requesting server admin credentials or asking to disable security settings.", "ar": "السيناريو: انتحال هوية مدير تقنية المعلومات يطلب بيانات الخادم أو تعطيل إعدادات الأمان."},
        {"en": "SCENARIO: Fake software license renewal — claim a critical hospital software license is expiring in 24 hours and requires immediate renewal via a suspicious portal.", "ar": "السيناريو: تجديد مزيف لترخيص برنامج حيوي ينتهي خلال 24 ساعة."},
        {"en": "SCENARIO: Fake firewall policy update — send a malicious Word document claiming to contain a new mandatory firewall security policy requiring macro enablement.", "ar": "السيناريو: سياسة جدار ناري مزيفة — مستند Word يطلب تفعيل الماكرو."},
        # EN: 4 extra IT scenarios added for more session-to-session variety.
        # AR: 4 سيناريوهات تقنية إضافية لزيادة التنوع بين الجلسات.
        {"en": "SCENARIO: Fake cloud backup credential alert — claim the hospital's cloud backup system detected a failed login and requires immediate credential re-verification via a suspicious link. NO attachment.", "ar": "السيناريو: تنبيه مزيف لنظام النسخ الاحتياطي السحابي — ادّعِ أنه رُصد فشل تسجيل دخول ويتطلب إعادة تحقق فورية عبر رابط مشبوه."},
        {"en": "SCENARIO: Fake Active Directory password expiry — claim the employee's Active Directory/network password expires today and must be reset via a suspicious portal link immediately. NO attachment.", "ar": "السيناريو: انتهاء كلمة مرور Active Directory مزيف — ادّعِ أن كلمة مرور الشبكة تنتهي اليوم ويجب إعادة ضبطها عبر بوابة مشبوهة فورًا."},
        {"en": "SCENARIO: Malicious database backup script attachment — send a fake urgent database backup verification request as an Excel attachment requiring the recipient to enable macros to run a 'repair script'.", "ar": "السيناريو: مرفق Excel خبيث لنص استعادة قاعدة بيانات — أرسل طلب تحقق نسخة احتياطية عاجل كمرفق Excel يطلب تفعيل الماكرو لتشغيل 'سكربت إصلاح'."},
        {"en": "SCENARIO: Fake EMR server maintenance window — impersonate the systems team claiming an emergency EMR server maintenance requires staff to re-authenticate via a suspicious link before the maintenance window starts. NO attachment.", "ar": "السيناريو: نافذة صيانة خادم EMR مزيفة — انتحل هوية فريق الأنظمة وادّعِ أن صيانة طارئة لخادم EMR تتطلب إعادة مصادقة الموظفين عبر رابط مشبوه قبل بدء الصيانة."},
    ],
    "other": [
        # 0 — ADMIN: يطابق OTHER_JOB_PROFILES[0] (billing coordinator)
        {"en": "ADMINISTRATIVE PHISHING — Billing/Payroll: Generate a fake HR payroll email claiming the employee's salary is ON HOLD until they update their IBAN/bank account details. MUST include: fake non-hospital domain, suspicious link to update bank details, deadline threat (end of month/48 hours). VARY each run: HR manager name, bank/IBAN detail type, deadline, suspicious URL. NEVER generate clinical or IT content.",
         "ar": "تصيد إداري — رواتب: رسالة مزيفة من قسم الموارد البشرية تدّعي أن الراتب موقوف حتى تحديث بيانات الحساب البنكي (الآيبان). يجب تضمين: نطاق مزيف، رابط مشبوه للتحديث، تهديد بالموعد النهائي. غيّر في كل مرة: اسم مدير الموارد البشرية، نوع البيانات البنكية، الموعد، الرابط."},
        # 1 — IT: يطابق OTHER_JOB_PROFILES[1] (network technician)
        {"en": "IT/TECHNICAL PHISHING — VPN/Network: Generate a fake IT security alert claiming the employee's hospital network account has detected suspicious login activity and must be re-verified immediately via a suspicious portal. MUST include: fake IT security domain, suspicious portal link, account lockout threat. VARY each run: alert type (suspicious login/account breach/security incident), suspicious link URL, IT security officer name. NEVER generate clinical or administrative content.",
         "ar": "تصيد تقني — VPN/شبكة: تنبيه أمني مزيف من قسم تقنية المعلومات يدّعي رصد نشاط مشبوه على حساب الموظف في الشبكة ويطلب إعادة التحقق فوراً. يجب تضمين: نطاق تقني مزيف، رابط بوابة مشبوهة، تهديد بتعليق الحساب. غيّر في كل مرة: نوع التنبيه، الرابط، اسم مسؤول الأمن."},
        # 2 — CLINICAL: يطابق OTHER_JOB_PROFILES[2] (pharmacist/lab)
        {"en": "CLINICAL PHISHING — Pharmacy/Lab System: Generate a fake pharmacy dispensing system or lab results portal credential harvest. Claim the employee's access to the pharmacy/lab system will be suspended unless they re-verify login credentials immediately. MUST include: fake pharmacy/lab system domain, suspicious credential update link, system suspension threat. VARY each run: system name (pharmacy dispensing/lab portal/medication system), fake domain, suspicious link. NEVER generate administrative or IT content.",
         "ar": "تصيد سريري — نظام صيدلية/مختبر: سرقة بيانات دخول نظام الصيدلية أو بوابة نتائج المختبر. ادّعِ أن وصول الموظف للنظام سيتوقف ما لم يعيد التحقق من بيانات الدخول فوراً. يجب تضمين: نطاق مزيف لنظام صيدلية/مختبر، رابط مشبوه للتحديث، تهديد بتعليق النظام. غيّر في كل مرة: اسم النظام، النطاق المزيف، الرابط."},
        # 3 — ADMIN: يطابق OTHER_JOB_PROFILES[3] (procurement officer)
        {"en": "ADMINISTRATIVE PHISHING — Procurement/Supplier Invoice: Generate a fake urgent supplier invoice email claiming an overdue medical equipment payment (SAR 75,000–200,000) must be approved immediately via a supplier portal link. MUST include: fake supplier company name, suspicious invoice portal link, overdue payment threat. VARY each run: supplier name, equipment type (surgical/lab/radiology/ICU), invoice amount, PDF filename, suspicious portal URL. NEVER generate clinical or IT content.",
         "ar": "تصيد إداري — مشتريات/فاتورة مورد: فاتورة مورد معدات طبية مزيفة تدّعي وجود مبلغ متأخر (75,000-200,000 ريال) يجب دفعه فوراً عبر بوابة المورد. يجب تضمين: اسم شركة مورد مزيف، رابط بوابة مزيف، تهديد بتعليق العقد. غيّر في كل مرة: اسم المورد، نوع المعدات، المبلغ، اسم ملف PDF، الرابط."},
        # 4 — IT: يطابق OTHER_JOB_PROFILES[4] (sysadmin)
        {"en": "IT/TECHNICAL PHISHING — Server/Firewall/CIO: Generate a fake CIO or IT Director impersonation email urgently requesting the employee to provide admin server credentials or disable a security setting immediately. MUST be pure social engineering (no link needed). VARY each run: executive name/title (CIO/CISO/IT Director), specific system (firewall/server/database/Active Directory), urgency reason. NEVER generate clinical or administrative content.",
         "ar": "تصيد تقني — خادم/جدار ناري/انتحال مدير: انتحال هوية مدير تقنية المعلومات أو المدير التنفيذي للتقنية يطلب تزويده ببيانات دخول الخادم أو تعطيل إعداد أمني فوراً. هندسة اجتماعية بحتة. غيّر في كل مرة: اسم ولقب المدير، النظام المحدد، سبب الاستعجال."},
        # 5 — CLINICAL: يطابق OTHER_JOB_PROFILES[5] (radiologist)
        {"en": "CLINICAL PHISHING — PACS/Radiology System: Generate a fake PACS imaging system or radiology portal credential harvest. Claim urgent patient scan results require the employee to log in via a suspicious link, or that the radiology system requires immediate credential re-verification. MUST include: fake radiology/PACS domain, suspicious link, patient urgency. VARY each run: system name (PACS/radiology portal/imaging system), fake domain, patient case reference, suspicious link. NEVER generate administrative or IT content.",
         "ar": "تصيد سريري — نظام PACS/أشعة: سرقة بيانات دخول نظام التصوير الطبي أو بوابة الأشعة. ادّعِ أن نتائج أشعة مريض عاجلة تتطلب تسجيل الدخول عبر رابط مشبوه، أو أن النظام يحتاج إعادة التحقق الفوري. يجب تضمين: نطاق أشعة مزيف، رابط مشبوه، إلحاح حالة مريض. غيّر في كل مرة: اسم النظام، النطاق، رقم الحالة، الرابط."},
    ],
}

# EN: RECIPIENT_POOLS — fixed (name, email) pairs for the generic roles
# (clinical/admin/it), shared by both the learning-phase prompt
# (build_prompt) and the assessment prompt (build_assess_prompt). We pick
# ONE pair per example slot using a session-randomized order, then tell
# the model explicitly to use that exact name in the greeting and that
# exact email in "to". This is the fix for the "Dear Nurse John" vs a
# completely different "to" address bug found repeatedly during testing.
# AR: قوائم مستلمين ثابتة (اسم + بريد) للأدوار العامة (سريري/إداري/تقني)،
# تستخدمها دالتا التوليد (التعلم والاختبار) معًا. نختار زوج واحد لكل خانة
# مثال بترتيب عشوائي خاص بالجلسة، ثم نطلب من الموديل صراحة يستخدم هذا الاسم
# بالتحية وهذا البريد بحقل "to" — يحل خلل "Dear Nurse John" مع عنوان مختلف
# كليًا اللي تكرر بالاختبار.
RECIPIENT_POOLS = {
    "clinical": [
        {"en": "Dr. Sarah Almutairi",  "ar": "د. سارة المطيري",      "email": "dr.sarah.almutairi@hospital.org"},
        {"en": "Dr. Ahmed Alotaibi",   "ar": "د. أحمد العتيبي",      "email": "dr.ahmed.alotaibi@hospital.org"},
        {"en": "Nurse Fatima Alharbi", "ar": "الممرضة فاطمة الحربي", "email": "n.fatima.alharbi@hospital.org"},
        {"en": "Nurse Khalid Alqahtani","ar": "الممرض خالد القحطاني","email": "n.khalid.alqahtani@hospital.org"},
        {"en": "Dr. Noura Alshamri",   "ar": "د. نورة الشمري",       "email": "dr.noura.alshamri@hospital.org"},
        {"en": "Dr. Faisal Aldosari",  "ar": "د. فيصل الدوسري",      "email": "dr.faisal.aldosari@hospital.org"},
        {"en": "Nurse Maha Alsubaie",  "ar": "الممرضة مها السبيعي", "email": "n.maha.alsubaie@hospital.org"},
        {"en": "Dr. Omar Alharthy",    "ar": "د. عمر الحارثي",       "email": "dr.omar.alharthy@hospital.org"},
        {"en": "Nurse Reem Alzahrani", "ar": "الممرضة ريم الزهراني","email": "n.reem.alzahrani@hospital.org"},
        {"en": "Dr. Yousef Alghamdi",  "ar": "د. يوسف الغامدي",      "email": "dr.yousef.alghamdi@hospital.org"},
        {"en": "Pharmacist Lama Alqahtani", "ar": "الصيدلانية لمى القحطاني", "email": "ph.lama.alqahtani@hospital.org"},
        {"en": "Pharmacist Ziad Alharbi",   "ar": "الصيدلاني زياد الحربي",   "email": "ph.ziad.alharbi@hospital.org"},
        {"en": "Lab Technician Huda Alsalmi","ar": "فنية المختبر هدى السالمي","email": "lab.huda.alsalmi@hospital.org"},
        {"en": "Radiology Technician Nasser Aldosari","ar": "فني الأشعة ناصر الدوسري","email": "rad.nasser.aldosari@hospital.org"},
    ],
    "admin": [
        {"en": "Sultan Alghamdi",  "ar": "سلطان الغامدي",  "email": "m.sultan.alghamdi@hospital.org"},
        {"en": "Reem Alsabiei",    "ar": "ريم السبيعي",     "email": "m.reem.alsabiei@hospital.org"},
        {"en": "Nadia Alsalmi",    "ar": "نادية السالمي",   "email": "t.nadia.alsalmi@hospital.org"},
        {"en": "Bandar Althubaiti","ar": "بندر الثبيتي",    "email": "t.bandar.althubaiti@hospital.org"},
        {"en": "Hessa Alqahtani",  "ar": "حصة القحطاني",    "email": "m.hessa.alqahtani@hospital.org"},
        {"en": "Majed Alharbi",    "ar": "ماجد الحربي",     "email": "m.majed.alharbi@hospital.org"},
        {"en": "Lama Alshehri",    "ar": "لمى الشهري",      "email": "t.lama.alshehri@hospital.org"},
        {"en": "Turki Aldosari",   "ar": "تركي الدوسري",    "email": "m.turki.aldosari@hospital.org"},
        {"en": "Amal Alzahrani",   "ar": "أمل الزهراني",    "email": "t.amal.alzahrani@hospital.org"},
        {"en": "Faisal Alotaibi",  "ar": "فيصل العتيبي",    "email": "m.faisal.alotaibi@hospital.org"},
    ],
    "it": [
        {"en": "Bandar Althubaiti","ar": "بندر الثبيتي",    "email": "t.bandar.althubaiti@hospital.org"},
        {"en": "Nadia Alsalmi",    "ar": "نادية السالمي",   "email": "t.nadia.alsalmi@hospital.org"},
        {"en": "Mohammed Alshahri","ar": "محمد الشهري",     "email": "t.mohammed.alshahri@hospital.org"},
        {"en": "Rania Almalki",    "ar": "رانية المالكي",   "email": "t.rania.almalki@hospital.org"},
        {"en": "Khalid Alharbi",   "ar": "خالد الحربي",     "email": "t.khalid.alharbi@hospital.org"},
        {"en": "Sara Alqahtani",   "ar": "سارة القحطاني",   "email": "t.sara.alqahtani@hospital.org"},
        {"en": "Yazeed Aldosari",  "ar": "يزيد الدوسري",    "email": "t.yazeed.aldosari@hospital.org"},
        {"en": "Hanan Alzahrani",  "ar": "حنان الزهراني",   "email": "t.hanan.alzahrani@hospital.org"},
        {"en": "Abdullah Alotaibi","ar": "عبدالله العتيبي", "email": "t.abdullah.alotaibi@hospital.org"},
        {"en": "Lina Alsubaie",    "ar": "لينا السبيعي",    "email": "t.lina.alsubaie@hospital.org"},
    ],
}

# =============================================================
# EN: OPEN-ENDED SCENARIO GENERATION (replaces the fixed scenario
# pool for clinical/admin/it — "other" keeps its existing
# profile-paired scenarios since changing those risks breaking the
# 1:1 link with OTHER_JOB_PROFILES).
# ---------------------------------------------------------------
# Instead of picking one of N pre-written scenarios, we now give
# the model a broad CATEGORY (e.g. "VPN / remote network access")
# and explicitly tell it to INVENT a brand-new, specific, realistic
# scenario within that category — different from anything already
# generated this session. This makes the variety effectively
# unlimited instead of capped at a fixed list size, while the
# category + role context still keeps it realistic and on-topic.
#
# AR: توليد سيناريوهات مفتوح (بدل القائمة الثابتة لسريري/إداري/تقني
# — دور "Other" يبقى على سيناريوهاته المرتبطة بالبروفايلات لأن
# تغييرها يهدد الربط 1:1 مع OTHER_JOB_PROFILES).
# ---------------------------------------------------------------
# بدل اختيار سيناريو جاهز من قائمة محدودة، الآن نعطي الموديل فئة
# عامة (مثل "VPN / وصول شبكة عن بُعد") ونطلب منه صراحة يخترع
# سيناريو جديد ومحدد وواقعي ضمن هذي الفئة — مختلف عن أي شي تولّد
# هذي الجلسة. هذا يخلي التنوع فعليًا لا محدود بدل ما يكون مسقوف
# بحجم قائمة ثابتة، مع بقاء الواقعية بفضل الفئة وسياق الدور.
# =============================================================
OPEN_CATEGORIES = {
    "clinical": [
        {"en": "EMR / patient-records system access", "ar": "نظام السجلات الطبية / الوصول لبيانات المرضى"},
        {"en": "lab or diagnostic results", "ar": "نتائج مختبر أو فحوصات تشخيصية"},
        {"en": "MOH clinical protocol or directive", "ar": "بروتوكول أو توجيه طبي من وزارة الصحة"},
        {"en": "medical staff impersonation (director / chief of staff / consultant)", "ar": "انتحال هوية كادر طبي (مدير طبي/رئيس أطباء/استشاري)"},
        {"en": "clinical scheduling / duty roster / shift system", "ar": "جدولة سريرية / مناوبات / نظام الشِفتات"},
        {"en": "specialized clinical system (pharmacy / blood bank / PACS / telemedicine)", "ar": "نظام سريري متخصص (صيدلية/بنك دم/PACS/طب عن بُعد)"},
    ],
    "admin": [
        {"en": "supplier / vendor invoice or procurement", "ar": "فاتورة مورد أو مشتريات"},
        {"en": "health insurance / employee benefits", "ar": "تأمين صحي / مزايا موظفين"},
        {"en": "payroll / HR / bank account details", "ar": "رواتب / موارد بشرية / بيانات حساب بنكي"},
        {"en": "executive (CEO / director) impersonation", "ar": "انتحال هوية مدير تنفيذي"},
        {"en": "hospital management or records system access", "ar": "نظام إدارة المستشفى أو السجلات الإدارية"},
        {"en": "leave balance / administrative portal", "ar": "رصيد إجازات / بوابة إدارية"},
    ],
    "it": [
        {"en": "VPN / remote network access", "ar": "VPN / وصول شبكة عن بُعد"},
        {"en": "SSL certificate / domain / website", "ar": "شهادة SSL / نطاق / موقع"},
        {"en": "IT helpdesk / remote support request", "ar": "مكتب مساعدة تقني / طلب دعم عن بُعد"},
        {"en": "executive (CIO / CISO) impersonation", "ar": "انتحال هوية مدير تقنية المعلومات"},
        {"en": "software license / subscription renewal", "ar": "ترخيص برنامج / تجديد اشتراك"},
        {"en": "server / backup / database / Active Directory", "ar": "خادم / نسخ احتياطي / قاعدة بيانات / Active Directory"},
    ],
}

_USED_TOPICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "used_topics.json")
_USED_DOMAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "used_domains.json")

def _load_used_store(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def _save_used_store(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        # Read-only filesystem on some hosts — fall back silently to
        # session-only memory for this run.
        pass

def get_avoid_list_text(role_type, key_suffix, is_ar):
    """
    EN: Returns a formatted "do not repeat these" instruction line built
    from topics already generated for this role/phase — combining the
    current session's memory AND a persistent on-disk store, so variety is
    maintained across page refreshes, logouts, and new sessions too (not
    just within one browser tab).
    AR: يرجع سطر تعليمات "لا تكرر هذي" مبني على المواضيع اللي تولّدت
    لهذا الدور/المرحلة — يجمع بين ذاكرة الجلسة الحالية وملف دائم على القرص،
    حتى يستمر التنوع بين الجلسات وتسجيلات الخروج، وليس فقط بنفس المتصفح.
    """
    key = f"used_topics_{role_type}_{key_suffix}"
    session_used = st.session_state.get(key, [])
    persisted = _load_used_store(_USED_TOPICS_PATH).get(key, [])
    used = list(dict.fromkeys(persisted + session_used))[-25:]
    if not used:
        return ""
    items = "؛ ".join(used) if is_ar else "; ".join(used)
    if is_ar:
        return f"\nمواضيع تولّدت سابقًا لهذا الدور — يجب أن يكون موضوعك الجديد مختلفًا عنها تمامًا: {items}\n"
    return f"\nTopics already generated earlier for this role — your new topic MUST be clearly different from all of these: {items}\n"

def remember_used_topic(role_type, key_suffix, topic_text):
    """Appends a short topic descriptor to BOTH the session's "used" list and
    the persistent on-disk store (capped), so future prompts — even from a
    brand-new session — avoid repeating it."""
    if not topic_text:
        return
    key = f"used_topics_{role_type}_{key_suffix}"
    entry = str(topic_text)[:80]
    lst = st.session_state.get(key, [])
    lst.append(entry)
    st.session_state[key] = lst[-12:]
    store = _load_used_store(_USED_TOPICS_PATH)
    persisted = store.get(key, [])
    persisted.append(entry)
    store[key] = persisted[-30:]
    _save_used_store(_USED_TOPICS_PATH, store)

# FIX 1: build_prompt — upgraded to llama-3.3-70b-versatile
# and enhanced difficulty rules with more detail
# =============================================================
def get_used_domains_text(role_type, key_suffix, is_ar):
    """Return domains already generated for this role/phase — combining
    session memory with a persistent on-disk store (see get_avoid_list_text)."""
    key = f"used_domains_{role_type}_{key_suffix}"
    session_used = st.session_state.get(key, [])
    persisted = _load_used_store(_USED_DOMAINS_PATH).get(key, [])
    used = list(dict.fromkeys(persisted + session_used))[-40:]
    if not used:
        return ""
    items = "، ".join(used) if is_ar else ", ".join(used)
    if is_ar:
        return f"\nالنطاقات التي استُخدمت سابقًا لهذا الدور ويُمنع تكرارها أو استخدام نطاق قريب منها: {items}\n"
    return f"\nDomains already used earlier for this role. Do NOT reuse these domains or close variants: {items}\n"

def extract_domains_from_result(result):
    """Extract domains from generated email fields for session-level anti-repeat memory."""
    if not isinstance(result, dict):
        return []
    text = " ".join(str(result.get(k, "")) for k in ["from", "body", "suspicious_link"])
    domains = re.findall(r'(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}', text)
    clean = []
    for d in domains:
        d = re.sub(r'^https?://', '', d).split('/')[0].strip().lower()
        if d and d not in clean and d not in {"hospital.org", "moh.gov.sa"}:
            clean.append(d)
    return clean[:5]

def remember_generated_artifacts(role_type, key_suffix, result):
    """Remember topic + domains so later generations (this session AND future
    sessions, via the on-disk store) avoid repetition."""
    remember_used_topic(role_type, key_suffix, result.get("email_type") or result.get("subject"))
    domains = extract_domains_from_result(result)
    if not domains:
        return
    key = f"used_domains_{role_type}_{key_suffix}"
    current = st.session_state.get(key, [])
    for d in domains:
        if d not in current:
            current.append(d)
    st.session_state[key] = current[-30:]
    store = _load_used_store(_USED_DOMAINS_PATH)
    persisted = store.get(key, [])
    for d in domains:
        if d not in persisted:
            persisted.append(d)
    store[key] = persisted[-60:]
    _save_used_store(_USED_DOMAINS_PATH, store)


def _domain_root(domain):
    domain = (domain or "").lower()
    domain = re.sub(r'^https?://', '', domain).split('/')[0]
    parts = domain.split('.')
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain

_OBVIOUS_ADVANCED_DOMAIN_WORDS = {
    "secure", "update", "verify", "verification", "login", "reset",
    "password", "urgent", "alert", "account", "emr"
}

def _contains_long_all_caps(s):
    return bool(re.search(r'\b[A-Z][A-Z\s]{18,}\b', s or ""))

def _has_generic_greeting(body):
    b = (body or "").lower()
    generic = [
        "dear staff", "dear team", "dear employee", "dear user",
        "عزيزي الموظف", "عزيزتي الموظفة", "عزيزي المستخدم", "عزيزي الفريق",
        "السادة الموظفين", "الموظف العزيز"
    ]
    return any(g in b for g in generic)

def _domain_has_obvious_advanced_word(domain):
    d = (domain or "").lower()
    return any(w in d for w in _OBVIOUS_ADVANCED_DOMAIN_WORDS)



# =============================================================
# STRICT ROLE + DIFFICULTY GUARDRAILS (patched)
# -------------------------------------------------------------
# These checks reject AI outputs that drift away from the selected
# job role (clinical/admin/IT) or violate the documented difficulty
# framework before the email is shown to the trainee.
# =============================================================
_ROLE_KEYWORDS = {
    "clinical": re.compile(r"\b(patient|clinical|emr|ehr|doctor|nurse|pharmac|medication|lab|radiology|pacs|icu|er|ward|handover|vitals|diagnostic|prescription|blood bank)\b|مريض|سريري|طبيب|ممرض|صيدل|دواء|مختبر|أشعة|مناوبة|قسم|بنك الدم|سجل طبي", re.I),
    "admin": re.compile(r"\b(invoice|procurement|vendor|supplier|payroll|insurance|billing|appointment|contract|leave|hr|administrative|records office)\b|فاتورة|مورد|مشتريات|رواتب|تأمين|فوترة|مواعيد|عقد|إجازات|إداري", re.I),
    "it": re.compile(r"\b(vpn|server|network|firewall|ssl|certificate|helpdesk|active directory|backup|database|endpoint|software|license|mfa|otp|cyber|it support)\b|شبكة|خادم|جدار ناري|شهادة|دعم تقني|نسخ احتياطي|قاعدة بيانات|ترخيص|تقنية|أمن سيبراني", re.I),
}
_ROLE_FORBIDDEN = {
    "clinical": re.compile(r"\b(payroll|invoice|vendor|procurement|leave balance|hr portal|vpn|ssl certificate|server|helpdesk|software license|records management team|document collaboration team|security team)\b|رواتب|فاتورة|مورد|مشتريات|إجازات|بوابة الموارد|دعم تقني|شبكة|خادم|فريق الأمن|فريق إدارة السجلات", re.I),
    "admin": re.compile(r"\b(emr|lab results|clinical handover|patient vitals|medication order|radiology image|vpn|ssl certificate|server|firewall)\b|تسليم سريري|نتائج مختبر|علامات حيوية|أمر دوائي|أشعة|شبكة|خادم", re.I),
    "it": re.compile(r"\b(lab results|clinical handover|patient vitals|payroll bank|supplier invoice|appointment booking)\b|نتائج مختبر|تسليم سريري|علامات حيوية|فاتورة مورد|رواتب|حجز مواعيد", re.I),
}

def _current_role_type_for_guardrail():
    try:
        role = st.session_state.get("role", "Clinical")
        return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]
    except Exception:
        return "clinical"

def _role_alignment_issues(result):
    role_type = _current_role_type_for_guardrail()
    if role_type == "other":
        return []
    text = " ".join(str(result.get(k, "")) for k in ["email_type", "from", "subject", "body", "attachment", "suspicious_text", "suspicious_link"])
    issues = []
    if not _ROLE_KEYWORDS[role_type].search(text):
        issues.append(f"email content is not clearly aligned with the selected {role_type} role")
    if _ROLE_FORBIDDEN[role_type].search(text):
        issues.append(f"email drifts into a different role instead of the selected {role_type} role")
    return issues

def _difficulty_structure_issues(result, difficulty, is_phishing=True):
    if not is_phishing or not isinstance(result, dict):
        return []
    body = str(result.get("body", ""))
    combined = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    issues = []
    has_qr = bool(re.search(r"\[\s*QR", body, re.I))
    has_attachment = bool(str(result.get("attachment", "")).strip())
    link = str(result.get("suspicious_link", "")).strip()
    if difficulty == "easy":
        if has_qr: issues.append("Beginner/Easy must not contain QR code")
        if has_attachment: issues.append("Beginner/Easy must not contain attachment")
        if link and link not in body: issues.append("Beginner/Easy must show the fake URL visibly in the body")
        if re.search(r"\[[^\]]+\]\(https?://", body): issues.append("Beginner/Easy must use plain visible URL, not a button")
    elif difficulty == "medium":
        if has_qr: issues.append("Intermediate must not contain QR code")
        if re.search(r"act now|today or|immediately or|account closes today|تصرف الآن|اليوم أو|فورًا وإلا", combined, re.I):
            issues.append("Intermediate urgency is too aggressive")
    elif difficulty == "hard":
        if not has_qr: issues.append("Advanced/Hard must include a QR code marker [QR: ...]")
        if re.search(r"https?://", body): issues.append("Advanced/Hard must not expose raw URL in body")
        if not has_attachment: issues.append("Advanced/Hard must include an official named attachment")
        if re.search(r"password|enter your login|provide your credentials|كلمة المرور|بيانات الدخول", combined, re.I):
            issues.append("Advanced/Hard must avoid direct credential requests")
    return issues

def get_generation_quality_issues(result, difficulty, is_phishing=True):
    """
    Lightweight guardrail used before showing generated content.
    It does not replace the model. It only rejects outputs that clearly violate
    the difficulty contract or repeat domains within the same session.
    """
    if not isinstance(result, dict):
        return ["result is not a JSON object"]

    body = (result.get("body") or "") or ""
    subject = (result.get("subject") or "") or ""
    sender = (result.get("from") or "") or ""
    link = (result.get("suspicious_link") or "") or ""
    combined = " ".join([body, subject, sender, link])
    domains = extract_domains_from_result(result)
    non_official = [d for d in domains if _domain_root(d) not in {"hospital.org", "moh.gov.sa"}]

    issues = []
    issues.extend(_role_alignment_issues(result))
    issues.extend(_difficulty_structure_issues(result, difficulty, is_phishing))
    if is_phishing:
        if not non_official and not result.get("attachment"):
            issues.append("phishing item needs a non-official fake domain/link or a suspicious attachment")
        if difficulty == "easy":
            if not _has_generic_greeting(body):
                issues.append("Beginner must use a generic greeting")
            if not re.search(r'password|credential|login|verify|account|كلمة مرور|بيانات الدخول|تحقق|حساب', combined, re.I):
                issues.append("Beginner needs an obvious sensitive request")
            if not re.search(r'urgent|immediately|today|suspended|terminated|locked|عاجل|فورًا|اليوم|تعليق|إيقاف', combined, re.I):
                issues.append("Beginner needs obvious urgency/threat")
        elif difficulty == "medium":
            if _contains_long_all_caps(combined):
                issues.append("Intermediate must not use aggressive all-caps")
            if re.search(r'permanent termination|within 1 hour|act now|account closed|إنهاء دائم|خلال ساعة|تصرف الآن', combined, re.I):
                issues.append("Intermediate threat is too aggressive")
        elif difficulty == "hard":
            if _has_generic_greeting(body):
                issues.append("Advanced must use a personalized greeting, not generic")
            if _contains_long_all_caps(combined):
                issues.append("Advanced must not contain all-caps urgency")
            if re.search(r'act now|failure to comply|account will be closed|enter your password|full credentials|تصرف الآن|سيتم إغلاق|أدخل كلمة المرور|بيانات الدخول كاملة', combined, re.I):
                issues.append("Advanced contains obvious beginner-style threat or direct password request")
            if any(_domain_has_obvious_advanced_word(d) for d in non_official):
                issues.append("Advanced fake domain is too obvious; avoid secure/update/verify/login/reset/password/urgent/emr")
    else:
        # Legitimate assessment items must stay genuinely safe.
        bad = [d for d in domains if _domain_root(d) not in {"hospital.org", "moh.gov.sa"}]
        if bad:
            issues.append("Legitimate item must not contain external or fake domains")
        if re.search(r'password|credential|verify your account|enter your login|كلمة مرور|بيانات الدخول|تحقق من حسابك', combined, re.I):
            issues.append("Legitimate item must not ask for credentials or account verification")
        if re.search(r'suspended|terminated|locked|account closed|تعليق|إيقاف|إغلاق الحساب', combined, re.I):
            issues.append("Legitimate item must not threaten account suspension")

    return issues

def build_retry_guidance(issues, is_ar):
    if not issues:
        return ""
    joined = "؛ ".join(issues) if is_ar else "; ".join(issues)
    if is_ar:
        return f"""

تم رفض المحاولة السابقة لأنها خالفت قواعد الجودة التالية:
{joined}

أعد التوليد الآن من الصفر بفكرة ونطاق ومرسل مختلفين تمامًا، والتزم بمستوى الصعوبة حرفيًا.
"""
    return f"""

The previous attempt was rejected because it violated these quality rules:
{joined}

Regenerate from scratch with a completely different idea, sender, and domain. Follow the difficulty level literally.
"""

def get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=False):
    """
    4-Axis Difficulty Framework — QR is EXCLUSIVE to hard/Advanced only.
    Axis 1: Sender Identity | Axis 2: Content & Style
    Axis 3: Technical Elements | Axis 4: Role & Healthcare Context
    """
    if is_ar:
        if is_phishing:
            rules = {
                "easy": """
مستوى سهل — المحاور الأربعة الإلزامية:

المحور الأول — هوية المرسل:
- التحية: عامة فقط ("عزيزي الموظف" أو "عزيزي الزميل") — ممنوع منعاً باتاً استخدام أي اسم شخصي.
- نطاق البريد: واضح التزوير تماماً وبعيد عن الرسمي (مثل hospital-alert.com أو medupdate.net).
- المرسل: جهة عامة أو اسم قسم مبهم وغير دقيق.

المحور الثاني — المحتوى والأسلوب:
- الأخطاء: ضع بالضبط خطأين إملائيين واضحين في جسم الرسالة — هذا إلزامي.
- الإلحاح: تهديد مباشر وصريح ("الآن فوراً" أو "سيُغلق حسابك اليوم").
- الطلب: طلب واضح ومباشر لكلمة المرور أو بيانات الدخول.
- الطول: قصير لكن واضح (5-7 أسطر مقروءة إجمالاً).

المحور الثالث — العناصر التقنية:
- الرابط: رابط خام مكشوف كنص عادي في الرسالة — لا تستخدم زراً أو markdown.
- QR: محظور تماماً في هذا المستوى — لا تضع أي رمز QR.
- المرفق: محظور في هذا المستوى.
- الزر: محظور — الرابط فقط كنص.

المحور الرابع — الارتباط الوظيفي والسياق الصحي:
- الارتباط بالوظيفة: عام لأي موظف بغض النظر عن دوره.
- السياق الصحي: ذكر كلمة "مستشفى" أو "صحة" فقط دون تفاصيل.
- المصداقية: منخفضة جداً — سهل الكشف.
""",
                "medium": """
مستوى متوسط — المحاور الأربعة الإلزامية:

المحور الأول — هوية المرسل:
- التحية: شبه مخصصة بالاسم الأول أو المسمى الوظيفي (مثل "عزيزي د. أحمد").
- نطاق البريد: مشابه للرسمي مع فرق عند التدقيق (مثل hospital-it.net).
- المرسل: قسم أو شخص يبدو معقولاً لكن غير رسمي تماماً.

المحور الثاني — المحتوى والأسلوب:
- الأخطاء: خطأ واحد فقط وخفيف في جسم الرسالة — هذا إلزامي.
- الإلحاح: مهني ومتوسط، موعد 24-72 ساعة بدون تهديد عدواني.
- الطلب: غير مباشر — تحديث حساب أو النقر على رابط خارجي.
- الطول: متوسط (6-10 أسطر).

المحور الثالث — العناصر التقنية:
- الرابط: مختصر أو مموّه جزئياً.
- QR: محظور تماماً في هذا المستوى — لا تضع أي رمز QR.
- المرفق: مرفق PDF بسيط وعام مسموح به.
- الزر: زر بسيط مسموح به.

المحور الرابع — الارتباط الوظيفي والسياق الصحي:
- الارتباط بالوظيفة: مرتبط بالقسم الوظيفي المحدد (سريري/إداري/تقني).
- السياق الصحي: اسم قسم حقيقي + اسم نظام أو إجراء داخلي.
- المصداقية: متوسطة — يحتاج انتباهاً للكشف.
""",
                "hard": """
مستوى صعب — المحاور الأربعة الإلزامية (الكل إلزامي):

المحور الأول — هوية المرسل:
- التحية: الاسم الكامل + اللقب الوظيفي الدقيق (مثل "عزيزتي د. نورة العتيبي، استشارية الأمراض الداخلية").
- نطاق البريد: شبه رسمي بذكاء (مثل moh-staff.net) — ممنوع: secure, update, verify, login, reset.
- المرسل: شخص واقعي جداً مع توقيع مهني كامل (الاسم + المنصب + القسم + رقم تحويلة حقيقي).

المحور الثاني — المحتوى والأسلوب:
- الأخطاء: صفر أخطاء — لغة احترافية كاملة.
- الإلحاح: مهذب وخفي فقط ("إجراء روتيني") — ممنوع تماماً أي تهديد.
- الطلب: لا تطلب كلمة المرور مباشرة — الخطر عبر إجراء يبدو طبيعياً.
- الطول: طويل ومفصّل بصيغة احترافية.

المحور الثالث — العناصر التقنية (كل ما يلي إلزامي):
- الرابط: مخفي خلف زر رسمي فقط — ممنوع ظهور الرابط الخام في النص.
- QR: إلزامي ومطلوب دائماً — يجب أن يظهر رمز QR. اكتب [QR: نص قصير وصفي] في موضعه.
- المرفق: مستند رسمي مسمّى بتفاصيل واقعية (مثل بروتوكول_الامتثال_2024.pdf).
- الزر: زر رسمي احترافي بتسمية وصفية.

المحور الرابع — الارتباط الوظيفي والسياق الصحي:
- الارتباط بالوظيفة: مرتبط بمهمة يومية محددة جداً للدور المختار.
- السياق الصحي: نظام EMR أو بروتوكول سريري أو رقم تذكرة داخلية.
- المصداقية: عالية جداً — يصعب كشفه.
- يجب ذكر اسم زميل أو قسم داخلي محدد.
""",
            }
        else:
            rules = {
                "easy": "رسالة شرعية سهلة: نطاق رسمي فقط (hospital.org أو moh.gov.sa)، لا أخطاء، لا تهديد، تحية واضحة، لا روابط خارجية.",
                "medium": "رسالة شرعية متوسطة: نطاق رسمي فقط، لا أخطاء، موعد طبيعي، تحية شبه مخصصة، لا طلب بيانات حساسة.",
                "hard": "رسالة شرعية متقدمة: نطاق رسمي فقط، لا أخطاء، قد تكون عاجلة مهنياً بدون تهديد، تحية مخصصة كاملة، تفاصيل دقيقة ومهنية.",
            }
    else:
        if is_phishing:
            rules = {
                "easy": """
EASY level — 4-Axis mandatory framework:

AXIS 1 — Sender Identity:
- Greeting: GENERIC ONLY ("Dear Employee" or "Dear Staff") — using any personal name is FORBIDDEN.
- Domain: obviously fake and far from official (e.g. hospital-alert.com, medupdate.net).
- Sender: vague department or suspicious generic sender name.

AXIS 2 — Content & Style:
- Errors: include EXACTLY TWO obvious spelling/grammar mistakes in the body — this is MANDATORY.
- Urgency: direct explicit threat ("Act NOW", "within hours", "or your account closes TODAY").
- Request: obvious direct password/credential/account-update request.
- Length: short but meaningful (5-7 readable lines total).

AXIS 3 — Technical Elements:
- Link: raw visible URL as plain text in body — do NOT use a button or markdown link format.
- QR Code: STRICTLY FORBIDDEN — do not include any QR code at this level.
- Attachment: FORBIDDEN at this level.
- Button: FORBIDDEN — plain text URL only.

AXIS 4 — Role & Healthcare Context:
- Role alignment: generic — suitable for ANY employee regardless of role.
- Healthcare context: mention word "hospital" only — no internal details.
- Believability: very low — easily detected.
""",
                "medium": """
INTERMEDIATE level — 4-Axis mandatory framework:

AXIS 1 — Sender Identity:
- Greeting: semi-personalized using first name or job title (e.g. "Dear Dr. Ahmed").
- Domain: plausible but imperfect — detectable difference (e.g. hospital-it.net, hr-moh.com).
- Sender: plausible department or person but not perfectly official.

AXIS 2 — Content & Style:
- Errors: EXACTLY ONE subtle spelling/grammar mistake in the body — this is MANDATORY.
- Urgency: moderate professional deadline 24-72 hours — no aggressive threats.
- Request: indirect — account update or click external link.
- Length: medium (6-10 lines).

AXIS 3 — Technical Elements:
- Link: shortened or partially hidden URL.
- QR Code: STRICTLY FORBIDDEN — do not include any QR code at this level.
- Attachment: simple generic PDF is allowed.
- Button: simple button is allowed.

AXIS 4 — Role & Healthcare Context:
- Role alignment: tied to the specific job department (Clinical/Administrative/IT).
- Healthcare context: real department name + system name or internal procedure.
- Believability: moderate — requires attention to detect.
""",
                "hard": """
ADVANCED level — 4-Axis mandatory framework (ALL elements are mandatory):

AXIS 1 — Sender Identity:
- Greeting: FULL NAME + precise job title (e.g. "Dear Dr. Noura Al-Otaibi, Internal Medicine Consultant").
- Domain: near-official but not matching (e.g. moh-staff.net). FORBIDDEN words: secure, update, verify, login, reset.
- Sender: realistic person with COMPLETE professional signature (name + title + dept + real extension number).

AXIS 2 — Content & Style:
- Errors: ZERO errors — completely flawless professional language.
- Urgency: polite and subtle ONLY ("routine procedure") — NO threats of any kind.
- Request: do NOT ask for password — risky action looks like normal workflow.
- Length: long and detailed in professional format.

AXIS 3 — Technical Elements (ALL mandatory):
- Link: hidden behind button ONLY — raw URL MUST NOT appear in body text.
- QR Code: MANDATORY AND REQUIRED — a QR code MUST appear in every Advanced email.
  Write [QR: short descriptive label] at the appropriate position in the body.
- Attachment: officially named document with realistic details (e.g. Compliance_Protocol_2024.pdf).
- Button: professionally styled with a descriptive label (NOT "Open Link" or "Click Here").

AXIS 4 — Role & Healthcare Context:
- Role alignment: tied to a specific daily task of the selected role.
- Healthcare context: EMR system / clinical protocol / internal ticket number.
- Believability: very high — difficult to detect.
- Must mention a specific colleague name or internal department.
""",
            }
        else:
            rules = {
                "easy": "Legitimate Easy: official hospital.org or moh.gov.sa domain only, no errors, no urgency, clear greeting, no sensitive data request, no external links.",
                "medium": "Legitimate Intermediate: official domain only, no errors, normal deadline, semi-personal greeting, no credentials request, realistic workflow detail.",
                "hard": "Legitimate Advanced: official domain only, no errors, may be professionally urgent but not threatening, personalized greeting, realistic sender, detailed healthcare context.",
            }
    return rules.get(difficulty, rules.get("medium"))



def get_role_unbounded_context(role_type, is_ar=False):
    """Role context only; not a scenario template list. The model must invent the actual scenario."""
    if is_ar:
        return {
            "clinical": "الدور سريري داخل مستشفى سعودي: أطباء، تمريض، صيدلة، مختبر، أشعة، عيادات، طوارئ، عناية مركزة، سجلات طبية، أنظمة مرضى، بروتوكولات وزارة الصحة.",
            "admin": "الدور إداري داخل مستشفى سعودي: استقبال، سكرتارية طبية، ملفات مرضى، تأمين، فوترة، مشتريات، موارد بشرية، اعتماد، جدولة، عقود موردين.",
            "it": "الدور تقني داخل مستشفى سعودي: شبكات، VPN، خوادم، EMR، نسخ احتياطي، Active Directory، شهادات، جدار ناري، تراخيص، مكتب مساعدة، أمن سيبراني.",
            "other": "الدور موظف عام في مستشفى سعودي. اختر بحرية قسمًا منطقيًا جديدًا في كل مرة: سريري أو إداري أو تقني أو تشغيلي.",
        }.get(role_type, "الدور موظف في مستشفى سعودي.")
    return {
        "clinical": "Clinical role in a Saudi hospital: doctors, nurses, pharmacy, lab, radiology, clinics, ER, ICU, EMR, patient systems, MOH clinical protocols.",
        "admin": "Administrative role in a Saudi hospital: reception, medical secretary, patient records, insurance, billing, procurement, HR, accreditation, scheduling, vendor contracts.",
        "it": "IT/Informatics role in a Saudi hospital: network, VPN, servers, EMR, backups, Active Directory, certificates, firewall, licenses, helpdesk, cybersecurity.",
        "other": "General Saudi hospital employee. Freely choose a fresh logical department each time: clinical, administrative, technical, operational, or support.",
    }.get(role_type, "Saudi hospital employee.")


# =============================================================
# HEALTHCARE SCENARIO LIBRARY (300 Scenario Cards)
# -------------------------------------------------------------
# These are NOT email templates. They are compact content directions.
# The AI still writes the full email, analysis and assessment text via API,
# but it no longer invents the core idea from nothing. This prevents repeated
# "password/account" emails and keeps the content healthcare-relevant.
# =============================================================
def _make_cards(role_type, groups):
    cards = []
    counter = 1
    prefix = {"clinical": "CL", "admin": "AD", "it": "IT"}[role_type]
    for sub_role, sender, system, topics in groups:
        for topic in topics:
            cards.append({
                "id": f"{prefix}_{counter:03d}",
                "role_type": role_type,
                "sub_role": sub_role,
                "sender": sender,
                "system": system,
                "scenario": topic,
                "action": random.choice([
                    "review the notice", "confirm the update", "open the referenced workflow",
                    "acknowledge the task", "check the attached or linked information"
                ]),
                "attack_options": ["credential_harvesting", "fake_portal", "malicious_pdf", "button_link", "qr_phishing"],
            })
            counter += 1
    return cards[:100]

_CLINICAL_GROUPS = [
    ("Doctor", "Medical Affairs Office", "EMR", [
        "OPD clinic schedule revision", "surgery list confirmation", "ICU rounds handover", "consultant on-call roster",
        "clinical privileges renewal", "resident evaluation review", "Morbidity and Mortality meeting note", "multidisciplinary team meeting invite",
        "telemedicine appointment queue", "patient transfer approval", "operative note completion", "discharge summary backlog",
        "antibiotic stewardship review", "clinical trial screening list", "CME credit confirmation", "BLS recertification reminder",
        "patient safety event review", "medication interaction alert", "urgent referral acceptance", "outpatient referral triage"
    ]),
    ("Nurse", "Nursing Affairs", "Nursing Portal", [
        "shift handover checklist", "ward staffing adjustment", "medication round checklist", "patient fall incident report",
        "pressure injury audit", "infection-control competency", "isolation room assignment", "bedside handover update",
        "nursing documentation correction", "CPR renewal schedule", "float pool assignment", "vaccination campaign roster",
        "uniform policy acknowledgement", "smart infusion pump update", "patient wristband verification", "ICU bed assignment",
        "charge nurse monthly report", "controlled medication witness log", "new admission task list", "nurse annual appraisal"
    ]),
    ("Pharmacist", "Pharmacy Safety Unit", "Pharmacy System", [
        "medication recall notice", "controlled drug inventory count", "Pyxis cabinet synchronization", "LASA medication alert",
        "vaccine cold-chain report", "formulary update approval", "antibiotic restriction request", "expired medication disposal",
        "IV preparation worksheet", "chemotherapy order verification", "ADR report follow-up", "medication reconciliation queue",
        "narcotic discrepancy review", "pharmacy rotation schedule", "drug shortage substitution", "clinical pharmacy note review",
        "high-alert medication policy", "prescription verification backlog", "outpatient refill exception", "ward stock adjustment"
    ]),
    ("Laboratory Specialist", "Laboratory Services", "LIS", [
        "critical value confirmation", "specimen rejection notice", "blood bank inventory check", "analyzer maintenance schedule",
        "microbiology culture report", "hematology QC review", "chemistry calibration update", "phlebotomy roster change",
        "sample recollection request", "lab accreditation checklist", "point-of-care testing update", "crossmatch verification",
        "pathology report correction", "reference lab send-out", "lab result release delay", "reagent lot verification",
        "blood component traceability", "STAT sample queue", "laboratory incident form", "LIS downtime procedure"
    ]),
    ("Radiology Technician", "Radiology Administration", "PACS", [
        "PACS image review", "CT protocol update", "MRI safety checklist", "ultrasound appointment queue",
        "radiology report addendum", "contrast media policy", "portable X-ray schedule", "interventional radiology list",
        "radiation badge reading", "DICOM viewer update", "critical imaging result alert", "radiology equipment maintenance",
        "patient preparation instruction", "after-hours imaging roster", "mammography audit", "fluoroscopy dose report",
        "radiology peer review", "PACS storage notice", "contrast allergy documentation", "ER imaging workflow"
    ]),
]

_ADMIN_GROUPS = [
    ("HR Officer", "Human Resources", "HR Portal", [
        "annual leave balance review", "payroll correction form", "attendance exception request", "staff evaluation cycle",
        "mandatory training enrollment", "new employee onboarding", "contract renewal acknowledgement", "housing allowance update",
        "transportation allowance confirmation", "staff survey invitation", "disciplinary policy acknowledgement", "promotion eligibility review",
        "credential file completion", "overtime approval", "shift allowance verification", "employee data update",
        "vacation carryover request", "performance improvement plan", "ID badge renewal", "employee benefits window"
    ]),
    ("Medical Secretary", "Medical Administration", "Scheduling System", [
        "clinic appointment reschedule", "consultant meeting agenda", "patient file indexing", "medical report release",
        "doctor office coverage", "referral letter queue", "department minutes approval", "patient complaint follow-up",
        "committee attendance sheet", "clinic template adjustment", "physician roster update", "outpatient slot release",
        "VIP patient coordination", "call center escalation", "medical certificate request", "appointment reminder batch",
        "doctor signature pending", "department circular", "clinic cancellation notice", "patient correspondence review"
    ]),
    ("Insurance Coordinator", "Insurance Office", "Claims Portal", [
        "insurance pre-authorization", "claim rejection review", "coverage update request", "reimbursement file audit",
        "payer portal migration", "medical necessity form", "policy number correction", "eligibility verification batch",
        "approval extension request", "denied claim appeal", "TPA document request", "co-payment exception",
        "insurance contract update", "patient guarantee letter", "case management review", "billing code correction",
        "authorization expiry notice", "payer meeting invite", "claim attachment upload", "utilization review list"
    ]),
    ("Procurement Officer", "Procurement Department", "Procurement Portal", [
        "vendor invoice approval", "medical equipment quotation", "supplier contract renewal", "purchase order confirmation",
        "tender committee schedule", "delivery note mismatch", "warehouse stock request", "vendor registration update",
        "maintenance contract review", "urgent device replacement", "consumables shortage notice", "capital equipment approval",
        "service level agreement", "supplier bank details", "contract variation order", "purchase requisition queue",
        "vendor compliance declaration", "price comparison sheet", "procurement policy update", "delivery appointment booking"
    ]),
    ("Finance Officer", "Finance Department", "Finance System", [
        "budget variance review", "expense reimbursement", "petty cash reconciliation", "month-end closing task",
        "audit evidence request", "invoice payment batch", "vendor payment schedule", "department budget transfer",
        "VAT certificate upload", "financial delegation update", "bank guarantee notice", "cashier report review",
        "cost center correction", "asset capitalization form", "revenue report adjustment", "finance committee minutes",
        "payment approval workflow", "fund request tracking", "payroll journal review", "accounts payable aging"
    ]),
]

_IT_GROUPS = [
    ("IT Support Engineer", "IT Helpdesk", "Service Desk", [
        "password expiry notice", "MFA enrollment", "VPN access renewal", "Outlook mailbox quota",
        "Teams meeting policy", "printer queue maintenance", "laptop compliance check", "remote support ticket",
        "software license renewal", "device registration", "email quarantine review", "shared drive access",
        "helpdesk ticket closure", "asset tag verification", "Windows update schedule", "browser certificate prompt",
        "mobile device management", "endpoint encryption check", "staff portal login", "IT satisfaction survey"
    ]),
    ("Network Engineer", "Network Operations", "Network Portal", [
        "firewall policy review", "WiFi controller upgrade", "switch maintenance window", "VPN gateway certificate",
        "network access request", "guest WiFi policy", "WAN failover test", "IP address conflict",
        "data center cabling plan", "NAC re-authentication", "DNS record update", "load balancer change",
        "internet bandwidth report", "site-to-site VPN tunnel", "network monitoring alert", "DHCP scope update",
        "wireless survey schedule", "router firmware notice", "network segmentation task", "VoIP extension migration"
    ]),
    ("Cybersecurity Analyst", "Cybersecurity Office", "Security Portal", [
        "phishing simulation notice", "security awareness quiz", "EDR alert review", "SIEM case assignment",
        "privileged access review", "vulnerability scan report", "patch compliance exception", "incident response drill",
        "USB control policy", "suspicious login alert", "security baseline update", "threat intelligence bulletin",
        "data loss prevention alert", "account lockout trend", "red team exercise", "security questionnaire",
        "certificate trust update", "ransomware readiness checklist", "MFA bypass review", "SOC escalation"
    ]),
    ("Systems Administrator", "Systems Team", "Infrastructure Portal", [
        "Active Directory password policy", "server maintenance window", "backup job failure", "database restore test",
        "virtual machine snapshot", "storage quota warning", "SSL certificate renewal", "domain controller health",
        "file server permission", "cloud backup registration", "application server restart", "database account review",
        "patch Tuesday reboot", "service account expiry", "monitoring agent update", "disaster recovery plan",
        "system log archive", "Windows server license", "intranet portal migration", "scheduled downtime notice"
    ]),
    ("Clinical Informatics Specialist", "Health Informatics", "HIS/EMR", [
        "EMR downtime notice", "PACS integration check", "LIS interface update", "HIS user acceptance test",
        "clinical order set update", "barcode medication administration", "e-prescribing workflow", "patient portal configuration",
        "single sign-on rollout", "clinic template in HIS", "nursing documentation form", "radiology interface queue",
        "lab result mapping", "appointment system sync", "clinical dashboard access", "ICD coding update",
        "HL7 message error", "bed management system", "telehealth platform update", "EMR training session"
    ]),
]

SCENARIO_LIBRARY = {
    "clinical": _make_cards("clinical", _CLINICAL_GROUPS),
    "admin": _make_cards("admin", _ADMIN_GROUPS),
    "it": _make_cards("it", _IT_GROUPS),
}

_ATTACK_BY_DIFFICULTY = {
    # Easy keeps obvious phishing markers, but the action must stay tied to the selected healthcare scenario.
    "easy": ["direct_credential_request", "visible_fake_url", "same_day_access_pressure", "obvious_fake_portal"],
    # Medium is semi-plausible: indirect request, moderate deadline, no QR.
    "medium": ["lookalike_portal", "simple_pdf", "simple_button", "external_review_link"],
    # Hard is polished and workflow-like: QR + official document + hidden button; no raw URL.
    "hard": ["qr_phishing", "official_attachment", "professional_button", "workflow_confirmation"],
}

# Dynamic content-shape engine. These are NOT fixed email templates.
# They are writing constraints that make the API generate different forms of email content each time.
EMAIL_WRITING_STYLES = {
    "English": [
        "internal notice", "workflow reminder", "department bulletin", "policy update",
        "audit follow-up", "maintenance notice", "training reminder", "incident follow-up",
        "system notification", "committee update", "compliance note", "handover reminder"
    ],
    "Arabic": [
        "إشعار داخلي", "تذكير بسير العمل", "تعميم قسم", "تحديث سياسة",
        "متابعة تدقيق", "إشعار صيانة", "تذكير تدريب", "متابعة حادثة",
        "تنبيه نظام", "تحديث لجنة", "ملاحظة امتثال", "تذكير تسليم"
    ],
}

EMAIL_STRUCTURES = {
    "English": [
        "Greeting → context → impact → requested action → deadline → sign-off",
        "Greeting → brief background → task detail → action line → verification route → sign-off",
        "Greeting → department update → why it matters → next step → closing note → sign-off",
        "Greeting → operational issue → affected workflow → requested confirmation → deadline → sign-off",
        "Greeting → notice summary → staff responsibility → link/attachment/button placement → support note → sign-off"
    ],
    "Arabic": [
        "تحية → سياق → أثر → إجراء مطلوب → مهلة → توقيع",
        "تحية → خلفية مختصرة → تفاصيل المهمة → إجراء → قناة تحقق → توقيع",
        "تحية → تحديث القسم → سبب الأهمية → الخطوة التالية → ملاحظة ختامية → توقيع",
        "تحية → مشكلة تشغيلية → سير العمل المتأثر → تأكيد مطلوب → مهلة → توقيع",
        "تحية → ملخص الإشعار → مسؤولية الموظف → موضع الرابط/المرفق/الزر → ملاحظة دعم → توقيع"
    ],
}

SUBJECT_PATTERNS = {
    "easy": {
        "English": ["Final Warning", "Immediate Password Verification Required", "Action Required Today", "Urgent Access Update", "Account Access Will Stop Today"],
        "Arabic": ["تحذير نهائي", "تحقق فوري من كلمة المرور", "إجراء مطلوب اليوم", "تحديث دخول عاجل", "سيتم إيقاف الوصول اليوم"]
    },
    "medium": {
        "English": ["Pending Review", "Department Update", "Confirmation Required Within 48 Hours", "Workflow Follow-Up", "Review Request"],
        "Arabic": ["مراجعة معلقة", "تحديث قسم", "تأكيد مطلوب خلال 48 ساعة", "متابعة سير عمل", "طلب مراجعة"]
    },
    "hard": {
        "English": ["Routine Procedure", "Internal Workflow Review", "Policy Acknowledgement", "Scheduled Governance Update", "Protocol Documentation Review"],
        "Arabic": ["إجراء روتيني", "مراجعة سير عمل داخلية", "إقرار سياسة", "تحديث حوكمة مجدول", "مراجعة توثيق بروتوكول"]
    },
}

SIGNATURE_PERSONAS = {
    "clinical": ["Clinical Governance", "Patient Safety Office", "Medical Affairs Office", "Nursing Affairs", "Pharmacy Safety Unit", "Laboratory Services", "Radiology Administration", "Infection Control Unit"],
    "admin": ["Human Resources", "Medical Administration", "Insurance Office", "Procurement Department", "Finance Department", "Quality Department", "Operations Office"],
    "it": ["IT Helpdesk", "Health Informatics", "Cybersecurity Office", "Systems Team", "Network Operations", "Service Desk"],
    "other": ["Hospital Operations", "Quality Department", "Training Office", "Staff Services"]
}

def build_content_shape(role_type, difficulty, is_ar=False):
    lang = "Arabic" if is_ar else "English"
    style = random.choice(EMAIL_WRITING_STYLES[lang])
    structure = random.choice(EMAIL_STRUCTURES[lang])
    subject_pattern = random.choice(SUBJECT_PATTERNS.get(difficulty, SUBJECT_PATTERNS["medium"])[lang])
    signature_pool = SIGNATURE_PERSONAS.get(role_type, SIGNATURE_PERSONAS["other"])
    signature = random.choice(signature_pool)
    # Make the model vary paragraph rhythm without making Easy too empty.
    if difficulty == "easy":
        length_hint = "2 short paragraphs, 5-7 readable lines total" if not is_ar else "فقرتان قصيرتان، 5-7 أسطر مقروءة إجمالاً"
    elif difficulty == "medium":
        length_hint = "3 paragraphs, 7-10 readable lines total" if not is_ar else "3 فقرات، 7-10 أسطر مقروءة إجمالاً"
    else:
        length_hint = "4-5 polished paragraphs, detailed but not excessive" if not is_ar else "4-5 فقرات مصقولة، مفصلة بدون إطالة مفرطة"
    return {
        "style": style,
        "structure": structure,
        "subject_pattern": subject_pattern,
        "signature": signature,
        "length_hint": length_hint,
        "nonce": random.randint(10000, 99999),
    }


# =============================================================
# SAFE EVENT-SEQUENCE LAYER
# -------------------------------------------------------------
# This layer changes only the story logic (what happened first,
# what changed, and why the recipient is contacted). It does NOT
# control rendering, URLs, indicators, analysis, difficulty rules,
# classification, or the final JSON schema. Those remain governed
# by the existing engine and post-generation guardrails.
# =============================================================
_EVENT_SEQUENCE_LIBRARY = {
    "clinical": [
        ("record_updated", "clinical_review_pending", "acknowledgement_requested"),
        ("result_released", "clarification_added", "follow_up_requested"),
        ("referral_submitted", "document_missing", "case_returned_for_completion"),
        ("schedule_changed", "coverage_conflict_detected", "confirmation_requested"),
        ("safety_notice_issued", "department_response_missing", "escalation_created"),
        ("handover_created", "incoming_team_comment_needed", "receipt_requested"),
        ("medication_item_flagged", "pharmacy_review_pending", "documented_response_requested"),
        ("specimen_processed", "discrepancy_detected", "correction_requested"),
        ("imaging_report_amended", "clinical_note_pending", "review_requested"),
        ("training_assigned", "completion_status_unrecorded", "attendance_confirmation_requested"),
    ],
    "admin": [
        ("request_submitted", "approval_pending", "confirmation_requested"),
        ("invoice_received", "cost_center_mismatch", "correction_requested"),
        ("schedule_published", "staffing_conflict_detected", "response_requested"),
        ("contract_review_started", "document_expiry_detected", "renewal_action_requested"),
        ("claim_processed", "supporting_document_missing", "case_returned_for_completion"),
        ("employee_record_updated", "verification_pending", "acknowledgement_requested"),
        ("purchase_order_created", "supplier_detail_changed", "approval_requested"),
        ("audit_opened", "evidence_gap_found", "document_upload_requested"),
    ],
    "it": [
        ("system_alert_created", "service_impact_detected", "technical_review_requested"),
        ("maintenance_scheduled", "dependency_conflict_found", "confirmation_requested"),
        ("certificate_checked", "renewal_window_opened", "administrator_action_requested"),
        ("backup_job_completed", "integrity_exception_detected", "validation_requested"),
        ("access_change_requested", "approval_pending", "identity_confirmation_requested"),
        ("security_event_logged", "owner_response_missing", "escalation_created"),
        ("software_update_released", "compatibility_issue_found", "deployment_response_requested"),
        ("device_inventory_updated", "ownership_mismatch_detected", "correction_requested"),
    ],
}

_EVENT_SEQUENCE_LABELS_AR = {
    "record_updated": "تم تحديث السجل", "clinical_review_pending": "أصبحت المراجعة السريرية معلقة",
    "acknowledgement_requested": "طُلب تأكيد الاستلام", "result_released": "صدرت النتيجة",
    "clarification_added": "أُضيف توضيح", "follow_up_requested": "طُلبت متابعة",
    "referral_submitted": "أُرسلت الإحالة", "document_missing": "ظهر مستند ناقص",
    "case_returned_for_completion": "أُعيدت الحالة للاستكمال", "schedule_changed": "تغيّر الجدول",
    "coverage_conflict_detected": "ظهر تعارض في التغطية", "confirmation_requested": "طُلب التأكيد",
    "safety_notice_issued": "صدر إشعار سلامة", "department_response_missing": "لم تُسجل استجابة القسم",
    "escalation_created": "تم إنشاء تصعيد", "handover_created": "أُنشئ تسليم",
    "incoming_team_comment_needed": "احتاج الفريق المستلم إلى تعليق", "receipt_requested": "طُلب تأكيد الاستلام",
    "medication_item_flagged": "تم تعليم بند دوائي", "pharmacy_review_pending": "أصبحت مراجعة الصيدلية معلقة",
    "documented_response_requested": "طُلب رد موثق", "specimen_processed": "تمت معالجة العينة",
    "discrepancy_detected": "اكتُشف اختلاف", "correction_requested": "طُلب تصحيح",
    "imaging_report_amended": "عُدّل تقرير الأشعة", "clinical_note_pending": "أصبحت الملاحظة السريرية معلقة",
    "review_requested": "طُلبت مراجعة", "training_assigned": "أُسند تدريب",
    "completion_status_unrecorded": "لم تُسجل حالة الإكمال", "attendance_confirmation_requested": "طُلب تأكيد الحضور",
}

_SEQUENCE_ACTION_COMPATIBILITY = {
    "clinical": re.compile(r"review|confirm|acknowledge|check|open|document|correct|follow|respond|مراجعة|تأكيد|إقرار|فتح|توثيق|تصحيح|متابعة|رد", re.I),
    "admin": re.compile(r"review|approve|confirm|acknowledge|check|upload|correct|renew|مراجعة|اعتماد|تأكيد|إقرار|رفع|تصحيح|تجديد", re.I),
    "it": re.compile(r"review|validate|confirm|approve|check|deploy|renew|respond|مراجعة|تحقق|تأكيد|اعتماد|فحص|نشر|تجديد|رد", re.I),
}


def _sequence_role(role_type):
    if role_type == "other":
        return random.choice(["clinical", "admin", "it"])
    return role_type if role_type in _EVENT_SEQUENCE_LIBRARY else "clinical"


def _sequence_signature(plan):
    return "|".join(plan.get("event_chain", [])) + "|" + str(plan.get("narrative_shape", ""))


def build_safe_event_plan(card, difficulty, phase="learn", is_phishing=True, is_ar=False):
    """Return a validated story plan. It never writes the email itself."""
    role_type = _sequence_role(card.get("role_type", "clinical"))
    pool = _EVENT_SEQUENCE_LIBRARY[role_type]
    memory_key = f"used_event_sequences_{phase}_{role_type}_{difficulty}_{is_phishing}"
    used = set(st.session_state.get(memory_key, []))

    shapes = [
        "chronological: event -> change/problem -> requested next step",
        "status-first: current status -> what caused it -> requested next step",
        "exception-first: detected exception -> operational effect -> requested next step",
        "handover-first: ownership transfer -> missing confirmation -> requested next step",
        "follow-up-first: previous action -> no recorded response -> requested next step",
    ]
    candidates = []
    for chain in pool:
        for shape in shapes:
            plan = {
                "role_type": role_type,
                "event_chain": list(chain),
                "narrative_shape": shape,
                "scenario_id": card.get("id", ""),
                "core_task": card.get("scenario", ""),
                "logical_sender": card.get("sender", ""),
                "internal_system": card.get("system", ""),
                "requested_action": card.get("action", "review the notice"),
                "difficulty": difficulty,
                "classification": "phishing" if is_phishing else "legitimate",
            }
            if _sequence_signature(plan) not in used:
                candidates.append(plan)
    if not candidates:
        st.session_state[memory_key] = []
        candidates = [{
            "role_type": role_type, "event_chain": list(random.choice(pool)),
            "narrative_shape": random.choice(shapes), "scenario_id": card.get("id", ""),
            "core_task": card.get("scenario", ""), "logical_sender": card.get("sender", ""),
            "internal_system": card.get("system", ""), "requested_action": card.get("action", "review the notice"),
            "difficulty": difficulty, "classification": "phishing" if is_phishing else "legitimate",
        }]
    plan = random.choice(candidates)
    issues = validate_safe_event_plan(plan)
    if issues:
        # Guaranteed local fallback: simple, compatible chain. Existing email engine remains untouched.
        plan["event_chain"] = list(pool[0])
        plan["narrative_shape"] = shapes[0]
        plan["fallback_used"] = True
    sig = _sequence_signature(plan)
    history = st.session_state.get(memory_key, [])
    history.append(sig)
    st.session_state[memory_key] = history[-40:]
    return plan


def validate_safe_event_plan(plan):
    issues = []
    role_type = plan.get("role_type")
    chain = plan.get("event_chain")
    if role_type not in _EVENT_SEQUENCE_LIBRARY:
        issues.append("unknown role type")
    if not isinstance(chain, list) or len(chain) != 3 or any(not str(x).strip() for x in chain):
        issues.append("event chain must contain exactly three non-empty events")
    action = str(plan.get("requested_action", ""))
    compat = _SEQUENCE_ACTION_COMPATIBILITY.get(role_type)
    if compat and action and not compat.search(action):
        issues.append("requested action is not compatible with selected role")
    if plan.get("classification") not in {"phishing", "legitimate"}:
        issues.append("invalid classification")
    return issues


def event_plan_to_prompt(plan, is_ar=False):
    """Serialize only story logic into prompt text; never analysis or technical placement rules."""
    if is_ar:
        chain = " ← ".join(_EVENT_SEQUENCE_LABELS_AR.get(x, x.replace("_", " ")) for x in plan["event_chain"])
        return f"""
خطة تسلسل أحداث آمنة — استخدمها لصياغة القصة فقط:
- تسلسل الأحداث الإلزامي: {chain}
- شكل السرد: {plan['narrative_shape']}
- المهمة الصحية الأساسية: {plan['core_task']}
- الجهة المنطقية: {plan['logical_sender']}
- النظام الداخلي: {plan['internal_system']}
- الإجراء النهائي: {plan['requested_action']}

حدود صارمة:
- لا تغيّر التصنيف أو مستوى الصعوبة.
- لا تحدد مكان الرابط أو QR أو المرفق؛ اتبع قواعد المستوى الحالية فقط.
- لا تكتب تحليل المعلم من هذه الخطة. التحليل يجب أن يُشتق من النص النهائي الظاهر.
- يجب أن تكون كل جملة منطقية مع تسلسل الأحداث، من دون خلط وظائف أو أقسام.
"""
    chain = " -> ".join(x.replace("_", " ") for x in plan["event_chain"])
    return f"""
Safe Event-Sequence Plan — use it for story logic only:
- Mandatory event chain: {chain}
- Narrative shape: {plan['narrative_shape']}
- Core healthcare task: {plan['core_task']}
- Logical unit/sender: {plan['logical_sender']}
- Internal system/procedure: {plan['internal_system']}
- Final requested action: {plan['requested_action']}

Hard boundaries:
- Do not change classification or difficulty.
- Do not decide URL, QR, attachment, or indicator placement; obey the existing difficulty rules only.
- Do not copy analysis from this plan. Tutor analysis must be derived from visible final email evidence.
- Every sentence must follow this event chain logically, without mixing roles or departments.
"""

def select_scenario_card(role_type, index, phase="learn"):
    """Pick a scenario card from the 300-card library without repeating the same card order.
    Other is a deliberate mix of clinical/admin/it, as requested.
    """
    if role_type == "other":
        mix_roles = ["clinical", "admin", "it"]
        role_type = mix_roles[index % len(mix_roles)]
    cards = SCENARIO_LIBRARY.get(role_type, SCENARIO_LIBRARY["clinical"])
    order_key = f"scenario_card_order_{phase}_{role_type}"
    order = get_session_random_order(len(cards), order_key)
    return cards[order[index % len(order)]]

def scenario_card_to_prompt(card, difficulty, is_ar=False, phase="learn", is_phishing=True):
    """Build a dynamic scenario instruction.

    The scenario card gives the healthcare idea. The content-shape engine gives
    style/structure/subject/signature variation. The API still writes the final
    email, but it is now constrained by both the 300-card content library and the
    difficulty framework.
    """
    attack_pool = _ATTACK_BY_DIFFICULTY.get(difficulty, _ATTACK_BY_DIFFICULTY["medium"])
    attack = random.choice(attack_pool) if is_phishing else "none — legitimate workplace communication"
    shape = build_content_shape(card.get("role_type", "clinical"), difficulty, is_ar)
    event_plan = build_safe_event_plan(card, difficulty, phase=phase, is_phishing=is_phishing, is_ar=is_ar)
    event_instruction = event_plan_to_prompt(event_plan, is_ar=is_ar)
    if is_ar:
        return f"""
بطاقة السيناريو المعتمدة — يجب الالتزام بها وعدم استبدالها:
- رقم السيناريو: {card['id']}
- النوع الرئيسي: {card['role_type']}
- الدور الداخلي: {card['sub_role']}
- الفكرة/المهمة الأساسية: {card['scenario']}
- الجهة/المرسل المنطقي: {card['sender']}
- النظام أو الإجراء الداخلي: {card['system']}
- الإجراء المطلوب: {card['action']}
- نوع الهجوم المناسب لهذا المستوى: {attack}

محرك تنويع محتوى الإيميل — إلزامي:
- أسلوب الكتابة المطلوب هذه المرة: {shape['style']}
- شكل ترتيب الفقرات هذه المرة: {shape['structure']}
- نمط العنوان المطلوب: {shape['subject_pattern']} + اسم المهمة الصحية أعلاه
- التوقيع المقترح: {shape['signature']} مع اسم شخص ومنصب منطقيين إذا كان المستوى صعباً
- الطول المطلوب: {shape['length_hint']}
- رقم تنويع داخلي: {shape['nonce']}
{event_instruction}
قواعد منع التكرار:
- لا تبدأ الرسالة دائمًا بنفس الجملة.
- لا تستخدم نفس نهاية الرسالة في كل مرة.
- لا تجعل كل الرسائل عن "الحساب" فقط؛ اربط الطلب بالمهمة الصحية المحددة.
- لا تجعل التحليل عامًا. كل مؤشر يجب أن يذكر عبارة/رابط/مرسل موجود فعلاً في الإيميل.
"""
    return f"""
Approved Scenario Card — you MUST use this scenario and must not replace it:
- Scenario ID: {card['id']}
- Main role: {card['role_type']}
- Internal sub-role: {card['sub_role']}
- Core scenario/task idea: {card['scenario']}
- Logical sender/unit: {card['sender']}
- Internal system/procedure: {card['system']}
- Requested action: {card['action']}
- Attack type suitable for this difficulty: {attack}

Dynamic Email Content Engine — mandatory for this generation:
- Writing style this time: {shape['style']}
- Paragraph structure this time: {shape['structure']}
- Subject pattern: {shape['subject_pattern']} + the healthcare task above
- Suggested signature unit: {shape['signature']} with a logical person/title if Advanced
- Required length: {shape['length_hint']}
- Internal diversity nonce: {shape['nonce']}
{event_instruction}
Anti-repetition rules:
- Do not always start with the same sentence.
- Do not always use the same closing line.
- Do not make every email only about a generic account; tie the request to the healthcare task.
- Do not make the AI Tutor Analysis generic. Every indicator must cite a phrase/link/sender that actually appears in the email.
"""

# =============================================================
# UNBOUNDED LEARNING PROMPT
# No fixed templates. No fixed scenario pool. No example domains.
# =============================================================
def build_prompt(role, index, language):
    is_ar = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    seed = random.randint(100000, 999999)

    recipient_email = get_recipient(role, index, language, phase="learn") if role_type != "other" else f"staff.{seed}@hospital.org"
    avoid_topics = get_avoid_list_text(role_type, "learn", is_ar)
    avoid_domains = get_used_domains_text(role_type, "learn", is_ar)
    role_context = get_role_unbounded_context(role_type, is_ar)
    scenario_card = select_scenario_card(role_type, index, phase="learn")
    scenario_instruction = scenario_card_to_prompt(scenario_card, difficulty, is_ar, phase="learn", is_phishing=True)
    diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=is_ar)

    if is_ar:
        return f"""
أنت مولّد أمثلة تدريبية للتوعية بالتصيد في بيئة مستشفى سعودي.

المطلوب: ولّد مثال تعلم واحد فقط لتصيد إلكتروني.

قواعد إلزامية — الارتباط بالوظيفة والسياق الصحي:
- يجب أن يكون الإيميل مرتبطًا 100٪ بالدور الوظيفي المحدد: {role_context}
- كل عنصر في الإيميل (المرسل، الموضوع، المحتوى، الطلب) يجب أن يعكس هذا الدور مباشرةً.
- إذا كان الدور سريريًا: يجب أن يكون المرسل والموضوع والطلب والمحتوى سريريًا فقط: EMR، سجلات مرضى، تسليم سريري، نتائج مختبر، أدوية، صيدلية، أشعة، قسم/ICU/ER. ممنوع مرسل تقني/أمني/إداري/إدارة مستندات/إدارة سجلات.
- إذا كان الدور إداريًا: يجب أن يدور حول العقود أو الفواتير أو الجداول أو سياسات العمل.
- إذا كان الدور تقنيًا: يجب أن يدور حول الأنظمة أو الشبكات أو الأجهزة أو VPN.
- ممنوع تمامًا إرسال إيميل عام لا يعكس الدور المحدد.
- ممنوع تكرار نوع "تقديم العروض التجارية" أو "برامج الرعاية" إلا إذا كان الدور يستدعيه.
- لا تستخدم أي قالب ثابت أو نطاق مكرر.
- استخدم بطاقة السيناريو المعتمدة أعلاه كفكرة الإيميل الأساسية، مع تنويع الصياغة.
- ممنوع استخدام النص الحرفي: suspicious_link داخل body. ضع رابطًا حقيقي الشكل.
- أخرج JSON فقط بدون Markdown.

يجب أن يكون تحليل المعلم الذكي مرتبطًا بالنص فعليًا: كل مؤشر يجب أن يشير لعبارة أو نطاق أو مرسل أو مرفق أو QR أو طلب ظاهر داخل الإيميل. ممنوع التحليل العام أو الثابت.

السياق الوظيفي الإلزامي:
{role_context}
{scenario_instruction}
المستلم: {recipient_email}
رقم عشوائي لكسر التكرار: {seed}
{avoid_topics}{avoid_domains}
قواعد مستوى الصعوبة (إلزامية):
{diff_rule}

قواعد التحية والتوقيع:
- التحية: يجب أن تتبع قواعد مستوى الصعوبة أعلاه بدقة.
- التوقيع: يجب أن ينتهي الإيميل بتوقيع كامل (الاسم + المنصب + القسم) — ممنوع الانتهاء بعنوان بريد إلكتروني مجرد.
- ممنوع وضع الرابط الخام مرتين — مرة كنص ومرة كزر.

قواعد QR الصارمة:
- إذا كان المستوى سهلاً أو متوسطاً: ممنوع منعاً باتاً وضع أي رمز QR — لا تكتب [QR:...] إطلاقاً.
- إذا كان المستوى صعباً: يجب وضع [QR: نص قصير وصفي] في موضع مناسب من الرسالة — هذا إلزامي.

أخرج JSON بهذا الشكل فقط:
{{
  "email_type": "اسم نوع التصيد الجديد",
  "from": "اسم مرسل واقعي <email@invented-domain>",
  "to": "{recipient_email}",
  "subject": "عنوان الرسالة",
  "attachment": "اسم المرفق أو فراغ",
  "body": "نص البريد الكامل",
  "suspicious_text": "أخطر عبارة في الرسالة",
  "suspicious_link": "الرابط المشبوه أو فراغ",
  "indicators": [
    {{"number": 1, "title": "علامة 1", "description": "شرح قصير"}},
    {{"number": 2, "title": "علامة 2", "description": "شرح قصير"}},
    {{"number": 3, "title": "علامة 3", "description": "شرح قصير"}}
  ],
  "why_risky": "لماذا الرسالة خطيرة",
  "learning_tip": "نصيحة تعليمية قصيرة"
}}
"""
    return f"""
You generate phishing-awareness learning examples for a Saudi hospital.

Task: Generate ONE new phishing learning email.

MANDATORY rules — Role Alignment & Healthcare Context:
- The email MUST be 100% aligned with the specified job role: {role_context}
- Every element (sender, subject, body, request) must directly reflect this specific role.
- If the role is CLINICAL: sender, subject, request, and body must be clinical only: EMR, patient records, clinical handover, lab results, medications, pharmacy, radiology, ward/ICU/ER. Do NOT use IT/security/admin/document-collaboration/records-management senders.
- If the role is ADMINISTRATIVE: the email must revolve around contracts, invoices, scheduling, or work policies.
- If the role is IT: the email must revolve around systems, networks, devices, VPN, or software licenses.
- FORBIDDEN: sending a generic wellness program, prize draw, or commercial offer email to a clinical role.
- AVOID repeating commercial offer-type phishing — use it only if rarely used in this session.
- Do NOT use a fixed template or reused domain.
- Use the approved Scenario Card above as the core email idea, while varying the wording.
- Never write the literal placeholder suspicious_link inside body. Use a realistic-looking URL.
- Return JSON only. No Markdown.

AI Tutor Analysis must be grounded: each indicator title/description must point to an actual visible phrase, domain, sender, attachment, QR marker, or request in the generated email. Do not use generic fixed analysis text.

Mandatory role context:
{role_context}
{scenario_instruction}
Recipient: {recipient_email}
Anti-repeat random seed: {seed}
{avoid_topics}{avoid_domains}
Difficulty rules (mandatory):
{diff_rule}

Greeting & Sign-off rules:
- Greeting: must follow the difficulty level rules above precisely.
- Sign-off: must end with a complete signature (name + title + department) — NEVER end with a bare email address.
- NEVER repeat the raw URL twice — once as text AND once as a button.

Strict QR rules:
- If difficulty is EASY or INTERMEDIATE: QR codes are STRICTLY FORBIDDEN — do NOT write [QR:...] anywhere.
- If difficulty is ADVANCED/HARD: a QR code is MANDATORY — you MUST include [QR: short descriptive label] in the body.

Return only this JSON structure:
{{
  "email_type": "new phishing type name",
  "from": "realistic sender name <email@invented-domain>",
  "to": "{recipient_email}",
  "subject": "email subject",
  "attachment": "filename or empty string",
  "body": "full email body",
  "suspicious_text": "most suspicious phrase",
  "suspicious_link": "suspicious URL or empty string",
  "indicators": [
    {{"number": 1, "title": "Indicator 1", "description": "short explanation"}},
    {{"number": 2, "title": "Indicator 2", "description": "short explanation"}},
    {{"number": 3, "title": "Indicator 3", "description": "short explanation"}}
  ],
  "why_risky": "why this email is risky",
  "learning_tip": "short practical learning tip"
}}
"""

# =============================================================
# UNBOUNDED ASSESSMENT PROMPT
# Phishing and legitimate questions are generated dynamically.
# =============================================================
def build_assess_prompt(role, index, is_phishing, language):
    is_ar = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    seed = random.randint(100000, 999999)

    recipient_email = get_recipient(role, index, language, phase="assess") if role_type != "other" else f"staff.{seed}@hospital.org"
    suffix = f"assess_{is_phishing}"
    avoid_topics = get_avoid_list_text(role_type, suffix, is_ar)
    avoid_domains = get_used_domains_text(role_type, suffix, is_ar)
    role_context = get_role_unbounded_context(role_type, is_ar)
    scenario_card = select_scenario_card(role_type, index, phase="assess")
    scenario_instruction = scenario_card_to_prompt(scenario_card, difficulty, is_ar, phase="assess", is_phishing=is_phishing)
    diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=is_phishing, is_ar=is_ar)

    if is_ar:
        label = "تصيد" if is_phishing else "شرعي"
        official = "إذا كانت الرسالة شرعية: استخدم فقط hospital.org أو moh.gov.sa، ولا تضع روابط خارجية أو طلب بيانات حساسة."
        return f"""
أنت مولّد أسئلة اختبار للتوعية بالتصيد في بيئة مستشفى سعودي.

المطلوب: ولّد رسالة اختبار واحدة. التصنيف الصحيح يجب أن يكون: {label}.

قواعد مهمة جدًا:
- لا تستخدم قوالب ثابتة.
- لا تستخدم أي نطاق من أمثلة محفوظة أو نطاقات تكررت سابقًا.
- استخدم بطاقة السيناريو المعتمدة أعلاه فقط، ولا تخترع فكرة مختلفة عنها.
- يجب أن يكون الاختبار متوازنًا: الرسائل الشرعية آمنة فعلًا، ورسائل التصيد فيها علامات حسب مستوى الصعوبة.
- ممنوع استخدام النص الحرفي: suspicious_link داخل body.
- أخرج JSON فقط بدون Markdown.
{official}

السياق:
{role_context}
{scenario_instruction}
المستلم: {recipient_email}
رقم عشوائي لكسر التكرار: {seed}
{avoid_topics}{avoid_domains}
قواعد الصعوبة:
{diff_rule}

أخرج JSON بهذا الشكل فقط:
{{
  "is_phishing": {str(is_phishing).lower()},
  "email_type": "نوع الرسالة الجديد",
  "from": "اسم مرسل واقعي <email@domain>",
  "to": "{recipient_email}",
  "subject": "عنوان الرسالة",
  "attachment": "اسم المرفق أو فراغ",
  "body": "نص البريد الكامل",
  "suspicious_text": "أخطر عبارة أو فراغ إذا شرعي",
  "suspicious_link": "الرابط المشبوه أو فراغ",
  "explanation": "شرح مختصر يوضح لماذا التصنيف صحيح"
}}
"""
    label = "PHISHING" if is_phishing else "LEGITIMATE"
    official = "If legitimate: use only hospital.org or moh.gov.sa, no external links, no sensitive-data request, and no threats."
    return f"""
You generate assessment questions for phishing awareness in a Saudi hospital.

Task: Generate ONE assessment email. Correct label must be: {label}.

Critical rules:
- Do NOT use fixed templates.
- Do NOT use memorized example domains or domains already used in this session.
- Use the approved Scenario Card above only; do not invent a different core idea.
- The assessment must be balanced: legitimate emails must be truly safe, phishing emails must show red flags according to difficulty.
- Never write the literal placeholder suspicious_link inside body. Use a realistic-looking URL when phishing needs a link.
- Return JSON only. No Markdown.
{official}

Context:
{role_context}
{scenario_instruction}
Recipient: {recipient_email}
Anti-repeat random seed: {seed}
{avoid_topics}{avoid_domains}
Difficulty rules:
{diff_rule}

Return only this JSON structure:
{{
  "is_phishing": {str(is_phishing).lower()},
  "email_type": "new email type",
  "from": "realistic sender name <email@domain>",
  "to": "{recipient_email}",
  "subject": "email subject",
  "attachment": "filename or empty string",
  "body": "full email body",
  "suspicious_text": "most suspicious phrase or empty string if legitimate",
  "suspicious_link": "suspicious URL or empty string",
  "explanation": "brief explanation of why the correct label is correct"
}}
"""

def get_system_prompt():
    """
    System prompt without fixed domain examples.
    The detailed 9-criteria difficulty rules live inside build_prompt/build_assess_prompt.
    """
    difficulty = st.session_state.get("difficulty", "medium")
    role = st.session_state.get("role", "Clinical")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    role_context = get_role_unbounded_context(role_type, False)

    return f"""
You are a cybersecurity training content generator for a Saudi healthcare phishing-awareness study.
Role context: {role_context}
Current difficulty: {difficulty}.

Hard rules:
- Return valid JSON only.
- Do not include Markdown or commentary.
- Do not use fixed templates.
- Do not reuse familiar demonstration domains.
- Invent fresh, realistic healthcare workplace scenarios.
- Keep content safe and educational.
- For legitimate emails, use only official domains such as hospital.org or moh.gov.sa and include no suspicious credential/payment/link behavior.
- For phishing emails, generate a clearly educational simulated phishing example with a fake domain and no real organization impersonation beyond generic hospital/MOH-style training context.
""".strip()

def _init_provider_metrics(provider):
    """Initialize metrics dict for a provider if not present"""
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {}
    if provider not in st.session_state["metrics"]:
        st.session_state["metrics"][provider] = {
            "speed": [],        # list of response times in seconds
            "json_ok": 0,       # successful JSON parses
            "json_fail": 0,     # failed JSON parses
            "errors": 0,        # API errors
            "calls": 0,         # total API calls
            "hashes": [],       # content hashes for diversity check
        }

def _record_metric(provider, speed_sec, json_success, content_hash=None, is_error=False):
    """Record a single API call metric, and persist it to disk so it
    survives refresh / new sessions (it used to live only in session_state).
    Also feeds the per-cycle pending-performance bucket so each saved
    cycle can carry its OWN speed/JSON/error/diversity snapshot, instead
    of only the all-time cumulative provider stats."""
    _init_provider_metrics(provider)
    m = st.session_state["metrics"][provider]
    m["calls"] += 1

    language = st.session_state.get("language", "English")
    perf_buckets = _load_json_dict(_PENDING_PERF_FILE_PATH)
    key = f"{provider}__{language}"
    perf_buckets.setdefault(key, []).append({
        "speed": None if is_error else round(speed_sec, 2),
        "json_success": None if is_error else bool(json_success),
        "is_error": is_error,
        "hash": None if is_error else content_hash,
    })
    _save_json_dict(_PENDING_PERF_FILE_PATH, perf_buckets)

    if is_error:
        m["errors"] += 1
        save_metrics_file(st.session_state["metrics"])
        try:
            push_metrics_snapshot_to_gsheet(provider, m)
        except Exception:
            pass
        return
    m["speed"].append(round(speed_sec, 2))
    if json_success:
        m["json_ok"] += 1
    else:
        m["json_fail"] += 1
    if content_hash and content_hash not in m["hashes"]:
        m["hashes"].append(content_hash)
    save_metrics_file(st.session_state["metrics"])
    try:
        push_metrics_snapshot_to_gsheet(provider, m)
    except Exception:
        pass

def call_ai(prompt, max_tokens=1600):
    import time
    provider = st.session_state.get("ai_provider", "groq")
    system_prompt = get_system_prompt()

    def get_secret(key):
        try:
            return st.secrets[key]
        except Exception:
            return os.environ.get(key, "")

    _init_provider_metrics(provider)
    start_time = time.time()

    try:
        if provider == "groq":
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {get_secret('GROQ_API_KEY')}"
                },
                json={
                    "model":       "llama-3.3-70b-versatile",
                    "max_tokens":  max_tokens,
                    "temperature": 0.85,
                    "messages":    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt}
                    ]
                },
                timeout=45
            )
            data = resp.json()
            elapsed = time.time() - start_time
            _record_metric(provider, elapsed, "choices" in data, str(hash(str(data)))[:8])
            return data

        elif provider == "openai":
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {get_secret('OPENAI_API_KEY')}"
                },
                json={
                    "model":       "gpt-4o",
                    "max_tokens":  max_tokens,
                    "temperature": 0.85,
                    "response_format": {"type": "json_object"},
                    "messages":    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt}
                    ]
                },
                timeout=60
            )
            data = resp.json()
            elapsed = time.time() - start_time
            _record_metric(provider, elapsed, "choices" in data, str(hash(str(data)))[:8])
            return data

        elif provider == "anthropic":
            # Claude needs a bigger token budget than the other providers for
            # the same request: longer/more detailed answers (as observed)
            # plus any internal reasoning blocks both draw from max_tokens.
            # Without enough headroom, Arabic responses in particular were
            # getting cut off before any real text was produced.
            anthropic_max_tokens = max(max_tokens, 3500)
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         get_secret("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model":      "claude-sonnet-4-6",
                    "max_tokens": anthropic_max_tokens,
                    "system":     system_prompt,
                    "messages":   [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=60
            )
            raw = resp.json()
            elapsed = time.time() - start_time
            if "content" in raw and len(raw["content"]) > 0:
                # Claude can return multiple content blocks (e.g. a "thinking"
                # block followed by the actual "text" block, especially for
                # longer/more complex responses such as Arabic generations).
                # Reading only content[0] silently returned an empty string
                # in those cases. We now join ALL text-type blocks instead.
                text = "".join(
                    (block.get("text") or "")
                    for block in raw["content"]
                    if block.get("type") == "text"
                )
                if not text.strip():
                    # No usable text block found at all — surface as an error
                    # instead of returning an empty string that later fails
                    # JSON parsing with a confusing "char 0" message.
                    _record_metric(provider, elapsed, False, is_error=True)
                    return {"error": {"message": f"Claude returned no text content (stop_reason={raw.get('stop_reason')}): {str(raw)[:300]}"}}
                _record_metric(provider, elapsed, True, str(hash(text))[:8])
                return {"choices": [{"message": {"content": text}}]}
            _record_metric(provider, elapsed, False, is_error=True)
            return {"error": {"message": str(raw)[:300]}}

        elif provider == "gemini":
            api_key = get_secret("GEMINI_API_KEY")
            # Gemini 2.5 Flash is a "thinking" model: by default it spends
            # part of maxOutputTokens on internal reasoning before writing
            # the visible answer. With a small token budget, thinking alone
            # could consume it all, leaving an empty text part (which is
            # exactly the "char 0" JSON parse error seen above). We disable
            # thinking (not needed for this task) and give a safe token
            # floor, mirroring the same fix already applied for Claude.
            gemini_max_tokens = max(max_tokens, 2400)
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": system_prompt + "\n\n" + prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": gemini_max_tokens,
                        "temperature":     0.85,
                        "responseMimeType": "application/json",
                        "thinkingConfig":  {"thinkingBudget": 0}
                    }
                },
                timeout=60
            )
            raw = resp.json()
            elapsed = time.time() - start_time
            try:
                parts = raw["candidates"][0]["content"]["parts"]
                text = "".join((p.get("text") or "") for p in parts if not p.get("thought"))
                if not text.strip():
                    _record_metric(provider, elapsed, False, is_error=True)
                    return {"error": {"message": f"Gemini returned no text content (finishReason={raw['candidates'][0].get('finishReason')}): {str(raw)[:300]}"}}
                _record_metric(provider, elapsed, True, str(hash(text))[:8])
                return {"choices": [{"message": {"content": text}}]}
            except (KeyError, IndexError):
                _record_metric(provider, elapsed, False, is_error=True)
                return {"error": raw}

        else:
            return {"error": f"Unknown provider: {provider}"}

    except Exception as e:
        elapsed = time.time() - start_time
        _record_metric(provider, elapsed, False, is_error=True)
        return {"error": str(e)}

def call_groq(prompt, max_tokens=1600):
    return call_ai(prompt, max_tokens)

def _escape_stray_inner_quotes(s):
    """Best-effort repair for a common AI-generation failure: a literal
    double-quote character used INSIDE a JSON string value (e.g. the model
    quoting a word in Arabic/English text) instead of a properly escaped
    \\" or a single quote. We walk the string and re-escape any double-quote
    that doesn't actually look like a string delimiter (i.e. it isn't
    immediately followed by a JSON structural character like , : } ] or
    whitespace+one of those)."""
    out = []
    in_string = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            if not in_string:
                in_string = True
                out.append(c)
            else:
                # We're inside a string — decide if this quote is the
                # closing delimiter or a stray quote inside the text.
                j = i + 1
                while j < n and s[j] in ' \t\r\n':
                    j += 1
                if j >= n or s[j] in ',:}]':
                    in_string = False
                    out.append(c)
                else:
                    out.append('\\"')
        else:
            out.append(c)
        i += 1
    return ''.join(out)

def parse_json_response(raw):
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    raw = fix_json_newlines(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r"(?<=\w)'(?=\w)", "\u2019", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Last resort: repair stray unescaped quotes inside string values
        try:
            return json.loads(_escape_stray_inner_quotes(candidate))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Cannot parse JSON", raw, 0)

def clean_result(result, is_arabic):
    for f in ["body","suspicious_text","why_risky","learning_tip","subject","email_type"]:
        if result.get(f):
            result[f] = clean_foreign_only(result[f])
            if is_arabic:
                result[f] = remove_foreign_latin_words(result[f])
    for ind in result.get("indicators",[]):
        for k in ["title","description"]:
            if ind.get(k):
                ind[k] = clean_foreign_only(ind[k])
                if is_arabic:
                    ind[k] = remove_foreign_latin_words(ind[k])
    result["from"] = clean_email_field((result.get("from") or ""))
    result["to"] = extract_to_email((result.get("to") or ""))
    if result.get("suspicious_link"):
        sl = result["suspicious_link"]
        sl = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',sl).strip()
        sl = re.sub(r'[\u0600-\u06ff\s]','',sl)
        result["suspicious_link"] = sl
    return result


def build_other_analysis_prompt(email_data, language, difficulty):
    """يطلب من اللـ LLM الـ AI Analysis فقط للإيميل الجاهز"""
    is_ar = (language == "ar")
    body_preview = email_data["body"][:300].replace('"', "'")
    
    lang_rule = "Respond entirely in Arabic." if is_ar else "Respond entirely in English."
    
    diff_hints = {
        "easy": "indicators should be obvious and clear for beginners",
        "medium": "indicators should be moderately detailed",
        "hard": "indicators should be subtle and detailed for advanced learners"
    }
    diff_hint = diff_hints.get(difficulty, diff_hints["medium"])
    
    return f"""You are a cybersecurity expert analyzing a phishing email for hospital staff awareness training.

{lang_rule}

Here is the phishing email to analyze:
FROM: {email_data["from"]}
SUBJECT: {email_data["subject"]}
BODY PREVIEW: {body_preview}

Your task: Generate ONLY the AI Tutor Analysis for this email.
Difficulty hint: {diff_hint}
Be concise: each indicator description is ONE short sentence (max ~20 words), why_risky is max 2 short sentences, learning_tip is ONE short sentence.

RETURN ONLY VALID JSON — no text before or after:
{{"indicators":[{{"number":1,"title":"indicator title","description":"detailed explanation"}},{{"number":2,"title":"indicator title","description":"detailed explanation"}},{{"number":3,"title":"indicator title","description":"detailed explanation"}}],"why_risky":"why this specific phishing email is dangerous for hospital staff","learning_tip":"practical tip for hospital staff to avoid this attack"}}"""


def generate_other_email(index, language, difficulty):
    """Dynamic Other learning email. No static templates."""
    role = "Other" if language != "Arabic" else "أخرى"
    is_ar = (language == "Arabic")
    last_issues = []
    for attempt in range(3):
        try:
            prompt = build_prompt(role, index, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data['error'].get('message', str(data['error']))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, is_ar)
            if (result.get("suspicious_link") or "").strip() and result["suspicious_link"] not in (result.get("body") or ""):
                result["body"] = _insert_before_signature(result.get("body") or "", result["suspicious_link"])
            result["body"] = _reposition_trailing_lone_link(result.get("body") or "", result.get("suspicious_link") or "")

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), True)
            if not last_issues or attempt == 2:
                remember_generated_artifacts("other", "learn", result)
                return result
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Generation failed quality checks."}


def generate_other_assess_email(index, is_phishing, language, difficulty):
    """Dynamic Other assessment email. No static templates."""
    role = "Other" if language != "Arabic" else "أخرى"
    is_ar = (language == "Arabic")
    last_issues = []
    for attempt in range(3):
        try:
            prompt = build_assess_prompt(role, index, is_phishing, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data['error'].get('message', str(data['error']))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, is_ar)
            result["is_phishing"] = bool(is_phishing)
            if (result.get("suspicious_link") or "").strip() and result["suspicious_link"] not in (result.get("body") or ""):
                result["body"] = _insert_before_signature(result.get("body") or "", result["suspicious_link"])
            result["body"] = _reposition_trailing_lone_link(result.get("body") or "", result.get("suspicious_link") or "")

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), is_phishing)
            if not last_issues or attempt == 2:
                remember_generated_artifacts("other", f"assess_{is_phishing}", result)
                return result
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Generation failed quality checks."}


def generate_email(role, index, language, difficulty="medium"):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        return generate_other_email(index, language, difficulty)

    is_ar = (language == "Arabic")
    last_issues = []
    for attempt in range(3):
        try:
            prompt = build_prompt(role, index, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data['error'].get('message', str(data['error']))}
            if "choices" not in data:
                return {"error": f"Unexpected API response: {str(data)[:200]}"}
            raw    = data["choices"][0]["message"]["content"].strip()
            result = parse_json_response(raw)
            result = clean_result(result, is_ar)
            result["to"] = get_recipient(role, index, language, phase="learn")
            if (result.get("suspicious_link") or "").strip():
                if result["suspicious_link"] not in (result.get("body") or ""):
                    result["body"] = _insert_before_signature(result.get("body") or "", result["suspicious_link"])
            result["body"] = _reposition_trailing_lone_link(result.get("body") or "", result.get("suspicious_link") or "")

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), True)
            if not last_issues:
                remember_generated_artifacts(role_type, "learn", result)
                return result
            if attempt == 2:
                # Normal retries exhausted. If the failure is specifically a
                # severe role-mismatch (generic/commercial content unrelated
                # to the role — the drift we kept seeing in testing), try ONE
                # more real, dynamically generated attempt from the AI itself
                # with an explicitly pinned scenario (still a live API call —
                # never hand-written text). Whatever the AI returns — pinned
                # attempt or the original — is what gets shown; we never
                # substitute our own authored content.
                severe = any(("role context" in i) or ("commercial" in i) for i in last_issues)
                if severe:
                    forced = _forced_role_aligned_attempt(role, role_type, index, language, is_ar)
                    if forced is not None:
                        remember_generated_artifacts(role_type, "learn", forced)
                        return forced
                remember_generated_artifacts(role_type, "learn", result)
                return result
        except json.JSONDecodeError as e:
            if attempt == 2:
                return {"error": f"JSON parse error: {e}"}
            last_issues = [f"invalid JSON: {e}"]
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Generation failed quality checks."}


def _forced_role_aligned_attempt(role, role_type, index, language, is_ar):
    """Last-resort guaranteed attempt: pin ONE specific, pre-approved Attack
    Playbook scenario for this role (instead of the open-ended 'invent your
    own idea' instruction) so there is no room left for the model to drift
    into a generic/commercial theme unrelated to the role. Returns None on
    any failure so the caller falls back to the previous (possibly
    imperfect) result rather than crashing."""
    try:
        pool = ATTACK_PLAYBOOK.get(role_type)
        if not pool:
            return None
        item = random.choice(pool)
        difficulty = st.session_state.get("difficulty", "medium")
        recipient_email = get_recipient(role, index, language, phase="learn")
        role_context = get_role_unbounded_context(role_type, is_ar)
        diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=is_ar)
        seed = random.randint(100000, 999999)
        scenario = item["ar"] if is_ar else item["en"]
        if is_ar:
            prompt = f"""
أنت مولّد أمثلة تصيد تدريبية لمستشفى سعودي.
يجب أن يكون الإيميل حصراً عن هذا السيناريو المحدد سلفاً، بدون أي انحراف عنه:
نوع الهجوم: {item['attack']}
السيناريو الإلزامي: {scenario}
ناقل الهجوم: {item['vector']}
السياق الوظيفي: {role_context}
المستلم: {recipient_email}
رقم عشوائي: {seed}
قواعد مستوى الصعوبة:
{diff_rule}
ممنوع منعاً باتاً أي محتوى عن جوائز أو عروض بنكية أو عروض تجارية أو مكافآت مالية — يجب أن يدور المحتوى فعلياً حول السيناريو أعلاه فقط.
أخرج JSON فقط بهذا الشكل:
{{"email_type": "{item['attack']}", "from": "اسم مرسل واقعي <email@invented-domain>", "to": "{recipient_email}", "subject": "عنوان الرسالة", "attachment": "", "body": "نص البريد الكامل", "suspicious_text": "أخطر عبارة", "suspicious_link": "الرابط المشبوه", "injected_errors": [], "indicators": [{{"number":1,"title":"مؤشر 1","description":"شرح"}},{{"number":2,"title":"مؤشر 2","description":"شرح"}},{{"number":3,"title":"مؤشر 3","description":"شرح"}}], "why_risky": "شرح الخطورة", "learning_tip": "نصيحة قصيرة"}}
"""
        else:
            prompt = f"""
You generate phishing training examples for a Saudi hospital.
The email MUST be exclusively about this pre-approved scenario, with NO deviation:
Attack type: {item['attack']}
Mandatory scenario: {scenario}
Attack vector: {item['vector']}
Role context: {role_context}
Recipient: {recipient_email}
Anti-repeat seed: {seed}
Difficulty rules:
{diff_rule}
STRICTLY FORBIDDEN: any content about prizes, bank offers, commercial promotions, or salary bonuses — the content must genuinely revolve around the scenario above only.
Return ONLY this JSON structure:
{{"email_type": "{item['attack']}", "from": "realistic sender name <email@invented-domain>", "to": "{recipient_email}", "subject": "email subject", "attachment": "", "body": "full email body", "suspicious_text": "most suspicious phrase", "suspicious_link": "suspicious URL", "injected_errors": [], "indicators": [{{"number":1,"title":"Indicator 1","description":"explanation"}},{{"number":2,"title":"Indicator 2","description":"explanation"}},{{"number":3,"title":"Indicator 3","description":"explanation"}}], "why_risky": "why this is risky", "learning_tip": "short tip"}}
"""
        data = call_groq(prompt, max_tokens=2000)
        if "error" in data or "choices" not in data:
            return None
        raw = data["choices"][0]["message"]["content"].strip()
        result = parse_json_response(raw)
        if not isinstance(result, dict) or "error" in result:
            return None
        result = clean_result(result, is_ar)
        result["to"] = recipient_email
        if (result.get("suspicious_link") or "").strip():
            if result["suspicious_link"] not in (result.get("body") or ""):
                result["body"] = _insert_before_signature(result.get("body") or "", result["suspicious_link"])
        result["body"] = _reposition_trailing_lone_link(result.get("body") or "", result.get("suspicious_link") or "")
        return result
    except Exception:
        return None


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        return generate_other_assess_email(index, is_phishing, language, difficulty)

    is_ar = (language == "Arabic")
    last_issues = []
    for attempt in range(3):
        try:
            prompt = build_assess_prompt(role, index, is_phishing, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data["error"].get("message", str(data["error"]))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, is_ar)
            result["to"] = get_recipient(st.session_state.get("role","Clinical"), index, language, phase="assess")
            result["is_phishing"] = bool(is_phishing)
            if (result.get("suspicious_link") or "").strip():
                if result["suspicious_link"] not in (result.get("body") or ""):
                    result["body"] = _insert_before_signature(result.get("body") or "", result["suspicious_link"])
            result["body"] = _reposition_trailing_lone_link(result.get("body") or "", result.get("suspicious_link") or "")

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), is_phishing)
            if not last_issues or attempt == 2:
                remember_generated_artifacts(role_type, f"assess_{is_phishing}", result)
                return result
        except json.JSONDecodeError:
            if attempt == 2:
                return {"error": "Failed to parse. Please try again."}
            last_issues = ["invalid JSON"]
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Generation failed quality checks."}

def render_email_window(email, is_arabic, show_badges=False):
    bd = 'rtl' if is_arabic else 'ltr'
    ta = 'right' if is_arabic else 'left'
    email_font = 'Tahoma,Arial,sans-serif' if is_arabic else "'Courier New',monospace"

    # FINAL TYPE-SAFETY NET: no matter what slipped through every earlier
    # parsing/recovery step, these fields must be plain strings here, or
    # every re.sub()/.strip() below will crash with "expected string or
    # bytes-like object, got 'dict'". Coerce defensively at the last mile.
    def _as_text(v):
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        return str(v)
    email = {**email, **{
        k: _as_text(email.get(k)) for k in
        ["body", "subject", "from", "to", "suspicious_text", "suspicious_link", "attachment"]
        if k in email
    }}

    body_raw        = re.sub(r'<[^>]+>','', (email.get("body") or ""))
    suspicious_text = re.sub(r'<[^>]+>','', (email.get("suspicious_text") or ""))
    suspicious_link = re.sub(r'<[^>]+>','', (email.get("suspicious_link") or "")).strip()


    # Badge numbers come from the same grounded indicators shown in the tutor.
    # This removes the old hard-coded 1/2/3/4 mismatch.
    _inds = email.get("indicators") if isinstance(email.get("indicators"), list) else []
    def _badge_for_target(target):
        for _it in _inds:
            if isinstance(_it, dict) and _it.get("target") == target:
                try:
                    return int(_it.get("number"))
                except Exception:
                    return None
        return None
    def _badge_for_key(key):
        for _it in _inds:
            if isinstance(_it, dict) and _it.get("key") == key:
                try:
                    return int(_it.get("number"))
                except Exception:
                    return None
        return None

    body_raw = re.sub(r'suspicious_link\s*:\s*', '', body_raw, flags=re.IGNORECASE)
    body_raw = re.sub(r'suspicious_text\s*:\s*', '', body_raw, flags=re.IGNORECASE)
    # Remove placeholder phone numbers like +966-XX-XXXXXXX or +1-XXX-XXX-XXXX
    body_raw = re.sub(r'\+?\d{1,4}[-\s]?(?:XX|\d{2,3})[-\s]?(?:XX+|\d{3,4})(?:[-\s]?(?:XX+|\d{3,4}))*', '', body_raw)
    body_raw = re.sub(r'Contact:\s*\n', '', body_raw)
    body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    # --------------------------------------------------------
    # NEW: detect a "[QR Code: label]" / "[QR: label]" placeholder
    # inside the body. Instead of deleting it, swap it for a unique
    # token that stays exactly where the model put it — so the real
    # QR image ends up rendered IN PLACE (not dumped at the very
    # bottom after the signature).
    # --------------------------------------------------------
    qr_label, has_qr = "", False
    _difficulty = st.session_state.get("difficulty", "medium")

    qr_match = re.search(r'\[\s*QR(?:\s*Code)?\s*:?\s*([^\]]*)\]', body_raw, re.I)
    if qr_match:
        if _difficulty in ("easy", "medium"):
            # QR is FORBIDDEN in easy/medium — remove it completely
            body_raw = body_raw[:qr_match.start()] + "" + body_raw[qr_match.end():]
            body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()
        else:
            # Hard/Advanced — QR is now OPTIONAL (varies per example, not
            # mandatory in every single one) — keep it only if the model
            # actually wrote a [QR:...] marker naturally in the body.
            has_qr = True
            qr_label = qr_match.group(1).strip()
            body_raw = body_raw[:qr_match.start()] + "@@QR_TOKEN@@" + body_raw[qr_match.end():]
    # NOTE: no forced/auto-injected QR anymore for hard/advanced — if the
    # model didn't naturally include one, this example simply has no QR,
    # which is the intended variation (some Advanced emails have QR,
    # some don't, per the updated requirement).

    # --------------------------------------------------------
    # NEW: detect a markdown-style "[Button label](https://...)"
    # link inside the body. Same idea: swap for a token so the real
    # clickable button renders IN PLACE of where the link was
    # written, not always at the bottom of the email.
    # --------------------------------------------------------
    link_label, link_url, has_link_button = "", "", False
    link_match = re.search(r'\[([^\]]{1,80})\]\s*\(\s*(https?://[^\)\s]+)\s*\)', body_raw)
    if link_match:
        if _difficulty == "easy":
            # Easy level: NO button allowed — convert to plain text URL
            _raw_url = link_match.group(2).strip()
            body_raw = body_raw[:link_match.start()] + _raw_url + body_raw[link_match.end():]
            # Also capture for suspicious_link display if not already set
            if not email.get("suspicious_link"):
                email["suspicious_link"] = _raw_url
        else:
            has_link_button = True
            link_label = link_match.group(1).strip()
            link_url   = link_match.group(2).strip()
            body_raw   = body_raw[:link_match.start()] + "@@LINK_TOKEN@@" + body_raw[link_match.end():]

    # Safety net: if the MODEL itself wrote the [QR:...] or [Label](url)
    # marker as the very last line (after the closing signature) instead
    # of inline where it's referenced, reposition it before the
    # signature — same fix as for a bare suspicious_link, generalized to
    # the token placeholders used here.
    for _marker_token in ("@@QR_TOKEN@@", "@@LINK_TOKEN@@"):
        if _marker_token in body_raw:
            body_raw = _reposition_trailing_lone_link(body_raw, _marker_token)

    # Also for easy: if suspicious_link exists but no visible URL in body, append it
    if _difficulty == "easy" and (email.get("suspicious_link") or "").strip():
        _sl = (email.get("suspicious_link") or "").strip()
        # Always run _place_link_in_body — it handles all cases:
        # link missing, link after signature, link already correct.
        if _sl and not has_link_button:
            body_raw = _place_link_in_body(body_raw, _sl)

    # Tidy up extra blank lines left behind after removing the placeholders above.
    body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    # --------------------------------------------------------
    # NEW: providers (Groq/Llama especially) often write a trailing
    # punctuation mark right after the markdown link/QR syntax, e.g.
    # "...here: [Button](url). Your input is crucial..." — once the
    # bracket+url is swapped for a token, that stray "." (or ",")
    # is left dangling right next to the token and renders as an
    # ugly lone punctuation line. Strip stray punctuation that sits
    # immediately before/after either token (allowing whitespace).
    # --------------------------------------------------------
    body_raw = re.sub(r'[ \t]*[.,؛;:]+[ \t]*(@@(?:QR|LINK)_TOKEN@@)', r'\1', body_raw)
    body_raw = re.sub(r'(@@(?:QR|LINK)_TOKEN@@)[ \t]*[.,؛]+(?=[ \t]*(\n|$))', r'\1', body_raw)
    body_raw = re.sub(r'(@@(?:QR|LINK)_TOKEN@@)[ \t]*[.,؛]+[ \t]*', r'\1 ', body_raw)
    body_raw = body_raw.strip()

    # --------------------------------------------------------
    # Remove duplicate URL — strip any standalone line that is exactly the suspicious_link
    # This applies ALWAYS (not just when has_qr or has_link_button) to prevent double display
    # NOTE: For easy level the link was already placed correctly by _place_link_in_body above.
    # This block only removes truly redundant duplicate occurrences.
    if suspicious_link and (has_qr or has_link_button):
        bare_link_pattern = re.escape(suspicious_link)
        bare_no_scheme    = re.escape(re.sub(r'^https?://', '', suspicious_link))
        body_raw = re.sub(rf'^[ \t]*{bare_link_pattern}[ \t]*$\n?', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(rf'^[ \t]*{bare_no_scheme}[ \t]*$\n?', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    # For medium/hard: if no button and no QR and link missing — place correctly, not at end
    if _difficulty == "medium" and suspicious_link and not has_qr and not has_link_button:
        link_bare = re.sub(r'^https?://', '', suspicious_link)
        if suspicious_link not in body_raw and link_bare not in body_raw:
            body_raw = _place_link_in_body(body_raw, suspicious_link)


    # Keep the suspicious URL exactly once and always before the signature.
    # Provider output may include it inline and again as a trailing standalone line.
    if suspicious_link:
        _url_re = re.compile(re.escape(suspicious_link), re.I)
        _seen = [0]
        def _keep_first_url(m):
            _seen[0] += 1
            return m.group(0) if _seen[0] == 1 else ""
        body_raw = _url_re.sub(_keep_first_url, body_raw)
        body_raw = re.sub(r'[ \t]+\n', '\n', body_raw)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()
        if _seen[0] == 0:
            body_raw = _place_link_in_body(body_raw, suspicious_link)
        body_raw = _reposition_trailing_lone_link(body_raw, suspicious_link)

    has_attachment  = bool((email.get("attachment") or "").strip())

    # Remove duplicate attachment filename that sometimes appears as plain text at end
    _att_name = (email.get("attachment") or "").strip()
    if _att_name:
        att_escaped = re.escape(_att_name)
        body_raw = re.sub(rf'^\s*Attachment\s*:\s*{att_escaped}\s*$', '', body_raw, flags=re.MULTILINE|re.IGNORECASE)
        body_raw = re.sub(rf'^\s*{att_escaped}\s*$', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    body_html = html_lib.escape(body_raw)

    def make_badge(n, color="#DC2626"):
        return (f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'width:20px;height:20px;border-radius:50%;background:{color};color:white;'
                f'font-size:.7rem;font-weight:800;margin:0 3px;vertical-align:middle;">{n}</span>')

    if show_badges:
        # Mark every body/greeting/link indicator with its own tutor number.
        for _it in _inds:
            if not isinstance(_it, dict):
                continue
            _target = _it.get("target")
            _evidence = str(_it.get("evidence") or "")
            try:
                _num = int(_it.get("number"))
            except Exception:
                continue
            if _target not in ("body", "greeting", "link") or not _evidence:
                continue
            _safe = html_lib.escape(_evidence)
            if _safe in body_html:
                _style = ('color:#60A5FA;text-decoration:underline;' if _target == "link" else 'color:#FCA5A5;')
                body_html = body_html.replace(
                    _safe,
                    f'<span style="border:2px solid rgba(239,68,68,.6);border-radius:7px;'
                    f'padding:.2rem .5rem;background:rgba(239,68,68,.08);{_style}'
                    f'box-decoration-break:clone;-webkit-box-decoration-break:clone;">'
                    f'{make_badge(_num)}{_safe}</span>', 1)

    body_html = body_html.replace("\n","<br>")

    # --------------------------------------------------------
    # NEW: real, scannable QR-code image (rendered after body text).
    # Uses the suspicious_link (or the button URL, as a fallback) as
    # the QR payload so the badge/number still points to a real risk.
    # --------------------------------------------------------
    qr_block_html = ""
    if has_qr:
        qr_data    = suspicious_link or link_url or "https://example-training-only.invalid/qr"
        qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={urllib.parse.quote(qr_data, safe='')}"
        qr_badge   = make_badge(next_badge()) if show_badges else ""
        qr_caption = html_lib.escape(qr_label) if qr_label else t("Scan with your phone","امسح بهاتفك")
        qr_block_html = f"""
<div style="margin:1rem 0;display:flex;align-items:center;gap:.8rem;direction:{bd};flex-wrap:wrap;">
  {qr_badge}
  <div style="background:#ffffff;padding:8px;border-radius:10px;border:2px solid rgba(239,68,68,.55);display:inline-block;line-height:0;">
    <img src="{qr_img_url}" width="120" height="120" alt="QR code" style="display:block;border-radius:2px;"/>
  </div>
  <div style="color:#94A3B8;font-size:.85rem;">{qr_caption}</div>
</div>"""

    # --------------------------------------------------------
    # NEW: a real clickable link "button" — same visual language as
    # the attachment chip — instead of leaving "[Label] (url)" as
    # inert bracketed text.
    # --------------------------------------------------------
    # Warning page trigger ID
    _warn_key = f"phishing_warn_{id(email)}"

    link_block_html = ""
    if has_link_button:
        link_badge = make_badge(next_badge()) if show_badges else ""
        # Replace generic "Open Link" label with descriptive fallback
        _display_label = link_label
        if _display_label.strip().lower() in ("open link", "click here", "link", "رابط", "اضغط هنا", "فتح الرابط"):
            _display_label = t("View Document", "عرض المستند")
        safe_label = html_lib.escape(_display_label)
        link_block_html = f"""
<div style="margin:.8rem 0;direction:{bd};">
  <button onclick="window.parent.postMessage({{type:'phishing_click',element:'link',label:'{safe_label}'}},\'*\')"
     style="display:inline-flex;align-items:center;gap:.5rem;border:1px solid #0078D4;
            border-radius:6px;padding:.5rem 1.2rem;background:#0078D4;color:white;
            font-size:.92rem;font-weight:700;cursor:pointer;font-family:inherit;
            box-shadow:0 2px 6px rgba(0,120,212,.4);"
     onmouseover="this.style.background='#006CBE'" onmouseout="this.style.background='#0078D4'">
    {link_badge}🔗 {safe_label}
  </button>
</div>"""

    # --------------------------------------------------------
    # NEW: inject the QR image / link button INTO the body at the
    # exact spot the model originally placed it (via the tokens
    # above), instead of always dumping it after the signature.
    # If, for any reason, the token didn't survive into body_html
    # (e.g. the model wrote the placeholder oddly), fall back to
    # appending the block at the end so it never gets silently lost.
    # --------------------------------------------------------
    if has_qr:
        if "@@QR_TOKEN@@" in body_html:
            body_html = body_html.replace("@@QR_TOKEN@@", qr_block_html)
        else:
            body_html += qr_block_html
    if has_link_button:
        if "@@LINK_TOKEN@@" in body_html:
            body_html = body_html.replace("@@LINK_TOKEN@@", link_block_html)
        else:
            body_html += link_block_html

    from_val = html_lib.escape((email.get("from") or ""))
    to_val   = html_lib.escape(email.get("to","employee@hospital.org"))
    subj_val = html_lib.escape((email.get("subject") or ""))
    att_val  = html_lib.escape((email.get("attachment") or ""))

    fl = t("From:","من:")
    tl = t("To:","إلى:")
    sl = t("Subject:","الموضوع:")

    _n_from = _badge_for_target("from")
    _n_subj = _badge_for_target("subject")
    _n_att  = _badge_for_target("attachment")
    b_from = make_badge(_n_from) if show_badges and _n_from else ""
    b_subj = make_badge(_n_subj) if show_badges and _n_subj else ""
    b_att  = make_badge(_n_att) if show_badges and _n_att else ""

    att_html = ""
    if att_val:
        att_html = (f'<div style="display:inline-flex;align-items:center;gap:.5rem;'
                    f'border:1px solid #0078D4;border-radius:4px;padding:.4rem .8rem;'
                    f'background:rgba(0,120,212,.12);color:#60A5FA;font-size:.88rem;margin:.4rem 0;'
                    f'cursor:pointer;" onclick="window.parent.postMessage({{type:\'phishing_click\',element:\'attachment\'}},\'*\')">'
                    f'{b_att}📎 {att_val}</div>')

    # Outlook-style toolbar buttons (visual only)
    reply_lbl    = "رد" if is_arabic else "Reply"
    forward_lbl  = "إعادة توجيه" if is_arabic else "Forward"
    delete_lbl   = "حذف" if is_arabic else "Delete"
    toolbar_dir  = "rtl" if is_arabic else "ltr"

    # Research v15: use generated display_time when available so dates vary realistically
    # and remain stable across Streamlit reruns. Fallback keeps old behavior.
    _email_time = (email.get("display_time") or "").strip()
    if not _email_time:
        _rand_hour   = random.randint(7, 16)
        _rand_min    = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        _period_en   = "AM" if _rand_hour < 12 else "PM"
        _hour_12     = _rand_hour if _rand_hour <= 12 else _rand_hour - 12
        _hour_12     = 12 if _hour_12 == 0 else _hour_12
        _period_ar   = "ص" if _rand_hour < 12 else "م"
        _time_en     = f"Today, {_hour_12}:{_rand_min:02d} {_period_en}"
        _time_ar     = f"اليوم، {_hour_12}:{_rand_min:02d} {_period_ar}"
        _email_time  = _time_ar if is_arabic else _time_en

    st.markdown(f"""
<div style="background:#D8DCE1;border:1px solid #C5CAD0;border-radius:12px 12px 0 0;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif;box-shadow:0 4px 24px rgba(0,0,0,.5);">
  <!-- Outlook-style title bar -->
  <div style="background:#CDD2D8;padding:.45rem 1rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #B5BAC0;">
    <div style="display:flex;gap:6px;align-items:center;">
      <div style="width:12px;height:12px;border-radius:50%;background:#FF5F57;"></div>
      <div style="width:12px;height:12px;border-radius:50%;background:#FFBD2E;"></div>
      <div style="width:12px;height:12px;border-radius:50%;background:#28C840;"></div>
    </div>
    <div style="color:#555;font-size:.78rem;letter-spacing:.5px;">📧 {"صندوق الوارد — Microsoft Outlook" if is_arabic else "Inbox — Microsoft Outlook"}</div>
    <div style="width:60px;"></div>
  </div>
  <!-- Outlook-style action toolbar -->
  <div style="background:#DDE1E6;padding:.35rem 1rem;display:flex;gap:.5rem;align-items:center;border-bottom:1px solid #C5CAD0;direction:{toolbar_dir};">
    <button style="background:#0078D4;color:white;border:none;border-radius:4px;padding:.28rem .85rem;font-size:.78rem;cursor:pointer;font-family:inherit;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.2);transition:background .15s;" onmouseover="this.style.background='#006CBE'" onmouseout="this.style.background='#0078D4'">↩ {reply_lbl}</button>
    <button style="background:#F3F4F6;color:#374151;border:1px solid #CBD5E1;border-radius:4px;padding:.28rem .85rem;font-size:.78rem;cursor:pointer;font-family:inherit;font-weight:500;" onmouseover="this.style.background='#E5E7EB'" onmouseout="this.style.background='#F3F4F6'">→ {forward_lbl}</button>
    <button style="background:#F3F4F6;color:#374151;border:1px solid #CBD5E1;border-radius:4px;padding:.28rem .85rem;font-size:.78rem;cursor:pointer;font-family:inherit;font-weight:500;" onmouseover="this.style.background='#FEE2E2';this.style.color='#DC2626'" onmouseout="this.style.background='#F3F4F6';this.style.color='#374151'">🗑 {delete_lbl}</button>
    <div style="flex:1;"></div>
    <span style="color:#6B7280;font-size:.75rem;font-style:italic;">{_email_time}</span>
  </div>
  <!-- Email header -->
  <div style="padding:.9rem 1.6rem .5rem;font-size:.92rem;color:#CBD5E1;direction:{bd};text-align:{ta};background:#DDE1E6;border-bottom:1px solid #C5CAD0;">
    <table style="width:100%;border-collapse:collapse;direction:{bd};">
      <tr style="vertical-align:top;">
        <td style="color:#6B7280;font-weight:600;padding:0 8px 6px 0;white-space:nowrap;width:80px;font-size:.85rem;">{fl}</td>
        <td style="color:#111827;padding:0 0 6px 0;word-break:break-all;">{b_from}{from_val}</td>
      </tr>
      <tr style="vertical-align:middle;">
        <td style="color:#6B7280;font-weight:600;padding:0 8px 6px 0;white-space:nowrap;font-size:.85rem;">{tl}</td>
        <td style="color:#60A5FA;padding:0 0 6px 0;direction:ltr;text-align:{('right' if bd=='rtl' else 'left')};overflow:hidden;text-overflow:ellipsis;">{to_val}</td>
      </tr>
      <tr style="vertical-align:top;">
        <td style="color:#6B7280;font-weight:600;padding:0 8px 6px 0;white-space:nowrap;font-size:.85rem;">{sl}</td>
        <td style="color:#111827;padding:0 0 6px 0;word-break:break-word;font-weight:700;">{b_subj}{subj_val}</td>
      </tr>
    </table>
    {att_html}
  </div>
</div>
<div style="background:#E6EAF2;border:1px solid #CDD1D6;border-top:none;
            border-radius:0 0 12px 12px;padding:1.2rem 1.6rem 1.6rem;
            font-family:'Segoe UI',Arial,sans-serif;
            font-size:.93rem;color:#1F2937;background:#E6EAF2;
            line-height:1.9;direction:{bd};text-align:{ta};
            box-shadow:0 20px 60px rgba(0,0,0,.3);">
  {body_html}
</div>""", unsafe_allow_html=True)


def page_phishing_caught():
    """Warning page shown when user clicks a phishing link, button, or scans QR."""
    is_arabic = st.session_state.get("language") == "Arabic"
    _dir = "rtl" if is_arabic else "ltr"
    _align = "right" if is_arabic else "left"

    if is_arabic:
        title     = "⚠️ تنبيه! لقد وقعت في فخ التصيد الاحتيالي"
        subtitle  = "هذا بريد تصيد تدريبي — في الواقع الحقيقي كنت ستتعرض للاختراق"
        msg       = "الرابط أو الزر الذي ضغطت عليه كان يؤدي إلى موقع مزيف مصمم لسرقة بياناتك. في بيئة حقيقية، كان يمكن للمهاجم الحصول على معلوماتك الشخصية أو بيانات دخولك."
        tip_title = "💡 ماذا كان يجب أن تفعل؟"
        tips = [
            "تحقق من نطاق البريد الإلكتروني للمرسل قبل الضغط على أي رابط.",
            "لا تضغط على روابط مجهولة المصدر — تواصل مع قسم تقنية المعلومات للتحقق.",
            "تحقق من عنوان URL الفعلي قبل إدخال أي بيانات.",
            "الجهات الرسمية لا تطلب كلمة المرور عبر البريد الإلكتروني.",
        ]
        btn_label = "← العودة ومتابعة التدريب"
    else:
        title     = "⚠️ Alert! You clicked a phishing link"
        subtitle  = "This was a training phishing email — in a real scenario you would have been compromised"
        msg       = "The link or button you clicked would have led to a fake website designed to steal your credentials. In a real attack, the attacker could have obtained your personal information or login details."
        tip_title = "💡 What should you have done?"
        tips = [
            "Always verify the sender's email domain before clicking any link.",
            "Never click unknown links — contact IT to verify first.",
            "Check the actual URL before entering any information.",
            "Official organizations never ask for your password via email.",
        ]
        btn_label = "← Return and Continue Training"

    tips_html = "".join(f'<li style="margin-bottom:.5rem;color:#E2E8F0;">{tip}</li>' for tip in tips)

    st.markdown(f"""
<div dir="{_dir}" style="max-width:700px;margin:3rem auto;text-align:{_align};">
  <div style="background:linear-gradient(135deg,rgba(127,29,29,.95),rgba(69,10,10,.9));
              border:2px solid #EF4444;border-radius:20px;padding:2.5rem;
              box-shadow:0 0 60px rgba(239,68,68,.3);">
    <div style="font-size:3rem;text-align:center;margin-bottom:1rem;">🎣</div>
    <h2 style="color:#FCA5A5;font-size:1.6rem;font-weight:900;text-align:center;margin-bottom:.5rem;">{title}</h2>
    <p style="color:#FECACA;font-size:1rem;text-align:center;margin-bottom:1.5rem;font-style:italic;">{subtitle}</p>
    <div style="background:rgba(0,0,0,.3);border-radius:12px;padding:1.2rem;margin-bottom:1.5rem;">
      <p style="color:#E2E8F0;line-height:1.8;margin:0;">{msg}</p>
    </div>
    <div style="margin-bottom:1.5rem;">
      <p style="color:#FCD34D;font-weight:700;font-size:1rem;margin-bottom:.7rem;">{tip_title}</p>
      <ul style="padding-{('right' if is_arabic else 'left')}:1.2rem;margin:0;line-height:1.9;">
        {tips_html}
      </ul>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(btn_label, use_container_width=True, type="primary"):
            st.session_state["page"] = "learning"
            st.session_state.pop("phishing_caught", None)
            st.rerun()


def page_home():
    is_arabic      = st.session_state.get("language") == "Arabic"
    dir_attr       = 'rtl' if is_arabic else 'ltr'
    text_align     = 'right' if is_arabic else 'left'
    hero_grid_cols = '1fr 230px' if is_arabic else '230px 1fr'

    st.markdown(f"""<style>
#MainMenu,header,footer{{visibility:hidden;}}
.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}
.block-container{{max-width:1160px;padding-top:2rem;}}
.hero-card{{border:1px solid rgba(37,99,235,.55);border-radius:24px;padding:2.2rem 2.4rem;background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(8,47,73,.88));box-shadow:0 24px 70px rgba(0,0,0,.42);margin-bottom:1.3rem;}}
.hero-grid{{display:grid;grid-template-columns:{hero_grid_cols};gap:2rem;align-items:center;}}
.shield-orb{{width:180px;height:180px;border-radius:50%;margin:auto;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle,rgba(37,99,235,.45),rgba(2,6,23,.15) 65%);border:1px solid rgba(56,189,248,.35);box-shadow:0 0 45px rgba(37,99,235,.36);position:relative;overflow:visible;}}
.shield-orb::before{{content:"";position:absolute;width:215px;height:215px;border-radius:50%;border:1px dashed rgba(56,189,248,.34);}}
.hero-content{{text-align:center;}}
.hero-title{{font-size:3.4rem;font-weight:900;color:#F8FAFC;margin-bottom:.7rem;}}
.hero-tagline{{font-size:1.45rem;font-weight:800;color:#1EA7FF;margin-bottom:.9rem;}}
.hero-desc{{font-size:1rem;color:#DCEBFF;line-height:1.7;max-width:620px;margin:0 auto;}}
.features-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.4rem;direction:{dir_attr};}}
.feature-card{{border:1px solid rgba(37,99,235,.55);background:rgba(2,6,23,.60);border-radius:18px;padding:1.5rem 1rem;min-height:175px;text-align:center;cursor:pointer;transition:.25s ease;}}
.feature-card:hover{{transform:translateY(-6px);border-color:#1EA7FF;box-shadow:0 0 28px rgba(30,167,255,.22);}}
.feature-icon{{height:60px;margin-bottom:.8rem;display:flex;justify-content:center;align-items:center;}}
.feature-title{{font-size:1rem;font-weight:800;color:white;margin-bottom:.5rem;}}
.feature-text{{color:#BFD7F5;font-size:.9rem;line-height:1.55;}}
.form-section{{direction:{dir_attr};text-align:{text_align};margin-bottom:.5rem;}}
.form-title{{font-size:1.35rem;font-weight:900;color:white;margin-bottom:1rem;}}
.section-label{{font-weight:800;color:white;margin-bottom:.5rem;direction:{dir_attr};text-align:{text_align};}}
[data-testid="column"]{{direction:{dir_attr};}}
.stButton>button{{width:100%;min-height:48px;background:rgba(15,23,42,.78);color:#EAF4FF;border:1px solid rgba(37,99,235,.55);border-radius:12px;font-weight:800;direction:{dir_attr};}}
.stButton>button:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7);color:white;border-color:#1EA7FF !important;}}
.stButton>button:focus,.stButton>button:focus-visible,.stButton>button:focus:not(:hover){{outline:none !important;box-shadow:none !important;}}
.start-btn>button,.start-btn button[kind="primary"]{{min-height:56px !important;background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.65) !important;font-size:1.05rem !important;font-weight:900 !important;border-radius:14px !important;box-shadow:none !important;}}
.start-btn>button:hover,.start-btn button[kind="primary"]:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border-color:#1EA7FF !important;}}
div[data-baseweb="select"] *{{color:#EAF4FF!important;-webkit-text-fill-color:#EAF4FF!important;}}
div[data-baseweb="select"] > div{{background:rgba(15,23,42,.82)!important;border:1px solid rgba(37,99,235,.65)!important;border-radius:12px!important;}}
div[data-baseweb="popover"] *{{color:#EAF4FF!important;-webkit-text-fill-color:#EAF4FF!important;}}
.stSelectbox>div>div,.stTextInput>div>div>input{{background-color:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;border-radius:12px !important;min-height:48px;direction:{dir_attr};text-align:{text_align};}}
div[data-baseweb="select"] span{{color:white !important;-webkit-text-fill-color:white!important;}}
div[data-baseweb="single-value"]{{color:white !important;-webkit-text-fill-color:white!important;}}
div[data-baseweb="select"] [data-value]{{color:white !important;-webkit-text-fill-color:white!important;}}
.stSelectbox div[class*="ValueContainer"] span,
.stSelectbox div[class*="singleValue"],
.stSelectbox div[class*="placeholder"]{{color:white !important;-webkit-text-fill-color:white!important;opacity:1!important;}}
div[data-baseweb="select"] input{{color:white !important;caret-color:white;-webkit-text-fill-color:white!important;}}
div[data-baseweb="popover"] ul li{{text-align:{text_align} !important;direction:{dir_attr} !important;}}
.footer-bar{{margin-top:2rem;padding:1.5rem 0;border-top:1px solid rgba(37,99,235,.35);display:flex;justify-content:space-between;align-items:center;color:#7DD3FC;font-size:.95rem;direction:{dir_attr};}}
.footer-side{{display:flex;align-items:center;gap:.8rem;}}
button[kind="secondary"]{{width:100% !important;min-height:52px !important;border-radius:14px !important;font-weight:800 !important;font-size:.95rem !important;transition:.2s ease !important;background:rgba(2,6,23,.55) !important;border:2px solid rgba(37,99,235,.35) !important;color:#94A3B8 !important;}}
button[kind="secondary"]:hover{{background:rgba(11,79,168,.25) !important;border-color:#1EA7FF !important;color:#FFFFFF !important;}}
button[kind="primary"]{{width:100% !important;min-height:52px !important;border-radius:14px !important;font-weight:800 !important;font-size:.95rem !important;background:linear-gradient(135deg,#0B4FA8,#0284C7) !important;border:2px solid #1EA7FF !important;color:#FFFFFF !important;box-shadow:0 0 18px rgba(30,167,255,.3) !important;opacity:1 !important;}}
button[kind="primary"]:hover,button[kind="primary"]:focus{{background:linear-gradient(135deg,#0B4FA8,#0284C7) !important;border:2px solid #1EA7FF !important;color:#FFFFFF !important;}}
@media(max-width:950px){{.hero-grid{{grid-template-columns:1fr;}}.features-grid{{grid-template-columns:1fr;}}.footer-bar{{flex-direction:column;gap:1rem;text-align:center;}}}}
</style>""", unsafe_allow_html=True)

    SHIELD_MAIN_SVG = """<svg width="130" height="148" viewBox="0 0 130 148" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="lock_g" x1="38" y1="72" x2="92" y2="132" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#C8E6FF"/></linearGradient><filter id="sh_glow" x="-25%" y="-25%" width="150%" height="150%"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><path d="M65 4L124 24V72C124 108 98 136 65 144C32 136 6 108 6 72V24L65 4Z" fill="rgba(30,90,180,0.15)" stroke="#4A9EFF" stroke-width="3" filter="url(#sh_glow)"/><path d="M65 13L114 30V72C114 103 91 128 65 136C39 128 16 103 16 72V30L65 13Z" fill="none" stroke="rgba(120,190,255,0.3)" stroke-width="1.5"/><path d="M44 82V67C44 52 86 52 86 67V82" stroke="url(#lock_g)" stroke-width="10" stroke-linecap="round" fill="none"/><rect x="34" y="79" width="62" height="48" rx="9" fill="url(#lock_g)"/><circle cx="65" cy="100" r="8" fill="#1558A8"/><rect x="61.5" y="100" width="7" height="13" rx="2.5" fill="#1558A8"/></svg>"""
    BRAIN_SVG  = """<svg width="52" height="52" viewBox="0 0 52 52" fill="none"><path d="M26 8C20 8 17 12 15 20C12 20 9 22 9 26C9 30 12 32 15 32C14 37 17 40 26 44" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round" fill="none"/><path d="M26 8C32 8 35 12 37 20C40 20 43 22 43 26C43 30 40 32 37 32C38 37 35 40 26 44" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round" fill="none"/><line x1="26" y1="8" x2="26" y2="44" stroke="#1EA7FF" stroke-width="2"/><path d="M15 20C18 18 22 20 26 18C30 20 34 18 37 20" stroke="#1EA7FF" stroke-width="2" fill="none"/><path d="M15 32C18 30 22 32 26 30C30 32 34 30 37 32" stroke="#1EA7FF" stroke-width="2" fill="none"/></svg>"""
    TARGET_SVG = """<svg width="52" height="52" viewBox="0 0 52 52" fill="none"><circle cx="26" cy="28" r="18" stroke="#1EA7FF" stroke-width="2.5" fill="none"/><circle cx="26" cy="28" r="10" stroke="#1EA7FF" stroke-width="2.5" fill="none"/><circle cx="26" cy="28" r="3" fill="#1EA7FF"/><path d="M34 10L42 10L42 18" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round" fill="none"/><line x1="42" y1="10" x2="29" y2="23" stroke="#1EA7FF" stroke-width="2.5"/></svg>"""
    CHART_SVG  = """<svg width="52" height="52" viewBox="0 0 52 52" fill="none"><line x1="10" y1="44" x2="10" y2="8" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round"/><line x1="10" y1="44" x2="46" y2="44" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round"/><rect x="15" y="32" width="7" height="12" rx="2" fill="#1EA7FF" opacity=".85"/><rect x="26" y="22" width="7" height="22" rx="2" fill="#1EA7FF" opacity=".85"/><rect x="37" y="12" width="7" height="32" rx="2" fill="#1EA7FF" opacity=".85"/></svg>"""
    SHIELD_SVG = """<svg width="52" height="56" viewBox="0 0 52 56" fill="none"><path d="M26 4L46 12V28C46 39 36 50 26 52C16 50 6 39 6 28V12L26 4Z" stroke="#1EA7FF" stroke-width="2.5" fill="none"/><path d="M18 28L23 33L34 22" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round" fill="none"/></svg>"""
    SHFOOT     = """<svg width="42" height="48" viewBox="0 0 42 48" fill="none"><path d="M21 3L39 10V24C39 34 31 43 21 46C11 43 3 34 3 24V10L21 3Z" stroke="#1EA7FF" stroke-width="2.5" fill="none"/></svg>"""
    ECG_SVG    = """<svg width="80" height="28" viewBox="0 0 80 28" fill="none"><polyline points="0,14 15,14 20,4 25,24 30,4 35,20 40,14 80,14" stroke="#1EA7FF" stroke-width="2.5" stroke-linecap="round" fill="none"/></svg>"""

    nav_login    = t("Login","تسجيل الدخول")
    nav_register = t("Register","إنشاء حساب")
    nav_brand    = t("AI Phishing Awareness","التوعية بالتصيد الإلكتروني")
    user_name    = (st.session_state.get("user_name") or "")
    shield_small = SHIELD_SVG.replace('width="52"','width="20"').replace('height="56"','height="22"')
    flex_dir     = "row-reverse" if is_arabic else "row"

    if user_name:
        st.markdown(f"""
<div style="background:rgba(11,46,104,0.55);border:1px solid rgba(37,99,235,.4);
     border-radius:14px;padding:8px 20px;margin-bottom:1.2rem;
     display:flex;align-items:center;justify-content:space-between;
     flex-direction:{flex_dir};min-height:52px;">
  <div style="display:flex;align-items:center;gap:8px;flex-direction:{"row-reverse" if is_arabic else "row"};">
    {shield_small}
    <span style="font-size:15px;font-weight:800;color:#F8FAFC;white-space:nowrap;">{nav_brand}</span>
  </div>
  <div style="display:inline-flex;align-items:center;gap:6px;
      background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.4);
      border-radius:20px;padding:5px 10px 5px 6px;">
    <div style="width:24px;height:24px;border-radius:50%;
        background:linear-gradient(135deg,#0B4FA8,#0284C7);
        display:flex;align-items:center;justify-content:center;font-size:11px;">👤</div>
    <span style="font-size:12px;color:#7DD3FC;font-weight:700;
        white-space:nowrap;max-width:140px;overflow:hidden;text-overflow:ellipsis;">
      {html_lib.escape(user_name)}</span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<style>
.nb-btn {{height:34px;padding:0 16px;border-radius:9px;font-size:12px;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;white-space:nowrap;text-decoration:none;}}
.nb-btn-ghost {{background:rgba(15,23,42,.88);color:#EAF4FF !important;border:1px solid rgba(37,99,235,.5);}}
.nb-btn-ghost:hover {{background:rgba(37,99,235,.25);border-color:#1EA7FF;color:#fff !important;}}
.nb-btn-solid {{background:linear-gradient(90deg,#0B4FA8,#0284C7);color:white !important;border:none;}}
.nb-btn-solid:hover {{background:linear-gradient(90deg,#1560C0,#0396E0);}}
</style>
<div style="background:rgba(11,46,104,0.55);border:1px solid rgba(37,99,235,.4);
     border-radius:14px;padding:8px 20px;margin-bottom:1.2rem;
     display:flex;align-items:center;justify-content:space-between;
     flex-direction:{flex_dir};min-height:52px;">
  <div style="display:flex;align-items:center;gap:8px;flex-direction:{"row-reverse" if is_arabic else "row"};">
    {shield_small}
    <span style="font-size:15px;font-weight:800;color:#F8FAFC;white-space:nowrap;">{nav_brand}</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <a href="?nav=login&lang={st.session_state.get('language','English')}" class="nb-btn nb-btn-ghost">{nav_login}</a>
    <a href="?nav=register&lang={st.session_state.get('language','English')}" class="nb-btn nb-btn-solid">{nav_register}</a>
  </div>
</div>""", unsafe_allow_html=True)

    title   = t("AI Phishing Awareness","التوعية بالتصيد الإلكتروني بالذكاء الاصطناعي")
    tagline = t("Smart, Personalised, Protective.","ذكي، مخصص، وقائي")
    desc    = t("AI-powered training and assessment to help healthcare employees recognise and avoid phishing threats.",
                "تدريب وتقييم مدعوم بالذكاء الاصطناعي لمساعدة الموظفين الصحيين على التعرف على تهديدات التصيد الإلكتروني وتجنبها")
    time_badge = t("⏱️ ~15 minutes","⏱️ ~١٥ دقيقة")
    sh = f'<div class="shield-orb">{SHIELD_MAIN_SVG}</div>'
    co = f'''<div class="hero-content">
      <div class="hero-title">{title}</div>
      <div class="hero-tagline">{tagline}</div>
      <div class="hero-desc">{desc}</div>
      <div style="display:inline-flex;align-items:center;gap:6px;margin-top:.8rem;
                  background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.35);
                  border-radius:8px;padding:5px 14px;">
        <span style="font-size:.9rem;color:#7DD3FC;font-weight:600;">{time_badge}</span>
      </div>
    </div>'''
    gi = co+sh if is_arabic else sh+co
    st.markdown(f'<div class="hero-card"><div class="hero-grid">{gi}</div></div>', unsafe_allow_html=True)

    cards = [
        (BRAIN_SVG, t("AI-Powered Learning","تعلم بالذكاء الاصطناعي"), t("Personalised content adapted to your role.","محتوى تعليمي مخصص حسب دورك الوظيفي")),
        (TARGET_SVG,t("Smart Assessment","تقييم ذكي"),                 t("Short, focused assessments to test your awareness.","تقييمات قصيرة ومركزة لاختبار وعيك")),
        (CHART_SVG, t("Personalised Feedback","تغذية راجعة مخصصة"),   t("Detailed results with insights and recommendations.","نتائج مفصلة تتضمن ملاحظات وتوصيات مخصصة")),
        (SHIELD_SVG,t("Stronger Together","معًا أكثر أمانًا"),         t("Building a secure healthcare environment for everyone.","بناء بيئة صحية آمنة للجميع")),
    ]
    st.markdown('<div class="features-grid">'+"".join(f'<div class="feature-card"><div class="feature-icon">{i}</div><div class="feature-title">{tt}</div><div class="feature-text">{tx}</div></div>' for i,tt,tx in cards)+'</div>', unsafe_allow_html=True)

    form_col, panel_col = st.columns([3, 1], gap="large")

    with form_col:
        form_title_txt = t("Let's personalise your experience","لنخصص تجربتك")
        st.markdown(f'<div class="form-section"><div class="form-title">👤 {form_title_txt}</div></div>', unsafe_allow_html=True)

        def step_label(n, txt):
            return f'''<div style="font-size:.85rem;color:#94A3B8;margin-bottom:.5rem;
                        display:flex;align-items:center;gap:6px;direction:{dir_attr};">
              <span style="display:inline-flex;align-items:center;justify-content:center;
                           width:18px;height:18px;border-radius:50%;
                           background:rgba(37,99,235,.5);color:#7DD3FC;
                           font-size:10px;font-weight:800;">{n}</span>
              {txt}
            </div>'''

        st.markdown(step_label("1", t("Select your preferred language","اختر اللغة المفضلة")), unsafe_allow_html=True)
        cur_lang     = (st.session_state.get("language") or "")
        lang_chosen0 = st.session_state.get("lang_explicitly_chosen", False)
        col1,col2 = st.columns(2)
        with col1:
            st.button("English", key="english", on_click=set_language, args=("English",),
                       use_container_width=True, type=("primary" if lang_chosen0 and cur_lang == "English" else "secondary"))
        with col2:
            st.button("العربية", key="arabic",  on_click=set_language, args=("Arabic",),
                       use_container_width=True, type=("primary" if lang_chosen0 and cur_lang == "Arabic" else "secondary"))

        st.markdown(step_label("2", t("Select your role","اختر دورك الوظيفي")), unsafe_allow_html=True)
        opts = [t("Choose your role","اختر دورك الوظيفي"),t("Clinical","سريري"),t("Admin / Management","إداري / إدارة"),t("IT / Informatics","تقنية المعلومات / المعلوماتية"),t("Other","أخرى")]
        sel  = st.selectbox("role",opts,index=0,label_visibility="collapsed")
        other_role = ""
        if sel==opts[-1]: other_role=st.text_input(t("Please specify your role","يرجى كتابة دورك الوظيفي"),placeholder=t("Type your role here","اكتب دورك الوظيفي هنا"))

        st.markdown(step_label("3", t("Select difficulty level","اختر مستوى الصعوبة")), unsafe_allow_html=True)

    with panel_col:
        ph_label  = t("Learning phase","مرحلة التعلم")
        as_label  = t("Assessment","الاختبار")
        rep_label = t("Performance report","تقرير الأداء")
        exp_title = t("WHAT TO EXPECT","ماذا تتوقع")
        diff_title= t("DIFFICULTY","الصعوبة")
        beg_lbl   = t("Beginner","مبتدئ")
        mid_lbl   = t("Intermediate","متوسط")
        adv_lbl   = t("Advanced","متقدم")
        small_brain  = BRAIN_SVG.replace('width="52"','width="18"').replace('height="52"','height="18"')
        small_target = TARGET_SVG.replace('width="52"','width="18"').replace('height="52"','height="18"')
        small_chart  = CHART_SVG.replace('width="52"','width="18"').replace('height="52"','height="18"')
        st.markdown(f"""
<div style="background:rgba(8,47,73,.2);border:1px solid rgba(37,99,235,.25);border-radius:14px;padding:1.2rem 1rem;margin-top:1rem;direction:{dir_attr};">
  <div style="font-size:.75rem;font-weight:800;color:#7DD3FC;letter-spacing:.06em;margin-bottom:14px;">{exp_title}</div>
  <div style="display:flex;flex-direction:column;gap:9px;margin-bottom:16px;">
    <div style="display:flex;align-items:center;gap:9px;padding:9px 10px;background:rgba(15,23,42,.7);border:1px solid rgba(37,99,235,.3);border-radius:9px;">{small_brain}<span style="font-size:.82rem;font-weight:700;color:#E2E8F0;">{ph_label}</span></div>
    <div style="display:flex;align-items:center;gap:9px;padding:9px 10px;background:rgba(15,23,42,.7);border:1px solid rgba(37,99,235,.3);border-radius:9px;">{small_target}<span style="font-size:.82rem;font-weight:700;color:#E2E8F0;">{as_label}</span></div>
    <div style="display:flex;align-items:center;gap:9px;padding:9px 10px;background:rgba(15,23,42,.7);border:1px solid rgba(37,99,235,.3);border-radius:9px;">{small_chart}<span style="font-size:.82rem;font-weight:700;color:#E2E8F0;">{rep_label}</span></div>
  </div>
  <div style="border-top:1px solid rgba(37,99,235,.2);padding-top:12px;">
    <div style="font-size:.75rem;font-weight:800;color:#7DD3FC;letter-spacing:.05em;margin-bottom:8px;direction:{dir_attr};text-align:{text_align};">{diff_title}</div>
    <div style="display:flex;flex-direction:column;gap:5px;direction:{dir_attr};text-align:{text_align};">
      <div style="font-size:.8rem;color:#94A3B8;">🟢 {beg_lbl}</div>
      <div style="font-size:.8rem;color:#94A3B8;">🟡 {mid_lbl}</div>
      <div style="font-size:.8rem;color:#94A3B8;">🔴 {adv_lbl}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with form_col:
        current_diff  = st.session_state.get("difficulty","medium")
        if st.session_state.get("language","English") == "Arabic":
            ordered = [("easy","🟢  مبتدئ"),("medium","🟡  متوسط"),("hard","🔴  متقدم")]
        else:
            ordered = [("easy","🟢  Beginner"),("medium","🟡  Intermediate"),("hard","🔴  Advanced")]

        diff_cols = st.columns(3)
        if st.session_state.get("language","English") == "Arabic":
            ordered_display = list(reversed(ordered))
        else:
            ordered_display = ordered

        for i,(dk,lbl) in enumerate(ordered_display):
            with diff_cols[i]:
                is_sel  = current_diff == dk and st.session_state.get("diff_explicitly_chosen", False)
                if st.button(lbl, key=f"diff_{dk}", use_container_width=True,
                             type=("primary" if is_sel else "secondary")):
                    st.session_state["difficulty"] = dk
                    st.session_state["diff_explicitly_chosen"] = True
                    st.rerun()

        if st.query_params.get("mode") == "researcher":
            st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.75rem;font-weight:800;color:#F59E0B;letter-spacing:.06em;margin-bottom:.5rem;direction:{dir_attr};">🔬 RESEARCHER MODE — AI Provider</div>', unsafe_allow_html=True)
            provider_options = {
                "groq":      "🟠 Groq  (LLaMA 3.3-70b) — Baseline v3",
                "openai":    "🟢 ChatGPT  (GPT-4o) — Most used globally",
                "anthropic": "🟣 Claude  (claude-sonnet-4-6) — Best writing quality",
                "gemini":    "🔵 Gemini  (1.5 Pro) — Fastest growing",
            }
            cur_provider = st.session_state.get("ai_provider", "openai")
            prov_cols = st.columns(2)
            prov_items = list(provider_options.items())
            for i, (pk, plbl) in enumerate(prov_items):
                with prov_cols[i % 2]:
                    is_psel = cur_provider == pk
                    if st.button(plbl, key=f"prov_{pk}", use_container_width=True,
                                 type=("primary" if is_psel else "secondary")):
                        set_active_provider(pk)
                        st.rerun()
            st.markdown(f'<div style="font-size:.72rem;color:#64748B;margin-top:.3rem;direction:{dir_attr};">Active: <b style="color:#F59E0B;">{provider_options.get(cur_provider,"")}</b></div>', unsafe_allow_html=True)

        st.markdown('<div class="start-btn" style="margin-top:.8rem;">',unsafe_allow_html=True)
        if st.button(t("Start Personalised Training","ابدأ التدريب المخصص"),key="start_training", use_container_width=True, type="primary"):
            fr = other_role.strip() if sel==opts[-1] else sel
            lang_chosen = st.session_state.get("lang_explicitly_chosen", False)
            diff_chosen = st.session_state.get("diff_explicitly_chosen", False)
            user_logged = (st.session_state.get("user_name") or "").strip() != ""

            if not user_logged:
                st.warning(t("⚠️ Please login or register first using the button at the top.",
                             "⚠️ يرجى تسجيل الدخول أو إنشاء حساب أولاً عبر الزر في الأعلى"))
            elif not lang_chosen:
                st.warning(t("⚠️ Please select your preferred language first.",
                             "⚠️ يرجى اختيار اللغة المفضلة أولاً"))
            elif fr==opts[0]:
                st.warning(t("⚠️ Please select your role.",
                             "⚠️ يرجى اختيار دورك الوظيفي"))
            elif not diff_chosen:
                st.warning(t("⚠️ Please select a difficulty level.",
                             "⚠️ يرجى اختيار مستوى الصعوبة"))
            else:
                if "scenario_order" in st.session_state: del st.session_state["scenario_order"]
                # EN: fresh "Start Training" click → fresh scenario/recipient
                # order too, so re-running training mid-session also gets new
                # variety (not just a full logout/retake).
                # AR: ضغطة جديدة على "بدء التدريب" → ترتيب سيناريو/مستلم جديد
                # كذلك، حتى إعادة تشغيل التدريب بنفس الجلسة يطلع له تنوع جديد.
                for k in list(st.session_state.keys()):
                    if k.startswith(("scenario_order_", "recipient_order_", "category_order_", "used_topics_")):
                        st.session_state.pop(k, None)
                go_to_learning(fr); st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

    ft = t("Together, let's build a stronger, phishing-resistant healthcare environment.","معًا نبني بيئة صحية أكثر مقاومة للتصيد الإلكتروني")
    fs = t("Stay aware, Stay secure, Save lives.","كن واعيًا، ابق آمنًا، وساهم في حماية الأرواح")
    if is_arabic:
        f1=f'<div class="footer-side" style="direction:ltr;"><span style="direction:rtl;">{fs}</span>&nbsp;{ECG_SVG}</div>'
        f2=f'<div class="footer-side" style="direction:ltr;justify-content:flex-end;"><span style="direction:rtl;">{ft}</span>{SHFOOT}</div>'
    else:
        f1=f'<div class="footer-side">{SHFOOT}<span>{ft}</span></div>'
        f2=f'<div class="footer-side">{ECG_SVG}&nbsp;{fs}</div>'
    st.markdown(f'<div class="footer-bar">{f1}{f2}</div>',unsafe_allow_html=True)


def page_learning():
    is_arabic  = st.session_state["language"]=="Arabic"
    dir_attr   = 'rtl' if is_arabic else 'ltr'
    text_align = 'right' if is_arabic else 'left'
    TOTAL      = 6
    idx       = st.session_state["example_index"]

    # Inject JS listener for phishing click events from email buttons
    import streamlit.components.v1 as _comp
    _comp.html("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'phishing_click') {
            // Send to Streamlit via query param change to trigger rerun
            const url = new URL(window.parent.location.href);
            url.searchParams.set('phishing_click', '1');
            window.parent.history.replaceState({}, '', url.toString());
        }
    });
    </script>
    """, height=0, width=0)

    if st.query_params.get("phishing_click") == "1":
        st.query_params.clear()
        st.session_state["page"] = "phishing_caught"
        st.rerun()

    if st.session_state.get("cache_version",0) < 20:
        st.session_state["emails"]={}
        st.session_state.pop("assess_emails", None)
        st.session_state["cache_version"]=20

    st.markdown(f"""<style>
#MainMenu,header,footer{{visibility:hidden;}}
.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}
.block-container{{max-width:1200px;padding-top:2rem;}}
.tutor-panel{{background:rgba(2,6,23,.7);border:1px solid rgba(37,99,235,.45);border-radius:16px;padding:1.4rem 1.5rem;direction:{dir_attr};text-align:{text_align};}}
.tutor-section{{font-size:1rem;font-weight:800;color:#F1F5F9;margin:1rem 0 .4rem;direction:{dir_attr};text-align:{text_align};}}
.tutor-text{{color:#94A3B8;font-size:.92rem;line-height:1.65;direction:{dir_attr};text-align:{text_align};}}
.tip-box{{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.35);border-radius:10px;padding:.8rem 1rem;color:#6EE7B7;font-size:.9rem;line-height:1.6;margin-top:.8rem;direction:{dir_attr};text-align:{text_align};}}
.stButton>button{{min-height:52px;background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;color:white !important;border:none !important;font-weight:800 !important;border-radius:12px !important;font-size:1rem !important;}}
</style>""", unsafe_allow_html=True)

    if idx not in st.session_state["emails"]:
        with st.spinner(t("🤖 Generating phishing example...","🤖 جارٍ توليد مثال التصيد...")):
            st.session_state["emails"][idx] = generate_email(st.session_state["role"], idx, st.session_state["language"], st.session_state.get("difficulty", "medium"))
            st.rerun()

    email = st.session_state["emails"].get(idx,{})
    pct   = int((idx/TOTAL)*100)

    st.markdown(f"""
<div style="margin-bottom:1.5rem;direction:{dir_attr};">
  <div style="font-size:2.2rem;font-weight:900;color:#F8FAFC;margin-bottom:.4rem;text-align:{text_align};">
    {t("AI Tutor-Guided Learning Phase","مرحلة التعلم بتوجيه الذكاء الاصطناعي")}
  </div>
  <div style="width:100%;height:6px;background:rgba(37,99,235,.25);border-radius:99px;margin:.8rem 0;">
    <div style="height:6px;border-radius:99px;background:linear-gradient(90deg,#1EA7FF,#2563EB);width:{pct}%;transition:width .4s ease;"></div>
  </div>
  <div style="color:#7DD3FC;font-size:.95rem;font-weight:600;">
    {t(f"Example {idx+1} of {TOTAL}",f"مثال {idx+1} من {TOTAL}")}
  </div>
</div>""", unsafe_allow_html=True)

    if "error" in email:
        st.error(f"**{t('Error','خطأ')}:** " + _safe_error_text(email['error'], st.session_state['language']))
        if st.button(t("🔄 Try Again","🔄 حاول مرة أخرى"),key="retry_btn"):
            del st.session_state["emails"][idx]; st.rerun()
        return

    if is_arabic:
        col_tutor, col_email = st.columns([1,1.1],gap="large")
    else:
        col_email, col_tutor = st.columns([1.1,1],gap="large")

    with col_email:
        render_email_window(email, is_arabic, show_badges=True)

    with col_tutor:
        indicators    = email.get("indicators",[])
        indicators_html = ""
        for pos, ind in enumerate(indicators, 1):
            # Backward compatibility: providers/fallbacks may occasionally
            # return a plain string instead of the expected dictionary.
            if isinstance(ind, dict):
                ind_number = ind.get("number") or pos
                ind_title = ind.get("title") or ind.get("name") or ""
                ind_description = ind.get("description") or ind.get("detail") or ""
            else:
                ind_number = pos
                ind_title = str(ind or "")
                ind_description = ""
            row_dir = 'rtl' if is_arabic else 'ltr'
            pad     = 'padding-right:2rem;' if is_arabic else 'padding-left:2rem;'
            ta2     = 'right' if is_arabic else 'left'
            indicators_html += f"""
<div style="margin-bottom:1rem;">
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.3rem;direction:{row_dir};">
    <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;background:#DC2626;color:white;font-size:.75rem;font-weight:800;flex-shrink:0;">{ind_number}</span>
    <span style="font-weight:700;color:#E2E8F0;font-size:.95rem;">{html_lib.escape(str(ind_title))}</span>
  </div>
  <div style="color:#94A3B8;font-size:.9rem;line-height:1.65;{pad};direction:{row_dir};text-align:{ta2};">{html_lib.escape(str(ind_description))}</div>
</div>"""

        st.markdown(f"""
<div class="tutor-panel">
  <div style="font-size:1.3rem;font-weight:900;color:#F8FAFC;margin-bottom:.2rem;">🎯 {t("AI Tutor Analysis","تحليل المعلم الذكي")}</div>
  <div style="color:#64748B;font-size:.85rem;margin-bottom:1.2rem;">{t("AI-guided phishing awareness","شرح توعوي بالتصيد")}</div>
  <div class="tutor-section">{t("What is suspicious?","ما هو المشبوه؟")}</div>
  {indicators_html}
  <div class="tutor-section">{t("Why is it risky?","لماذا هو خطير؟")}</div>
  <div class="tutor-text">{(email.get("why_risky") or "")}</div>
  <div class="tutor-section">💡 {t("Learning Tip","نصيحة تعليمية")}</div>
  <div class="tip-box">{(email.get("learning_tip") or "")}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>',unsafe_allow_html=True)
    bc,_ = st.columns([1,3])
    with bc:
        if idx<TOTAL-1:
            if st.button(t("Next Example →","← المثال التالي"),key="next_btn"):
                st.session_state["example_index"]+=1; st.rerun()
        else:
            if st.button(t("Complete Learning Phase →","← إتمام مرحلة التعلم"),key="complete_btn"):
                st.session_state["page"]="complete"; st.rerun()


def page_complete():
    is_arabic=st.session_state["language"]=="Arabic"; da='rtl' if is_arabic else 'ltr'
    def tc(e,a): return a if is_arabic else e
    st.markdown("""<style>#MainMenu,header,footer{visibility:hidden;}.stApp{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}.block-container{max-width:700px;padding-top:4rem;}.stButton>button{min-height:52px;background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;color:white !important;border:none !important;font-weight:800 !important;border-radius:12px !important;font-size:1rem !important;width:100%;}</style>""",unsafe_allow_html=True)
    msg=tc("You have completed all 6 learning examples.\nYou are now ready to test your phishing awareness skills.","لقد أكملت جميع الأمثلة التعليمية الـ 6.\nأنت الآن جاهز لاختبار مستوى وعيك بالتصيد الإلكتروني.")
    st.markdown(f'<div style="text-align:center;padding:3rem 2rem;border:1px solid rgba(37,99,235,.45);border-radius:24px;background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(8,47,73,.88));direction:{da};"><div style="font-size:4rem;margin-bottom:1rem;">🎓</div><div style="font-size:2rem;font-weight:900;color:#F8FAFC;margin-bottom:1rem;">{tc("Great job","ممتاز")}</div><div style="font-size:1.05rem;color:#DCEBFF;line-height:1.8;margin-bottom:2rem;white-space:pre-line;">{msg}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>',unsafe_allow_html=True)
    if st.button(tc("Start Assessment →","← ابدأ الاختبار"),key="go_assessment"):
        if "assess_scenario_order" in st.session_state: del st.session_state["assess_scenario_order"]
        for k in list(st.session_state.keys()):
            if k.startswith(("assess_recipient_order_", "assess_scenario_order_", "assess_category_order_", "used_topics_")):
                st.session_state.pop(k, None)
        st.session_state.update({"page":"assessment","assess_index":0,"assess_emails":{},"assess_answers":{}})
        st.rerun()


def page_assessment():
    is_arabic=st.session_state["language"]=="Arabic"; da='rtl' if is_arabic else 'ltr'
    TOTAL=10; idx=st.session_state.get("assess_index",0)
    def ta(e,a): return a if is_arabic else e

    if "assess_pattern" not in st.session_state:
        p=[True]*5+[False]*5; random.shuffle(p); st.session_state["assess_pattern"]=p
    pattern=st.session_state["assess_pattern"]

    st.markdown(f"""<style>
#MainMenu,header,footer{{visibility:hidden;}}
.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}
.block-container{{max-width:960px;padding-top:2rem;}}
.stButton>button{{min-height:52px;font-weight:800 !important;border-radius:12px !important;font-size:1rem !important;background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;width:100% !important;transition:.2s ease;}}
.stButton>button:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border-color:#1EA7FF !important;}}
</style>""",unsafe_allow_html=True)

    if idx not in st.session_state["assess_emails"]:
        with st.spinner(ta("🤖 Generating scenario...","🤖 جارٍ توليد السيناريو...")):
            st.session_state["assess_emails"][idx]=generate_assess_email(st.session_state["role"], idx, pattern[idx], st.session_state["language"], st.session_state.get("difficulty", "medium"))
            st.rerun()

    email=st.session_state["assess_emails"].get(idx,{})
    answered_count = len(st.session_state.get("assess_answers", {}))
    pct=int((answered_count/TOTAL)*100)
    st.markdown(f"""
<div style="margin-bottom:1.5rem;direction:{da};">
  <div style="font-size:2rem;font-weight:900;color:#F8FAFC;margin-bottom:.4rem;">{ta("AI-Generated Assessment","مرحلة الاختبار")}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin:.8rem 0 .3rem;">
    <div style="color:#7DD3FC;font-size:.95rem;font-weight:600;">{ta(f"Question {idx+1} of {TOTAL}",f"السؤال {idx+1} من {TOTAL}")}</div>
    <div style="color:#F59E0B;font-size:.9rem;font-weight:700;">{pct}%</div>
  </div>
  <div style="width:100%;height:8px;background:rgba(37,99,235,.2);border-radius:99px;overflow:hidden;">
    <div style="height:8px;border-radius:99px;background:linear-gradient(90deg,#F59E0B,#EF4444);width:{pct}%;transition:width .5s ease;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:.4rem;">
    {"".join(f'<div style="width:{100//TOTAL}%;height:4px;border-radius:2px;background:{"#F59E0B" if i < answered_count else "rgba(255,255,255,.1)"};margin:0 1px;"></div>' for i in range(TOTAL))}
  </div>
</div>""",unsafe_allow_html=True)

    if "error" in email:
        st.error(f"{t('Error','خطأ')}: " + _safe_error_text(email['error'], st.session_state['language']))
        if st.button(ta("🔄 Try Again","🔄 حاول مرة أخرى"),key="assess_retry"):
            del st.session_state["assess_emails"][idx]; st.rerun()
        return

    if is_arabic: col_action,col_email=st.columns([1,1.2],gap="large")
    else:         col_email,col_action=st.columns([1.2,1],gap="large")

    with col_email: render_email_window(email,is_arabic,show_badges=False)

    with col_action:
        q=ta("Is this email phishing or legitimate?","هل هذه الرسالة تصيد إلكتروني أم شرعية؟")
        st.markdown(f'<div style="background:rgba(2,6,23,.7);border:1px solid rgba(37,99,235,.45);border-radius:16px;padding:1.5rem;text-align:center;direction:{da};margin-bottom:1rem;"><div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;">{q}</div></div>',unsafe_allow_html=True)

        answered=idx in st.session_state["assess_answers"]
        if not answered:
            c1,c2=st.columns(2)
            with c1:
                if st.button(f"🚨 {ta('Phishing','تصيد إلكتروني')}",key=f"ph_{idx}", use_container_width=True):
                    st.session_state["assess_answers"][idx]="phishing"; st.rerun()
            with c2:
                if st.button(f"✅ {ta('Legitimate','شرعية')}",key=f"lg_{idx}", use_container_width=True):
                    st.session_state["assess_answers"][idx]="legitimate"; st.rerun()
        else:
            ua=st.session_state["assess_answers"][idx]; ca2="phishing" if pattern[idx] else "legitimate"; ok=ua==ca2
            c="#6EE7B7" if ok else "#FCA5A5"; bg="rgba(16,185,129,.15)" if ok else "rgba(239,68,68,.15)"; br="rgba(16,185,129,.5)" if ok else "rgba(239,68,68,.5)"
            ic="✅" if ok else "❌"; lb=ta("Correct!","إجابة صحيحة!") if ok else ta("Incorrect!","إجابة خاطئة!")
            st.markdown(f'<div style="background:{bg};border:2px solid {br};border-radius:12px;padding:1rem;text-align:center;color:{c};font-weight:800;font-size:1.1rem;margin-bottom:1rem;">{ic} {lb}</div>',unsafe_allow_html=True)
            if idx<TOTAL-1:
                if st.button(ta("Next Question →","← السؤال التالي"),key=f"na_{idx}", use_container_width=True):
                    st.session_state["assess_index"]+=1; st.rerun()
            else:
                if st.button(ta("View Results →","← عرض النتائج"),key="vr", use_container_width=True):
                    st.session_state["page"]="results"; st.rerun()


def page_results():
    is_arabic=st.session_state["language"]=="Arabic"; da='rtl' if is_arabic else 'ltr'; TOTAL=10
    def tr(e,a): return a if is_arabic else e
    st.markdown("""<style>#MainMenu,header,footer{visibility:hidden;}.stApp{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}.block-container{max-width:900px;padding-top:2rem;}.stButton>button{min-height:52px;background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;color:white !important;border:none !important;font-weight:800 !important;border-radius:12px !important;}div[style*="direction:rtl"]{text-align:right;}</style>""",unsafe_allow_html=True)
    answers=st.session_state.get("assess_answers",{}); pattern=st.session_state.get("assess_pattern",[True]*5+[False]*5); emails=st.session_state.get("assess_emails",{})
    score=sum(1 for i in range(TOTAL) if answers.get(i)==("phishing" if pattern[i] else "legitimate"))
    pct=int((score/TOTAL)*100)
    sc="#10B981" if pct>=80 else "#F59E0B" if pct>=60 else "#EF4444"
    sl=tr("Excellent 🎉","ممتاز 🎉") if pct>=80 else tr("Good job 👍","جيد 👍") if pct>=60 else tr("Keep practicing 💪","استمر في التدريب 💪")
    st.markdown(f'<div style="text-align:center;padding:2rem;border:1px solid rgba(37,99,235,.45);border-radius:24px;background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(8,47,73,.88));margin-bottom:2rem;direction:{da};"><div style="font-size:1.5rem;font-weight:900;color:#F8FAFC;margin-bottom:1rem;">{tr("Your Results","نتائجك")}</div><div style="font-size:4rem;font-weight:900;color:{sc};">{score}/{TOTAL}</div><div style="font-size:1.2rem;color:{sc};font-weight:700;">{sl}</div><div style="color:#94A3B8;margin-top:.5rem;">{tr(f"You answered {score} of {TOTAL} correctly ({pct}%)",f"أجبت على {score} من {TOTAL} بشكل صحيح ({pct}٪)")}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.3rem;font-weight:900;color:#F8FAFC;margin-bottom:1rem;direction:{da};">📋 {tr("Review","مراجعة الإجابات")}</div>',unsafe_allow_html=True)
    for i in range(TOTAL):
        em=emails.get(i,{})
        if not em or "error" in em: continue
        ua=answers.get(i,""); ca2="phishing" if pattern[i] else "legitimate"; ok=ua==ca2
        bc2="rgba(16,185,129,.5)" if ok else "rgba(239,68,68,.5)"; bg2="rgba(16,185,129,.05)" if ok else "rgba(239,68,68,.05)"
        ri="✅" if ok else "❌"; tl=tr("Phishing","تصيد") if pattern[i] else tr("Legitimate","شرعية"); ic="🚨" if pattern[i] else "✅"
        exp=re.sub(r'<[^>]+>','',(em.get("explanation") or ""))
        # FIX: bidi issue — Arabic explanations often embed Latin/domain
        # substrings (e.g. "hosp1tal-clinic.org") in the middle of a
        # sentence, sometimes inside parentheses with Arabic words on
        # both sides. The browser's bidi algorithm can let that embedded
        # LTR run "leak" and flip the visual order of the surrounding
        # Arabic words, especially at a line wrap. Wrapping each such run
        # in a <bdi dir="ltr"> tag creates a true isolated bidi run, which
        # is the standards-correct fix and keeps the rest of the Arabic
        # sentence in correct right-to-left reading order regardless of
        # where the line wraps.
        if is_arabic:
            exp = re.sub(
                r'[A-Za-z][A-Za-z0-9\.\-_/:@]*[A-Za-z0-9]',
                lambda m: f'<bdi dir="ltr">{m.group(0)}</bdi>',
                exp
            )
        st.markdown(f'<div style="border:1px solid {bc2};border-radius:14px;padding:1.2rem 1.5rem;background:{bg2};margin-bottom:1rem;direction:{da};"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;flex-wrap:wrap;gap:.5rem;"><span style="font-weight:800;color:#E2E8F0;">{ri} {tr(f"Q{i+1}",f"س{i+1}")} — {html_lib.escape((em.get("subject") or ""))}</span><span style="background:{"rgba(239,68,68,.2)" if pattern[i] else "rgba(16,185,129,.2)"};color:{"#FCA5A5" if pattern[i] else "#6EE7B7"};padding:.2rem .8rem;border-radius:99px;font-size:.85rem;font-weight:700;">{ic} {tl}</span></div><div dir="{da}" style="color:#94A3B8;font-size:.9rem;line-height:1.6;direction:{da};text-align:{"right" if is_arabic else "left"};unicode-bidi:embed;">{exp}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    if st.button(tr("Go to Report →","← الانتقال للتقرير"),key="go_report"):
        st.session_state["page"]="report"; st.rerun()


def page_report():
    is_arabic=st.session_state["language"]=="Arabic"; da='rtl' if is_arabic else 'ltr'; TOTAL=10
    def tp(e,a): return a if is_arabic else e
    st.markdown(f"""<style>#MainMenu,header,footer{{visibility:hidden;}}.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}.block-container{{max-width:900px;padding-top:2rem;}}.stButton>button{{min-height:52px !important;font-weight:800 !important;border-radius:12px !important;background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;width:100% !important;}}.stButton>button:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border-color:#1EA7FF !important;}}div[style*="direction:rtl"]{{text-align:right;}}</style>""",unsafe_allow_html=True)
    answers=st.session_state.get("assess_answers",{}); pattern=st.session_state.get("assess_pattern",[True]*5+[False]*5)
    role=(st.session_state.get("role") or ""); lang=st.session_state.get("language","English")
    score=pc=lc=0; pt=sum(1 for p in pattern if p); lt=TOTAL-pt
    for i in range(TOTAL):
        ca2="phishing" if pattern[i] else "legitimate"
        if answers.get(i)==ca2:
            score+=1
            if pattern[i]: pc+=1
            else: lc+=1
    pct=int((score/TOTAL)*100); pp=int(pc/pt*100) if pt else 0; lp=int(lc/lt*100) if lt else 0
    aw="🥇" if pct>=80 else "🥈" if pct>=60 else "🥉"
    sc2="#10B981" if pct>=80 else "#F59E0B" if pct>=60 else "#EF4444"
    awl=tp("High","عالي") if pct>=80 else tp("Moderate","متوسط") if pct>=60 else tp("Needs Improvement","يحتاج تحسين")
    strengths=[]; areas=[]
    if pp>=70: strengths.append(tp("Good at identifying phishing emails","جيد في تحديد رسائل التصيد"))
    else: areas.append(tp("Review phishing indicators more carefully","راجع مؤشرات التصيد بعناية أكبر"))
    if lp>=70: strengths.append(tp("Good at identifying legitimate emails","جيد في تمييز الرسائل الشرعية"))
    else: areas.append(tp("Be cautious not to flag legitimate emails","احذر من تصنيف الرسائل الشرعية كتصيد"))
    recs=[tp("Always verify sender email addresses carefully","تحقق دائماً من عنوان البريد الإلكتروني للمرسل"),
          tp("Never click suspicious links — type URLs directly","لا تنقر على الروابط المشبوهة — اكتب العنوان مباشرة"),
          tp("Be cautious with unexpected attachments","كن حذراً مع المرفقات غير المتوقعة"),
          tp("When in doubt, contact IT or the sender directly","عند الشك، تواصل مع تقنية المعلومات أو المرسل مباشرة")]
    user_name  = (st.session_state.get("user_name") or "")
    user_email = (st.session_state.get("user_email") or "")
    name_line  = f'<div style="color:#F8FAFC;font-size:1rem;font-weight:700;margin-bottom:.2rem;">{html_lib.escape(user_name)}</div>' if user_name else ""
    email_line = f'<div style="color:#64748B;font-size:.8rem;margin-bottom:.3rem;">{html_lib.escape(user_email)}</div>' if user_email else ""
    st.markdown(f'<div style="text-align:center;padding:2rem;border:1px solid rgba(37,99,235,.45);border-radius:24px;background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(8,47,73,.88));margin-bottom:1.5rem;direction:{da};">'
                f'<div style="font-size:1.6rem;font-weight:900;color:#F8FAFC;margin-bottom:.5rem;">📊 {tp("Your Performance Report","تقرير أدائك")}</div>'
                f'{name_line}{email_line}'
                f'<div style="color:#7DD3FC;font-size:.95rem;">{tp(f"Role: {role}","الدور: "+role)}</div></div>',
                unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    card = "border:1px solid rgba(37,99,235,.45);border-radius:16px;padding:1.5rem;text-align:center;background:rgba(2,6,23,.6);direction:{da};height:200px;display:flex;flex-direction:column;justify-content:center;align-items:center;box-sizing:border-box;"
    with c1: st.markdown(f'<div style="{card.format(da=da)}"><div style="color:#94A3B8;font-size:.85rem;margin-bottom:.4rem;">{tp("Overall Score","النتيجة الإجمالية")}</div><div style="font-size:2.5rem;font-weight:900;color:{sc2};">{score}/{TOTAL}</div><div style="color:{sc2};font-size:.9rem;">{pct}%</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div style="{card.format(da=da)}"><div style="color:#94A3B8;font-size:.85rem;margin-bottom:.4rem;">{tp("Awareness Level","مستوى الوعي")}</div><div style="font-size:2.5rem;">{aw}</div><div style="color:{sc2};font-weight:700;font-size:.95rem;">{awl}</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div style="{card.format(da=da)}"><div style="color:#94A3B8;font-size:.85rem;margin-bottom:.3rem;">{tp("Detection Rate","معدل الاكتشاف")}</div><div style="font-size:1rem;font-weight:700;color:#FCA5A5;margin-bottom:.3rem;">🚨 {tp("Phishing detected","التصيد المكتشف")}: {pp}%</div><div style="font-size:1rem;font-weight:700;color:#6EE7B7;">✅ {tp("Legitimate identified","الشرعية المميزة")}: {lp}%</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    s1,s2=st.columns(2)
    with s1:
        si="".join([f'<div style="color:#6EE7B7;margin-bottom:.4rem;text-align:{"right" if is_arabic else "left"};">✅ {s}</div>' for s in strengths]) or f'<div style="color:#94A3B8;">{tp("Keep practicing","استمر في التدريب")}</div>'
        st.markdown(f'<div style="border:1px solid rgba(16,185,129,.35);border-radius:14px;padding:1.2rem;background:rgba(16,185,129,.05);direction:{da};text-align:{"right" if is_arabic else "left"};min-height:160px;"><div style="font-weight:800;color:#F1F5F9;margin-bottom:.8rem;text-align:{"right" if is_arabic else "left"};">💪 {tp("Strengths","نقاط القوة")}</div>{si}</div>',unsafe_allow_html=True)
    with s2:
        ai2="".join([f'<div style="color:#FCA5A5;margin-bottom:.4rem;text-align:{"right" if is_arabic else "left"};">⚠️ {a}</div>' for a in areas]) or f'<div style="color:#94A3B8;">{tp("Great work!","عمل رائع!")}</div>'
        st.markdown(f'<div style="border:1px solid rgba(239,68,68,.35);border-radius:14px;padding:1.2rem;background:rgba(239,68,68,.05);direction:{da};text-align:{"right" if is_arabic else "left"};min-height:160px;"><div style="font-weight:800;color:#F1F5F9;margin-bottom:.8rem;text-align:{"right" if is_arabic else "left"};">📈 {tp("Areas to Improve","مجالات التحسين")}</div>{ai2}</div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    ri="".join([f'<div style="color:#DCEBFF;margin-bottom:.5rem;text-align:{"right" if is_arabic else "left"};">📌 {r}</div>' for r in recs])
    st.markdown(f'<div style="border:1px solid rgba(37,99,235,.45);border-radius:14px;padding:1.2rem 1.5rem;background:rgba(2,6,23,.6);margin-bottom:1.5rem;direction:{da};text-align:{"right" if is_arabic else "left"};"><div style="font-weight:800;color:#F1F5F9;margin-bottom:.8rem;text-align:{"right" if is_arabic else "left"};">💡 {tp("Recommendations","التوصيات")}</div>{ri}</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;padding:.8rem;border:1px solid rgba(37,99,235,.3);border-radius:10px;background:rgba(37,99,235,.08);color:#7DD3FC;margin-bottom:1.5rem;">⭐ {tp("Your awareness helps keep your organization safe","وعيك يساهم في حماية مؤسستك")}</div>',unsafe_allow_html=True)
    if st.button(tp("Retake Training","إعادة التدريب من البداية"),key="retake", use_container_width=True):
        # FIX 6: مسح كامل لكل session data لضمان تنوع المحتوى
        keys_to_clear = [
            "page","example_index","emails","assess_index",
            "assess_emails","assess_answers","assess_pattern",
            "cache_version","role","scenario_order",
            "assess_scenario_order","difficulty",
            "user_name","user_email",
            "lang_explicitly_chosen","diff_explicitly_chosen",
            "login_mode","assess_index",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        # EN: also drop the dynamic per-role scenario/recipient random-order
        # picks (scenario_order_*, recipient_order_*, assess_*_order_*) so
        # "Retake Training" actually reshuffles them instead of reusing the
        # same session pick — otherwise picking the same role again would
        # silently replay the same scenario/recipient order as before.
        # AR: نمسح كمان اختيارات الترتيب العشوائي الديناميكية لكل دور
        # (scenario_order_*, recipient_order_*, assess_*_order_*) حتى زر
        # "إعادة التدريب" يعيد خلطها فعليًا بدل إعادة استخدام نفس اختيار
        # الجلسة القديم — لو اخترتِ نفس الدور بعد إعادة المحاولة.
        for k in list(st.session_state.keys()):
            if k.startswith(("scenario_order_", "recipient_order_", "assess_recipient_order_", "assess_scenario_order_",
                              "category_order_", "assess_category_order_", "used_topics_")):
                st.session_state.pop(k, None)
        # تجديد الـ cache_version لإجبار النموذج على توليد محتوى جديد
        st.session_state["cache_version"] = int(__import__("time").time()) % 99999 + 20
        st.rerun()


def page_login():
    is_arabic = st.session_state["language"] == "Arabic"
    da = 'rtl' if is_arabic else 'ltr'
    mode = st.session_state.get("login_mode","login")
    def tl(e,a): return a if is_arabic else e

    is_reg = mode == "register"
    page_title = tl("Create Account","إنشاء حساب") if is_reg else tl("Welcome Back","مرحباً بك")
    page_sub   = tl("Enter your details to get started","أدخل بياناتك للبدء") if is_reg else tl("Enter your details to personalise your experience","أدخل بياناتك لتخصيص تجربتك التدريبية")
    page_icon  = "✨" if is_reg else "👤"

    st.markdown(f"""<style>
#MainMenu,header,footer{{visibility:hidden;}}
.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}
.block-container{{max-width:480px;padding-top:4rem;}}
.stTextInput>div>div>input{{background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;border-radius:12px !important;min-height:48px;direction:{da};font-size:.95rem !important;}}
.stTextInput label{{color:#94A3B8 !important;font-size:.85rem !important;}}
.stButton>button{{width:100% !important;min-height:48px !important;max-height:48px !important;font-weight:700 !important;border-radius:12px !important;font-size:.9rem !important;padding:0 16px !important;line-height:48px !important;}}
div[data-testid="stHorizontalBlock"] > div:first-child .stButton>button{{background:rgba(15,23,42,.88) !important;color:#EAF4FF !important;border:1px solid rgba(37,99,235,.55) !important;}}
div[data-testid="stHorizontalBlock"] > div:last-child .stButton>button{{background:rgba(15,23,42,.88) !important;color:#EAF4FF !important;border:1px solid rgba(37,99,235,.55) !important;}}
</style>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="text-align:center;padding:2.5rem 2rem 2rem;border:1px solid rgba(37,99,235,.45);border-radius:24px;background:linear-gradient(135deg,rgba(2,6,23,.96),rgba(8,47,73,.88));direction:{da};margin-bottom:1.5rem;">
  <div style="font-size:2.8rem;margin-bottom:.8rem;">{page_icon}</div>
  <div style="font-size:1.4rem;font-weight:900;color:#F8FAFC;margin-bottom:.4rem;">{page_title}</div>
  <div style="font-size:.9rem;color:#94A3B8;">{page_sub}</div>
</div>""", unsafe_allow_html=True)

    if is_arabic:
        st.markdown('<style>.stTextInput label{direction:rtl;text-align:right;display:block;}.stTextInput input{text-align:right;direction:rtl;}</style>', unsafe_allow_html=True)

    user_name  = st.text_input(tl("Full name","الاسم الكامل"), value=(st.session_state.get("user_name") or ""), placeholder=tl("e.g. Dr. Sarah Al-Mutairi","مثال: د. سارة المطيري"))
    user_email = st.text_input(tl("Email address","البريد الإلكتروني"), value=(st.session_state.get("user_email") or ""), placeholder="name@hospital.org")

    st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)
    st.markdown("""<style>div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button{height:48px !important;min-height:48px !important;max-height:48px !important;padding-top:0 !important;padding-bottom:0 !important;display:flex !important;align-items:center !important;justify-content:center !important;box-sizing:border-box !important;}</style>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button(tl("← Back","← رجوع"), key="login_back", use_container_width=True):
            st.session_state["page"] = "home"; st.rerun()
    with c2:
        if st.button(tl("Continue","متابعة"), key="login_continue", use_container_width=True):
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
            if not user_name.strip():
                st.warning(tl("⚠️ Please enter your full name.","⚠️ يرجى إدخال اسمك الكامل"))
            elif not user_email.strip():
                st.warning(tl("⚠️ Please enter your email address.","⚠️ يرجى إدخال بريدك الإلكتروني"))
            elif not email_pattern.match(user_email.strip()):
                st.warning(tl("⚠️ Please enter a valid email address (e.g. name@hospital.org).","⚠️ يرجى إدخال بريد إلكتروني صحيح مثل: name@hospital.org"))
            else:
                st.session_state["user_name"]  = user_name.strip()
                st.session_state["user_email"] = user_email.strip()
                st.session_state["page"] = "home"; st.rerun()


# ══════════════════════════════════════════════════════════════
# ADMIN PANEL — مخفي، يظهر فقط عبر ?admin=true
# ══════════════════════════════════════════════════════════════
def page_admin():
    def get_secret(key):
        try:
            return st.secrets[key]
        except Exception:
            return os.environ.get(key, "")

    ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD") or "admin2025"

    # ── اللغة المختارة من المستخدم في الصفحة الرئيسية ──
    _lang = st.session_state.get("language", "English")
    _is_ar = (_lang == "Arabic")

    _T = {
        "admin_title":      {"en": "Admin Panel",                          "ar": "لوحة التحكم"},
        "admin_subtitle":   {"en": "Researcher Access Only",               "ar": "للباحثين فقط"},
        "pwd_placeholder":  {"en": "Enter admin password",                 "ar": "أدخل كلمة سر الإدارة"},
        "login_btn":        {"en": "🔓 Login",                             "ar": "🔓 دخول"},
        "back_btn":         {"en": "← Back to App",                       "ar": "→ رجوع للتطبيق"},
        "wrong_pwd":        {"en": "❌ Incorrect password",                "ar": "❌ كلمة السر غير صحيحة"},
        "panel_title":      {"en": "🔬 Admin Research Panel",              "ar": "🔬 لوحة البحث الإدارية"},
        "authenticated":    {"en": "● Authenticated",                      "ar": "● تم تسجيل الدخول"},
        "tab_provider":     {"en": "⚙️ Provider Control",                  "ar": "⚙️ التحكم بالمزوّد"},
        "tab_score":        {"en": "📊 Score Card",                       "ar": "📊 بطاقة التقييم"},
        "tab_manual":       {"en": "👍 Manual Ratings",                   "ar": "👍 التقييم اليدوي"},
        "select_provider":  {"en": "Select Active AI Provider",            "ar": "اختر مزوّد الذكاء الاصطناعي النشط"},
        "active":           {"en": "ACTIVE",                               "ar": "نشط"},
        "activate_btn":     {"en": "Activate",                             "ar": "تفعيل"},
        "api_status":       {"en": "API Keys Status",                      "ar": "حالة مفاتيح API"},
        "not_set":          {"en": "Pending setup",                          "ar": "بانتظار الإضافة"},
        "difficulty_lvl":   {"en": "Difficulty Level",                     "ar": "مستوى الصعوبة"},
        "language_lbl":     {"en": "Language",                             "ar": "اللغة"},
        "set_btn":          {"en": "Set",                                  "ar": "تطبيق"},
        "easy":             {"en": "Easy",                                 "ar": "سهل"},
        "medium":           {"en": "Medium",                               "ar": "متوسط"},
        "hard":             {"en": "Hard",                                 "ar": "صعب"},
        "clear_cache":      {"en": "🔄 Clear All Cached Emails (Force Regenerate)", "ar": "🔄 مسح كل الرسائل المخزّنة (توليد من جديد)"},
        "cache_cleared":    {"en": "✅ Cache cleared — next generation will produce fresh content", "ar": "✅ تم مسح الذاكرة المؤقتة — التوليد القادم سينتج محتوى جديد"},
        "logout_btn":       {"en": "🚪 Logout",                            "ar": "🚪 تسجيل خروج"},
        "score_title":      {"en": "📊 Comparison Score Card",             "ar": "📊 بطاقة المقارنة"},
        "metric_col":       {"en": "Metric",                               "ar": "المعيار"},
        "speed_metric":     {"en": "⚡ Avg Speed (s)",                     "ar": "⚡ متوسط السرعة (ث)"},
        "json_metric":      {"en": "✅ JSON Success Rate",                 "ar": "✅ نسبة نجاح JSON"},
        "error_metric":     {"en": "🚫 Error Rate",                        "ar": "🚫 نسبة الأخطاء"},
        "diversity_metric": {"en": "🔄 Unique Responses",                  "ar": "🔄 الردود الفريدة"},
        "quality_metric":   {"en": "🎯 Quality",                          "ar": "🎯 الجودة"},
        "difficulty_metric":{"en": "📊 Difficulty Level",                 "ar": "📊 مستوى الصعوبة"},
        "arabic_metric":    {"en": "🌐 Arabic Quality",                   "ar": "🌐 جودة العربية"},
        "medical_metric":   {"en": "🏥 Medical Realism",                  "ar": "🏥 الواقعية الطبية"},
        "auto_manual_note": {"en": "Auto-tracked: Speed, JSON, Errors, Diversity | Manual: Quality, Difficulty, Arabic, Medical",
                              "ar": "تلقائي: السرعة، JSON، الأخطاء، التنوع | يدوي: الجودة، الصعوبة، العربية، الطبي"},
        "reset_metrics":    {"en": "🗑️ Reset All Metrics",                "ar": "🗑️ إعادة ضبط كل المعايير"},
        "metrics_reset":    {"en": "✅ All metrics reset",                 "ar": "✅ تم إعادة ضبط كل المعايير"},
        "rate_title":       {"en": "Rate the last generated content",      "ar": "قيّم آخر محتوى تم توليده"},
        "active_provider":  {"en": "Active provider",                     "ar": "المزوّد النشط"},
        "saved_permanently": {"en": "saved permanently",                  "ar": "محفوظ بشكل دائم"},
        "quality_label":    {"en": "🎯 Model Quality",                    "ar": "🎯 جودة النموذج"},
        "quality_desc":     {"en": "How accurate and realistic is the phishing email?",
                              "ar": "ما مدى دقة وواقعية رسالة التصيد؟"},
        "diff_acc_label":   {"en": "📊 Difficulty Accuracy",              "ar": "📊 دقة الصعوبة"},
        "diff_acc_desc":    {"en": "Does the difficulty level (Easy/Medium/Hard) feel right?",
                              "ar": "هل مستوى الصعوبة (سهل/متوسط/صعب) مناسب؟"},
        "arabic_label":     {"en": "🌐 Arabic Quality",                   "ar": "🌐 جودة اللغة العربية"},
        "arabic_desc":      {"en": "How good is the Arabic language quality? (skip if English)",
                              "ar": "ما مدى جودة اللغة العربية؟ (تخطّى إذا إنجليزي)"},
        "medical_label":    {"en": "🏥 Medical Realism",                  "ar": "🏥 الواقعية الطبية"},
        "medical_desc":     {"en": "How realistic is the healthcare/hospital context?",
                              "ar": "ما مدى واقعية السياق الطبي/الصحي؟"},
        "note_label":       {"en": "📝 Note (optional)",                  "ar": "📝 ملاحظة (اختياري)"},
        "note_placeholder": {"en": "Any observation about this provider...",
                              "ar": "أي ملاحظة حول هذا المزوّد..."},
        "save_btn":         {"en": "💾 Save Ratings",                     "ar": "💾 حفظ التقييم"},
        "ratings_saved":    {"en": "✅ Ratings saved for",                 "ar": "✅ تم حفظ التقييم لـ"},
        "quick_rating":     {"en": "Quick Rating",                        "ar": "تقييم سريع"},
        "good_btn":         {"en": "👍 Good (4/5)",                   "ar": "👍 جيد (4/5)"},
        "avg_btn":          {"en": "😐 Average (3/5)",                   "ar": "😐 متوسط (3/5)"},
        "poor_btn":         {"en": "👎 Poor (2/5)",                   "ar": "👎 ضعيف (2/5)"},
        "saved_45":         {"en": "✅ Saved this cycle — Overall 4/5",         "ar": "✅ تم حفظ هذي الدورة — الانطباع العام 4/5"},
        "saved_35":         {"en": "✅ Saved this cycle — Overall 3/5",         "ar": "✅ تم حفظ هذي الدورة — الانطباع العام 3/5"},
        "saved_25":         {"en": "✅ Saved this cycle — Overall 2/5",         "ar": "✅ تم حفظ هذي الدورة — الانطباع العام 2/5"},
        "rating_history":   {"en": "Rating History for",                  "ar": "سجل التقييم لـ"},
        "avg_label":        {"en": "avg",                                 "ar": "المتوسط"},
        "ratings_label":    {"en": "ratings",                             "ar": "تقييم"},
    }

    def T(key):
        return _T.get(key, {}).get("ar" if _is_ar else "en", key)

    _dir = 'rtl' if _is_ar else 'ltr'
    _align = 'right' if _is_ar else 'left'
    _align_opp = 'left' if _is_ar else 'right'
    _flex_dir = 'row-reverse' if _is_ar else 'row'

    st.markdown(f"""<style>
#MainMenu,header,footer{{visibility:hidden;}}
.stApp{{background:radial-gradient(circle at top left,#0B1A0B 0%,#020617 40%,#020617 100%);color:white;direction:{_dir};}}
.block-container{{max-width:1200px;padding-top:1.5rem;direction:{_dir};text-align:{_align};}}
[data-testid="stMarkdownContainer"]{{direction:{_dir};text-align:{_align};}}
[data-testid="stMarkdownContainer"] > div{{direction:{_dir};text-align:{_align};}}
.block-container div,.block-container span,.block-container p,.block-container label{{
    direction:{_dir};
}}
[data-baseweb="tab-list"]{{direction:{_dir};}}
[data-baseweb="tab-border"]{{direction:{_dir};}}
.stTabs [data-baseweb="tab"]{{direction:{_dir};}}
div[style*="justify-content:space-between"]{{flex-direction:{_flex_dir}!important;}}
.stButton>button{{
    min-height:44px;
    font-weight:700!important;
    border-radius:10px!important;
    background:rgba(15,23,42,.8)!important;
    color:#94A3B8!important;
    border:1px solid rgba(37,99,235,.35)!important;
    direction:{_dir};
}}
.stButton>button:hover{{
    background:rgba(11,79,168,.3)!important;
    color:#E2E8F0!important;
    border-color:#1EA7FF!important;
}}
a[download],
a[download]:link,
a[download]:visited,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] a{{
    font-weight:700!important;
    border-radius:10px!important;
    background:rgba(15,23,42,.8)!important;
    color:#94A3B8!important;
    border:1px solid rgba(37,99,235,.35)!important;
    text-decoration:none!important;
}}
a[download]:hover,
a[download]:focus,
a[download]:active,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stDownloadButton"] button:focus,
div[data-testid="stDownloadButton"] button:active,
div[data-testid="stDownloadButton"] a:hover,
div[data-testid="stDownloadButton"] a:focus,
div[data-testid="stDownloadButton"] a:active{{
    background:rgba(11,79,168,.3)!important;
    color:#E2E8F0!important;
    border-color:#1EA7FF!important;
    text-decoration:none!important;
}}
button[kind="primary"]{{
    background:rgba(37,99,235,.18)!important;
    color:#93C5FD!important;
    border:1px solid rgba(37,99,235,.5)!important;
}}
button[kind="primary"]:hover{{
    background:rgba(37,99,235,.28)!important;
    color:#BFDBFE!important;
}}
.stButton>button:disabled,.stButton>button[disabled]{{
    background:rgba(37,99,235,.12)!important;
    color:#60A5FA!important;
    border:1px solid rgba(37,99,235,.4)!important;
    opacity:1!important;
    cursor:default!important;
}}
.stTextInput input{{
    direction:{_dir};
    text-align:{_align};
    background:transparent!important;
    color:#E2E8F0!important;
    border:none!important;
    box-shadow:none!important;
}}
.stTextInput input::placeholder{{color:#6B7280!important;}}
.stTextInput>div>div,
.stTextInput div[data-baseweb="base-input"],
.stTextInput div[data-baseweb="input"]{{
    background:rgba(15,23,42,.80)!important;
    border:2px solid rgba(148,163,184,.75)!important;
    border-radius:12px!important;
    box-shadow:0 0 0 3px rgba(148,163,184,.08)!important;
}}
.stTextInput>div>div:focus-within,
.stTextInput div[data-baseweb="base-input"]:focus-within,
.stTextInput div[data-baseweb="input"]:focus-within{{
    border:2px solid #60A5FA!important;
    box-shadow:0 0 0 3px rgba(96,165,250,.20)!important;
}}
div[data-baseweb="select"] *{{color:#EAF4FF!important;}}
div[data-baseweb="select"] > div{{background:rgba(15,23,42,.78)!important;border:1px solid rgba(37,99,235,.55)!important;}}
.stSelectbox label{{color:#EAF4FF!important;}}
</style>""", unsafe_allow_html=True)

    def _div(content, extra=""):
        """صندوق نصي يحترم اتجاه اللغة الحالية"""
        return f'<div dir="{_dir}" style="text-align:{_align};{extra}">{content}</div>'

    # ── Authentication ──────────────────────────────────────────
    if not st.session_state.get("admin_authenticated", False):
        st.markdown(f"""
<div style="max-width:420px;margin:6rem auto;padding:2.5rem;
     border:1px solid rgba(34,197,94,.4);border-radius:20px;
     background:linear-gradient(135deg,rgba(2,6,23,.97),rgba(4,20,4,.9));
     text-align:center;">
  <div style="font-size:2.5rem;margin-bottom:.5rem;">🔐</div>
  <div style="font-size:1.3rem;font-weight:900;color:#F0FDF4;margin-bottom:.3rem;">{T('admin_title')}</div>
  <div style="font-size:.85rem;color:#6B7280;margin-bottom:1.5rem;">{T('admin_subtitle')}</div>
</div>""", unsafe_allow_html=True)
        pwd = st.text_input(T('pwd_placeholder'), type="password", placeholder=T('pwd_placeholder'),
                            label_visibility="collapsed")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button(T('login_btn'), use_container_width=True):
                if pwd == ADMIN_PASSWORD:
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error(T('wrong_pwd'))
        with col2:
            if st.button(T('back_btn'), use_container_width=True):
                st.query_params.clear()
                st.session_state["page"] = "home"
                st.rerun()
        return

    # ══════════════════════════════════════════════════════════
    # MAIN ADMIN PANEL (after login)
    # ══════════════════════════════════════════════════════════
    st.markdown(f"""
<div dir="{_dir}" style="display:flex;justify-content:space-between;align-items:center;
     padding:.8rem 1.2rem;border:1px solid rgba(34,197,94,.35);border-radius:14px;
     background:rgba(4,20,4,.6);margin-bottom:1.5rem;">
  <div style="font-size:1.3rem;font-weight:900;color:#F0FDF4;">{T('panel_title')}</div>
  <div style="font-size:.8rem;color:#4ADE80;">{T('authenticated')}</div>
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([T('tab_provider'), T('tab_score'), T('tab_manual'), "🐞 Debug Log"])

    _persist_pk = st.session_state.get("ai_provider", load_persistent_provider("openai"))
    _persist_labels = {
        "groq":      "🟠 Groq (LLaMA 3.3-70b)",
        "anthropic": "🟣 Claude (claude-sonnet-4-6)",
        "openai":    "🟢 OpenAI (GPT-4o)",
        "gemini":    "🔵 Gemini",
    }
    st.markdown(f"""
<div dir="{_dir}" style="display:flex;justify-content:space-between;align-items:center;
     padding:.6rem 1.2rem;border:1px solid rgba(245,158,11,.5);border-radius:12px;
     background:rgba(40,30,4,.5);margin-bottom:1rem;">
  <div style="font-size:.85rem;color:#FCD34D;">💾 {T('active_provider')} ({T('saved_permanently')}):</div>
  <div style="font-size:.95rem;font-weight:900;color:#FBBF24;">{_persist_labels.get(_persist_pk, _persist_pk)}</div>
</div>""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # TAB 1 — Provider Control
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

        provider_info = {
            "groq":      {"label": "🟠 Groq — LLaMA 3.3-70b",       "secret": "GROQ_API_KEY",      "color": "#F97316"},
            "anthropic": {"label": "🟣 Claude — claude-sonnet-4-6",  "secret": "ANTHROPIC_API_KEY", "color": "#A855F7"},
            "openai":    {"label": "🟢 OpenAI — GPT-4o",             "secret": "OPENAI_API_KEY",    "color": "#22C55E"},
            "gemini":    {"label": "🔵 Gemini — 2.5 Flash",          "secret": "GEMINI_API_KEY",    "color": "#3B82F6"},
        }

        cur = st.session_state.get("ai_provider", "openai")

        st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#D1FAE5;margin-bottom:.8rem;">{T("select_provider")}</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (pk, pv) in enumerate(provider_info.items()):
            with cols[i]:
                is_sel = cur == pk
                status_line = f'<div style="font-size:.7rem;color:#9CA3AF;margin-top:.3rem;">● {T("active")}</div>' if is_sel else '<div style="font-size:.7rem;margin-top:.3rem;">&nbsp;</div>'
                st.markdown(f'<div dir="{_dir}" style="background:rgba(0,0,0,.2);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:.8rem;text-align:center;margin-bottom:.5rem;height:84px;display:flex;flex-direction:column;justify-content:center;align-items:center;overflow:hidden;">'
                            f'<div style="font-size:.85rem;font-weight:700;color:#9CA3AF;line-height:1.25;display:flex;align-items:center;justify-content:center;height:2.5em;">{pv["label"]}</div>'
                            f'{status_line}'
                            f'</div>', unsafe_allow_html=True)
                if not is_sel:
                    if st.button(T('activate_btn'), key=f"adm_prov_{pk}", use_container_width=True):
                        set_active_provider(pk)
                        # Clear cached emails to regenerate with new provider
                        st.session_state["emails"] = {}
                        st.session_state.pop("assess_emails", None)
                        st.session_state["cache_version"] = int(__import__("time").time()) % 99999 + 20
                        st.rerun()
                else:
                    st.button(T('active'), key=f"adm_prov_{pk}_disabled", use_container_width=True, disabled=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#D1FAE5;margin-bottom:.8rem;">{T("api_status")}</div>', unsafe_allow_html=True)
        key_cols = st.columns(4)
        for i, (pk, pv) in enumerate(provider_info.items()):
            with key_cols[i]:
                val = get_secret(pv["secret"])
                has_key = bool(val and len(val) > 10)
                dot = "🟢" if has_key else "🔴"
                preview = f"...{val[-4:]}" if has_key else T('not_set')
                st.markdown(f'<div style="border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.7rem;text-align:center;">'
                            f'<div style="font-size:.75rem;color:#9CA3AF;margin-bottom:.3rem;">{pv["label"].split("—")[0].strip()}</div>'
                            f'<div style="font-size:.85rem;font-weight:700;color:white;">{dot} {preview}</div>'
                            f'</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#D1FAE5;margin-bottom:.5rem;">{T("language_lbl")}</div>', unsafe_allow_html=True)
        cur_lang = st.session_state.get("language", "English")
        lang_display = {"English": "English", "Arabic": "العربية"}
        col_lang1, col_lang2 = st.columns(2)
        for lk, lcol in zip(["English", "Arabic"], [col_lang1, col_lang2]):
            with lcol:
                is_l = cur_lang == lk
                label = f"{lang_display[lk]} ✓" if is_l else lang_display[lk]
                if st.button(label, key=f"adm_lang_{lk}", use_container_width=True,
                             type="primary" if is_l else "secondary"):
                    if not is_l:
                        st.session_state["language"] = lk
                        st.session_state["emails"] = {}
                        st.rerun()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        _gs_ok, _gs_msg = gsheet_setup_status()
        _gs_id = _get_gsheet_id()
        _gs_border = "rgba(34,197,94,.4)" if _gs_ok else "rgba(245,158,11,.4)"
        _gs_bg = "rgba(4,20,4,.5)" if _gs_ok else "rgba(40,30,4,.5)"
        _gs_title_color = "#86EFAC" if _gs_ok else "#FCD34D"
        _gs_title = "📊 " + ("نسخة Google Sheets الدائمة" if _is_ar else "Durable Google Sheets backup")
        _gs_link_html = ""
        if _gs_ok and _gs_id:
            _gs_link_text = "فتح الشيت ↗" if _is_ar else "Open Sheet ↗"
            _gs_link_html = (
                '<a href="https://docs.google.com/spreadsheets/d/' + _gs_id + '/edit" '
                'target="_blank" style="color:#60A5FA;font-weight:700;text-decoration:none;">'
                + _gs_link_text + '</a>'
            )
        _gs_html = (
            '<div dir="' + _dir + '" style="padding:.8rem 1rem;border-radius:10px;'
            'border:1px solid ' + _gs_border + ';background:' + _gs_bg + ';'
            'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">'
            '<div>'
            '<div style="font-weight:800;color:' + _gs_title_color + ';">' + _gs_title + '</div>'
            '<div style="font-size:.8rem;color:#9CA3AF;">' + _gs_msg + '</div>'
            '</div>'
            + _gs_link_html +
            '</div>'
        )
        st.markdown(_gs_html, unsafe_allow_html=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        if st.button(T('clear_cache'), use_container_width=True):
            st.session_state["emails"] = {}
            st.session_state.pop("assess_emails", None)
            st.session_state["cache_version"] = int(__import__("time").time()) % 99999 + 20
            st.success(T('cache_cleared'))
            st.rerun()

        if st.button(T('logout_btn'), use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.query_params.clear()
            st.rerun()

    # ──────────────────────────────────────────────────────────
    # TAB 2 — Score Card: one independent card per provider
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        metrics = st.session_state.get("metrics", {})
        runs = load_runs()

        PROV_ORDER = ["groq", "anthropic", "openai", "gemini"]
        PROV_META = {
            "groq":      {"label": "🟠 Groq — LLaMA 3.3-70b",      "color": "#F97316"},
            "anthropic": {"label": "🟣 Claude — claude-sonnet-4-6", "color": "#A855F7"},
            "openai":    {"label": "🟢 OpenAI — GPT-4o",            "color": "#22C55E"},
            "gemini":    {"label": "🔵 Gemini — 2.5 Flash",         "color": "#3B82F6"},
        }
        TARGET_PER_LANG = 5

        def avg(lst):
            return round(sum(lst)/len(lst), 1) if lst else None

        def get_m(p): return metrics.get(p, {})

        # ── Rotation plan reference table (informational only) ──
        with st.expander("📋 " + ("جدول التدوير المنظّم (10 دورات)" if _is_ar else "Systematic Rotation Plan (10 cycles)")):
            rcols = st.columns([1,2,2,1.5])
            for ci, hdr in enumerate([("#" if _is_ar else "#"), ("الوظيفة" if _is_ar else "Role"), ("المستوى" if _is_ar else "Difficulty"), ("اللغة" if _is_ar else "Language")]):
                with rcols[ci]:
                    st.markdown(f'<div style="font-weight:800;color:#9CA3AF;font-size:.78rem;">{hdr}</div>', unsafe_allow_html=True)
            for plan in ROTATION_PLAN:
                rcols = st.columns([1,2,2,1.5])
                role_show = plan["role_ar"] if _is_ar else plan["role_en"]
                diff_show = {"easy": ("سهل" if _is_ar else "Easy"), "medium": ("متوسط" if _is_ar else "Medium"), "hard": ("صعب" if _is_ar else "Hard")}[plan["difficulty"]]
                lang_show = ("🇸🇦 عربي" if plan["language"]=="Arabic" else "🇬🇧 EN") if _is_ar else ("Arabic" if plan["language"]=="Arabic" else "English")
                for ci, val in enumerate([str(plan["cycle"]), role_show, diff_show, lang_show]):
                    with rcols[ci]:
                        st.markdown(f'<div style="color:#E2E8F0;font-size:.82rem;padding:.15rem 0;">{val}</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        # ── One independent card per provider ──
        _gsheet_auto_latest = pull_latest_auto_metrics_from_gsheet()
        for p in PROV_ORDER:
            meta = PROV_META[p]
            m = get_m(p)
            speeds = m.get("speed", [])
            total_j = m.get("json_ok",0) + m.get("json_fail",0)
            json_rate = int(m.get("json_ok",0)/total_j*100) if total_j > 0 else None
            calls = m.get("calls",0)
            err_rate = int(m.get("errors",0)/calls*100) if calls > 0 else None
            hashes = m.get("hashes",[])

            # No local activity recorded for this provider this session
            # (e.g. right after a container restart) — fall back to the
            # last snapshot synced to Google Sheets instead of showing
            # blank boxes for data that genuinely exists.
            if calls == 0:
                _snap = _gsheet_auto_latest.get(p)
                if _snap:
                    try:
                        _sp = _snap.get("avg_speed_s")
                        speeds = [float(_sp)] if _sp not in ("", None) else []
                    except (ValueError, TypeError):
                        speeds = []
                    try:
                        json_rate = int(float(_snap.get("json_success_rate_pct"))) if _snap.get("json_success_rate_pct") not in ("", None) else None
                    except (ValueError, TypeError):
                        json_rate = None
                    try:
                        err_rate = int(float(_snap.get("error_rate_pct"))) if _snap.get("error_rate_pct") not in ("", None) else None
                    except (ValueError, TypeError):
                        err_rate = None
                    try:
                        calls = int(float(_snap.get("calls"))) if _snap.get("calls") not in ("", None) else 0
                    except (ValueError, TypeError):
                        calls = 0
                    try:
                        _uniq = int(float(_snap.get("unique_responses"))) if _snap.get("unique_responses") not in ("", None) else 0
                    except (ValueError, TypeError):
                        _uniq = 0
                    hashes = list(range(_uniq))  # only its length is used below

            p_runs_en = [r for r in runs if r.get("provider")==p and r.get("language")=="English"]
            p_runs_ar = [r for r in runs if r.get("provider")==p and r.get("language")=="Arabic"]
            ordered_runs = p_runs_en + p_runs_ar

            def _avg_field(field):
                vals = [r.get(field) for r in ordered_runs if r.get(field) is not None]
                return round(sum(vals)/len(vals), 1) if vals else None

            avg_diff    = _avg_field("auto_difficulty")
            avg_arabic  = _avg_field("auto_arabic")
            avg_quality = _avg_field("auto_quality")
            avg_medical = _avg_field("auto_medical")
            avg_overall = _avg_field("overall")

            st.markdown(f"""
<div dir="{_dir}" style="border:1px solid {meta['color']}55;border-radius:14px;padding:1rem 1.2rem;margin-bottom:1.2rem;background:rgba(255,255,255,.02);">
  <div style="font-weight:900;font-size:1.05rem;color:{meta['color']};margin-bottom:.6rem;">{meta['label']}</div>
</div>""", unsafe_allow_html=True)

            # 9 uniform boxes — 4 auto-performance + 4 auto-content + 1 manual overall
            box_items = [
                (T('speed_metric'),      f"{avg(speeds):.1f}s" if speeds else "—"),
                (T('json_metric'),       f"{json_rate}%" if json_rate is not None else "—"),
                (T('error_metric'),      f"{err_rate}%" if err_rate is not None else "—"),
                (T('diversity_metric'),  f"{len(hashes)}/{calls}" if calls > 0 else "—"),
                (("صعوبة%" if _is_ar else "Difficulty%"), f"{avg_diff}%" if avg_diff is not None else "—"),
                (("عربي%" if _is_ar else "Arabic%"),      f"{avg_arabic}%" if avg_arabic is not None else "—"),
                (("جودة%" if _is_ar else "Quality%"),     f"{avg_quality}%" if avg_quality is not None else "—"),
                (("طبي%" if _is_ar else "Medical%"),      f"{avg_medical}%" if avg_medical is not None else "—"),
                (("الانطباع⭐" if _is_ar else "Overall⭐"), f"{avg_overall}/5" if avg_overall is not None else "—"),
            ]
            box_rows = [box_items[i:i+3] for i in range(0, 9, 3)]
            for row in box_rows:
                bc = st.columns(3)
                for ci, (lbl, val) in enumerate(row):
                    with bc[ci]:
                        st.markdown(f'<div style="text-align:center;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:.5rem;margin-bottom:.5rem;">'
                                    f'<div style="font-size:.7rem;color:#9CA3AF;">{lbl}</div>'
                                    f'<div style="font-size:1rem;font-weight:800;color:#E2E8F0;">{val}</div></div>', unsafe_allow_html=True)

            n_en, n_ar = len(p_runs_en), len(p_runs_ar)
            tag_en = "✅" if n_en >= TARGET_PER_LANG else "🟡"
            tag_ar = "✅" if n_ar >= TARGET_PER_LANG else "🟡"
            st.markdown(f'<div style="color:#9CA3AF;font-size:.78rem;margin:.2rem 0 .6rem;">'
                        f'{tag_en} English: {n_en}/{TARGET_PER_LANG} &nbsp;&nbsp; {tag_ar} Arabic: {n_ar}/{TARGET_PER_LANG}</div>',
                        unsafe_allow_html=True)

            # Detailed per-cycle table, tucked away in an expander so the
            # 9 boxes above stay the clean at-a-glance summary.
            with st.expander("📋 " + ("عرض تفاصيل الـ10 دورات" if _is_ar else "View detailed 10-cycle breakdown")):
                col_widths = [0.5,0.7,0.8,0.7,0.7,0.8,0.9,0.9,0.9,0.9,0.9]
                cols_hdr = st.columns(col_widths)
                headers = [
                    "#", ("لغة" if _is_ar else "Lang"),
                    ("سرعة" if _is_ar else "Speed"), ("JSON%"), ("أخطاء%" if _is_ar else "Err%"),
                    ("تنوع" if _is_ar else "Divers."),
                    ("صعوبة%" if _is_ar else "Diff%"), ("عربي%" if _is_ar else "Arabic%"),
                    ("جودة%" if _is_ar else "Quality%"), ("طبي%" if _is_ar else "Medical%"),
                    ("انطباع" if _is_ar else "Overall"),
                ]
                for ci, hdr in enumerate(headers):
                    with cols_hdr[ci]:
                        st.markdown(f'<div style="font-weight:800;color:#9CA3AF;font-size:.68rem;border-bottom:1px solid rgba(255,255,255,.1);padding:.2rem 0;">{hdr}</div>', unsafe_allow_html=True)

                if ordered_runs:
                    for i, r in enumerate(ordered_runs, 1):
                        rc = st.columns(col_widths)
                        lang_short = "EN" if r.get("language")=="English" else "AR"
                        vals = [
                            str(i),
                            lang_short,
                            f"{r.get('avg_speed')}s" if r.get('avg_speed') is not None else "—",
                            f"{r.get('json_rate')}%" if r.get('json_rate') is not None else "—",
                            f"{r.get('error_rate')}%" if r.get('error_rate') is not None else "—",
                            r.get('diversity') or "—",
                            f"{r.get('auto_difficulty')}%" if r.get('auto_difficulty') is not None else "—",
                            f"{r.get('auto_arabic')}%" if r.get('auto_arabic') is not None else "—",
                            f"{r.get('auto_quality')}%" if r.get('auto_quality') is not None else "—",
                            f"{r.get('auto_medical')}%" if r.get('auto_medical') is not None else "—",
                            f"{r.get('overall')}/5" if r.get('overall') is not None else "—",
                        ]
                        for ci, val in enumerate(vals):
                            with rc[ci]:
                                st.markdown(f'<div style="color:#E2E8F0;font-size:.74rem;padding:.3rem .2rem;border-radius:6px;'
                                            f'background:{"rgba(255,255,255,.03)" if i%2==0 else "transparent"};">{val}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color:#6B7280;font-size:.8rem;padding:.5rem 0;">{("لا توجد دورات محفوظة لهذا المزوّد بعد" if _is_ar else "No cycles saved for this provider yet")}</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

        st.markdown(f'<div dir="{_dir}" style="font-size:.75rem;color:#6B7280;">{T("auto_manual_note")}</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

        if not runs:
            st.markdown(f'<div dir="{_dir}" style="text-align:{_align};font-size:.8rem;color:#6B7280;">⚠️ ' + ("احفظي تقييمًا واحدًا على الأقل من تبويب Manual Ratings لتفعيل التصدير." if _is_ar else "Save at least one rating from the Manual Ratings tab to enable export.") + '</div>', unsafe_allow_html=True)
        col_exp, col_reset = st.columns(2)
        with col_exp:
            if runs:
                st.download_button(
                    "⬇️ " + ("تصدير كل النتائج Excel" if _is_ar else "Export full results (Excel)"),
                    data=build_excel_export(),
                    file_name="phishing_research_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="export_excel_scorecard",
                )
            else:
                st.button("⬇️ " + ("تصدير كل النتائج Excel" if _is_ar else "Export full results (Excel)"), use_container_width=True, disabled=True, key="export_excel_scorecard_disabled")
        with col_reset:
            if st.button(T('reset_metrics'), use_container_width=True):
                st.session_state["metrics"] = {}
                save_metrics_file({})
                delete_all_runs()
                clear_gsheet_data()
                _save_json_list(_AUTO_EVAL_FILE_PATH, [])
                _save_pending_buckets({})
                _save_json_dict(_PENDING_PERF_FILE_PATH, {})
                st.success(T('metrics_reset'))
                st.rerun()

    # ──────────────────────────────────────────────────────────
    # TAB 3 — Manual Ratings (👍 اليدوية)
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        cur_prov = st.session_state.get("ai_provider", "groq")
        prov_label = provider_info.get(cur_prov, {}).get("label", cur_prov)

        st.markdown(
            f'<div style="font-weight:900;color:#D1FAE5;margin-bottom:.3rem;">'
            f'{"قيّمي الدورة الكاملة (٦ تعلّم + ١٠ اختبار) بعد ما تخلّصينها" if _is_ar else "Rate the FULL cycle (6 learning + 10 assessment) after finishing it"}'
            f'</div>'
            f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1rem;">{T("active_provider")}: {prov_label}</div>',
            unsafe_allow_html=True)

        # Compute which cycle # comes next (based on saved cycles for this
        # provider) BEFORE showing the language radio, so the radio can
        # default to whatever the rotation plan says for that cycle.
        _runs_now_pre = load_runs()
        _n_done_total = len([r for r in _runs_now_pre if r.get("provider")==cur_prov])
        _cycle_no = min(_n_done_total + 1, 10)
        _plan = ROTATION_PLAN[_cycle_no - 1]
        _plan_role = _plan["role_ar"] if _is_ar else _plan["role_en"]
        _plan_diff = {"easy": ("سهل" if _is_ar else "Easy"), "medium": ("متوسط" if _is_ar else "Medium"), "hard": ("صعب" if _is_ar else "Hard")}[_plan["difficulty"]]
        _plan_lang = _plan["language"]

        # Which language was this cycle run in? Defaults to whatever the
        # rotation plan says for this cycle number (can be overridden).
        lang_for_run = st.radio(
            ("🏷️ صنّفي بيانات هذي الدورة: المحتوى المولّد كان بـ" if _is_ar else "🏷️ Tag this cycle's data: generated content was in"),
            options=["English", "Arabic"],
            index=0 if _plan_lang=="English" else 1,
            horizontal=True,
            key="run_lang_selector",
        )

        # Progress counter for this provider+language combo (target 5+5=10)
        _runs_now = load_runs()
        _n_done_lang = len([r for r in _runs_now if r.get("provider")==cur_prov and r.get("language")==lang_for_run])
        _target_lang = 5
        _pct = min(_n_done_lang / _target_lang, 1.0)

        st.markdown(
            f'<div style="margin:.3rem 0 .6rem;">'
            f'<div style="color:#93C5FD;font-size:.9rem;font-weight:700;margin-bottom:.3rem;">'
            f'{("اللغة:" if _is_ar else "Language:")} {lang_for_run} — {_n_done_lang}/{_target_lang}'
            f'</div>'
            f'<div style="background:rgba(255,255,255,.1);border-radius:6px;height:8px;overflow:hidden;">'
            f'<div style="background:#22C55E;height:100%;width:{_pct*100:.0f}%;"></div>'
            f'</div></div>',
            unsafe_allow_html=True)

        _plan_lang_show = ("🇸🇦 عربي" if _plan_lang=="Arabic" else "🇬🇧 إنجليزي") if _is_ar else _plan_lang
        st.markdown(
            f'<div dir="{_dir}" style="border:1px solid rgba(245,158,11,.4);border-radius:10px;padding:.6rem .9rem;'
            f'background:rgba(40,30,4,.4);margin-bottom:1rem;font-size:.85rem;color:#FCD34D;">'
            f'📋 {("الدورة التالية حسب جدول التدوير رقم" if _is_ar else "Next cycle per rotation plan, #")} {_cycle_no}/10 — '
            f'{("الوظيفة" if _is_ar else "Role")}: <b>{_plan_role}</b> | {("المستوى" if _is_ar else "Difficulty")}: <b>{_plan_diff}</b> | {("اللغة" if _is_ar else "Language")}: <b>{_plan_lang_show}</b>'
            f'</div>',
            unsafe_allow_html=True)

        # Snapshot of the automatic scores collected since the last saved cycle
        _pending = _load_pending_buckets().get(f"{cur_prov}__{lang_for_run}", [])
        _pending_perf = _load_json_dict(_PENDING_PERF_FILE_PATH).get(f"{cur_prov}__{lang_for_run}", [])
        if _pending or _pending_perf:
            def _pavg(field):
                vals = [it[field] for it in _pending if it.get(field) is not None]
                return round(sum(vals)/len(vals), 1) if vals else None
            _perf_speeds = [it["speed"] for it in _pending_perf if it.get("speed") is not None]
            _perf_speed_avg = round(sum(_perf_speeds)/len(_perf_speeds), 1) if _perf_speeds else None
            _perf_calls = len(_pending_perf)
            _perf_errors = sum(1 for it in _pending_perf if it.get("is_error"))
            _perf_err_rate = round(_perf_errors/_perf_calls*100) if _perf_calls else None
            st.markdown(
                f'<div style="border:1px solid rgba(34,197,94,.35);border-radius:10px;padding:.6rem .9rem;'
                f'background:rgba(4,30,10,.4);margin-bottom:1rem;font-size:.82rem;color:#86EFAC;">'
                f'⚙️ {("نتائج آلية لهذي الدورة لحد الآن" if _is_ar else "Automatic scores collected so far this cycle")} '
                f'({len(_pending)} {("إيميل" if _is_ar else "emails")}): '
                f'{("صعوبة" if _is_ar else "Difficulty")} {_pavg("difficulty_score")}% · '
                f'{("عربي" if _is_ar else "Arabic")} {_pavg("arabic_score")}% · '
                f'{("جودة" if _is_ar else "Quality")} {_pavg("quality_score")}% · '
                f'{("طبي" if _is_ar else "Medical")} {_pavg("medical_score")}% · '
                f'{("سرعة" if _is_ar else "Speed")} {_perf_speed_avg if _perf_speed_avg is not None else "—"}s · '
                f'{("أخطاء" if _is_ar else "Errors")} {_perf_err_rate if _perf_err_rate is not None else "—"}%'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:#6B7280;font-size:.78rem;margin-bottom:1rem;">'
                f'{("لسا ما ولّدتي أي إيميل بهذي الدورة — النتائج الآلية بتظهر هنا أوتوماتيك" if _is_ar else "No emails generated yet this cycle — automatic scores will appear here")}'
                f'</div>', unsafe_allow_html=True)

        # تنسيق حقل الملاحظات بشكل شفاف مع إطار مثل باقي العناصر
        st.markdown("""<style>
.stTextInput input{
    background:transparent!important;
    color:#E2E8F0!important;
    border:none!important;
    box-shadow:none!important;
}
.stTextInput input::placeholder{color:#6B7280!important;}
.stTextInput>div>div,
.stTextInput div[data-baseweb="base-input"]{
    background:transparent!important;
    background-color:transparent!important;
    border:none!important;
    box-shadow:none!important;
}
.stTextInput div[data-baseweb="input"]{
    background:rgba(15,23,42,.5)!important;
    border:1px solid rgba(255,255,255,.15)!important;
    border-radius:8px!important;
    box-shadow:none!important;
}
</style>""", unsafe_allow_html=True)

        st.markdown(f'<div style="margin-bottom:.2rem;"><span style="font-weight:700;color:#E2E8F0;">⭐ '
                    f'{("الانطباع العام" if _is_ar else "Overall Impression")}</span>'
                    f'<span style="color:#6B7280;font-size:.8rem;margin-right:.5rem;"> — '
                    f'{("حكمك الشامل عن جودة/واقعية/لغة هذي الدورة بالكامل" if _is_ar else "Your holistic judgement of quality/realism/language for this whole cycle")}</span></div>',
                    unsafe_allow_html=True)
        # NEW: a per-(provider, language) "form version" counter. Bumping it
        # after every successful save changes the slider/note widget keys,
        # which forces Streamlit to reset them to their defaults instead of
        # silently keeping the just-saved values — this was the root cause
        # of accidental double-saves (clicking Save twice resubmitted the
        # same note/rating because the widgets never visibly reset).
        _form_ver_key = f"cycle_form_version_{cur_prov}_{lang_for_run}"
        _form_ver = st.session_state.get(_form_ver_key, 0)

        overall_rating = st.select_slider(
            label="overall",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: f"{'⭐'*x}{'☆'*(5-x)} ({x}/5)",
            key=f"rating_overall_{cur_prov}_{lang_for_run}_{_form_ver}",
            label_visibility="collapsed"
        )
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)

        col_note, _ = st.columns([2,1])
        with col_note:
            note = st.text_input(T('note_label'), placeholder=T('note_placeholder'),
                                 key=f"note_{cur_prov}_{lang_for_run}_{_form_ver}")

        def _save_cycle_rating(overall_val, note_text):
            snap = snapshot_and_clear_pending_cycle(cur_prov, lang_for_run)
            perf_snap = snapshot_and_clear_pending_perf(cur_prov, lang_for_run)
            record = {
                "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                "provider": cur_prov,
                "language": lang_for_run,
                "overall": overall_val,
                "auto_difficulty": snap["difficulty_score"],
                "auto_arabic": snap["arabic_score"],
                "auto_quality": snap["quality_score"],
                "auto_medical": snap["medical_score"],
                "n_auto_emails": snap["n_emails"],
                "avg_speed": perf_snap["avg_speed"],
                "json_rate": perf_snap["json_rate"],
                "error_rate": perf_snap["error_rate"],
                "diversity": perf_snap["diversity"],
                "note": note_text or "",
            }
            save_run(record)
            # Bump the form version so the slider/note reset to defaults on
            # the next render, making it visually obvious the save went
            # through and preventing a second click from resubmitting it.
            st.session_state[_form_ver_key] = _form_ver + 1

        if st.button(T('save_btn'), use_container_width=True):
            _save_cycle_rating(overall_rating, note)
            st.success(f"{T('ratings_saved')} {prov_label} ({lang_for_run}) — {_n_done_lang+1}/{_target_lang}")
            st.rerun()

        # Quick thumbs up/down shortcut — still one record per FULL cycle
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:.4rem;">{T("quick_rating")}</div>', unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button(T('good_btn'), use_container_width=True, key="quick_good"):
                _save_cycle_rating(4, note)
                st.success(T('saved_45'))
                st.rerun()
        with qc2:
            if st.button(T('avg_btn'), use_container_width=True, key="quick_avg"):
                _save_cycle_rating(3, note)
                st.success(T('saved_35'))
                st.rerun()
        with qc3:
            if st.button(T('poor_btn'), use_container_width=True, key="quick_bad"):
                _save_cycle_rating(2, note)
                st.success(T('saved_25'))
                st.rerun()

        # Undo last entry for this provider+language, in case of a mis-click
        # (also restores the snapshot back into the pending bucket so no
        # automatic data is lost).
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        if st.button("↩️ " + ("تراجع عن آخر دورة محفوظة لهذا المزوّد/اللغة" if _is_ar else "Undo last saved cycle for this provider/language"),
                     use_container_width=True, key="undo_last_run"):
            all_runs = load_runs()
            for i in range(len(all_runs) - 1, -1, -1):
                if all_runs[i].get("provider") == cur_prov and all_runs[i].get("language") == lang_for_run:
                    removed_record = all_runs.pop(i)
                    try:
                        with open(_RUNS_FILE_PATH, "w", encoding="utf-8") as f:
                            json.dump(all_runs, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    delete_run_from_gsheet(removed_record)
                    st.success("✅ " + ("تم حذف آخر دورة" if _is_ar else "Last cycle removed"))
                    st.rerun()
                    break

        # History summary for this provider+language
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        my_runs = [r for r in load_runs() if r.get("provider")==cur_prov and r.get("language")==lang_for_run]
        if my_runs:
            overall_vals = [r.get("overall") for r in my_runs if r.get("overall") is not None]
            if overall_vals:
                a = round(sum(overall_vals)/len(overall_vals), 1)
                st.markdown(f'<div style="font-weight:700;color:#D1FAE5;margin-bottom:.2rem;">{T("rating_history")} {prov_label} ({lang_for_run})</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#9CA3AF;font-size:.82rem;">{("الانطباع العام" if _is_ar else "Overall")}: {T("avg_label")} {a}/5 ({len(overall_vals)} {("دورة" if _is_ar else "cycles")}) {"⭐"*round(a)}</div>', unsafe_allow_html=True)

        # Same full Excel export, available here too so the researcher
        # doesn't need to switch tabs just to download her results.
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        _all_runs_now = load_runs()
        if _all_runs_now:
            st.download_button(
                "⬇️ " + ("تصدير كل النتائج Excel" if _is_ar else "Export full results (Excel)"),
                data=build_excel_export(),
                file_name="phishing_research_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="export_excel_manual",
            )
        else:
            st.markdown(f'<div dir="{_dir}" style="text-align:{_align};font-size:.8rem;color:#6B7280;">⚠️ ' + ("احفظي تقييمًا واحدًا على الأقل أعلاه لتفعيل التصدير." if _is_ar else "Save at least one rating above to enable export.") + '</div>', unsafe_allow_html=True)
            st.button("⬇️ " + ("تصدير كل النتائج Excel" if _is_ar else "Export full results (Excel)"), use_container_width=True, disabled=True, key="export_excel_manual_disabled")

    with tab4:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1rem;">'
            f'{"آخر 20 خطأ تقني حصل أثناء التوليد (محفوظ على القرص، يبقى حتى بعد تحديث الصفحة)" if _is_ar else "Last 20 technical generation errors (saved to disk — survives page reloads)"}'
            f'</div>', unsafe_allow_html=True)
        disk_log = _load_debug_log()
        session_log = st.session_state.get("_debug_log", [])
        # Merge + dedupe by (stage, ts) while preserving order, disk is the
        # source of truth since it survives reloads; session entries cover
        # the (rare) case a write to disk failed on a read-only host.
        seen = set()
        merged = []
        for entry in disk_log + session_log:
            key = (entry.get("stage"), str(entry.get("ts")), str(entry.get("error"))[:50])
            if key not in seen:
                seen.add(key)
                merged.append(entry)
        debug_log = list(reversed(merged))[:20]
        if not debug_log:
            st.info("لا توجد أخطاء مسجّلة حتى الآن." if _is_ar else "No errors logged yet.")
        else:
            if st.button("🗑️ " + ("تفريغ السجل" if _is_ar else "Clear log"), key="clear_debug_log"):
                st.session_state["_debug_log"] = []
                try:
                    with open(_DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
                        json.dump([], f)
                except Exception:
                    pass
                st.rerun()
            for i, entry in enumerate(debug_log):
                with st.expander(f"#{len(debug_log)-i} — {entry.get('stage','?')}"):
                    st.json(entry.get("error"))






# =============================================================
# FINAL DIVERSITY + DIFFICULTY OVERRIDE — Study 3
# -------------------------------------------------------------
# This block intentionally overrides the earlier prompt/quality
# functions before the Streamlit router runs. It keeps the UI and
# provider code unchanged, but strengthens:
# 1) unbounded learning generation,
# 2) unbounded assessment generation,
# 3) the 9-criterion difficulty contract,
# 4) Arabic/English parity,
# 5) analysis quality and attack-type diversity.
# =============================================================

ADVANCED_BANNED_DOMAIN_WORDS = {
    "secure", "update", "verify", "verification", "login", "reset",
    "password", "urgent", "alert", "account", "emr", "phish", "fake"
}

ATTACK_PLAYBOOK = {
    "clinical": [
        {"attack":"Credential Harvesting", "vector":"external portal link", "persuasion":"routine clinical workflow", "en":"patient-records access review", "ar":"مراجعة وصول السجلات الطبية"},
        {"attack":"Attachment Malware", "vector":"PDF/DOCX/XLSM attachment", "persuasion":"patient safety", "en":"urgent patient report attachment", "ar":"مرفق تقرير مريض عاجل"},
        {"attack":"QR Phishing", "vector":"QR code instruction", "persuasion":"mobile workflow", "en":"mobile clinical checklist QR", "ar":"رمز QR لقائمة تحقق سريرية"},
        {"attack":"Authority Impersonation", "vector":"reply with sensitive data", "persuasion":"medical director authority", "en":"consultant/director request", "ar":"طلب من مدير أو استشاري"},
        {"attack":"MFA / OTP Abuse", "vector":"OTP or MFA confirmation", "persuasion":"access continuity", "en":"MFA confirmation for clinical system", "ar":"تأكيد MFA لنظام سريري"},
        {"attack":"Spear Phishing", "vector":"contextual link or reply", "persuasion":"personalized prior meeting", "en":"follow-up to department meeting", "ar":"متابعة لاجتماع القسم"},
        {"attack":"Legitimate-Looking Compliance Phish", "vector":"policy acknowledgement", "persuasion":"compliance", "en":"infection-control acknowledgement", "ar":"إقرار مكافحة العدوى"},
        {"attack":"Cloud Document Scam", "vector":"shared document", "persuasion":"collaboration", "en":"shared clinical handover document", "ar":"مستند تسليم سريري مشترك"},
    ],
    "admin": [
        {"attack":"Invoice Fraud", "vector":"invoice attachment or portal", "persuasion":"payment deadline", "en":"supplier invoice approval", "ar":"اعتماد فاتورة مورد"},
        {"attack":"BEC / Executive Impersonation", "vector":"reply or bank-transfer instruction", "persuasion":"executive authority", "en":"director payment request", "ar":"طلب دفع من مدير"},
        {"attack":"Payroll Phishing", "vector":"bank-detail update", "persuasion":"salary continuity", "en":"payroll account confirmation", "ar":"تأكيد حساب الرواتب"},
        {"attack":"Benefits Enrollment Scam", "vector":"benefits portal", "persuasion":"reward/opportunity", "en":"employee benefits enrollment", "ar":"التسجيل في مزايا الموظفين"},
        {"attack":"Vendor Portal Scam", "vector":"supplier portal", "persuasion":"contract renewal", "en":"vendor contract renewal", "ar":"تجديد عقد مورد"},
        {"attack":"Shared Document Scam", "vector":"document share", "persuasion":"administrative process", "en":"shared patient-files report", "ar":"تقرير ملفات مرضى مشترك"},
        {"attack":"Insurance Portal Phish", "vector":"coverage verification", "persuasion":"claim processing", "en":"insurance claim validation", "ar":"التحقق من مطالبة تأمين"},
        {"attack":"Meeting Invitation Scam", "vector":"calendar attachment/link", "persuasion":"routine meeting", "en":"accreditation review meeting", "ar":"اجتماع مراجعة الاعتماد"},
    ],
    "it": [
        {"attack":"Credential Harvesting", "vector":"admin portal", "persuasion":"service continuity", "en":"network access review", "ar":"مراجعة وصول الشبكة"},
        {"attack":"MFA Fatigue / OTP Abuse", "vector":"MFA approval request", "persuasion":"security validation", "en":"MFA push validation", "ar":"اعتماد إشعار MFA"},
        {"attack":"Attachment Malware", "vector":"script/XLSM/DOCM attachment", "persuasion":"system repair", "en":"backup verification script", "ar":"سكربت تحقق النسخ الاحتياطي"},
        {"attack":"CIO/CISO Impersonation", "vector":"reply with configuration/data", "persuasion":"executive pressure", "en":"security exception request", "ar":"طلب استثناء أمني"},
        {"attack":"License Renewal Scam", "vector":"renewal portal", "persuasion":"service deadline", "en":"software license renewal", "ar":"تجديد ترخيص برنامج"},
        {"attack":"Cloud Backup Scam", "vector":"cloud console link", "persuasion":"data protection", "en":"cloud backup access review", "ar":"مراجعة وصول النسخ السحابي"},
        {"attack":"QR Phishing", "vector":"QR enrollment", "persuasion":"device enrollment", "en":"device compliance QR", "ar":"رمز امتثال الجهاز"},
        {"attack":"Spear Phishing", "vector":"contextual reply/link", "persuasion":"known incident follow-up", "en":"follow-up to outage ticket", "ar":"متابعة تذكرة عطل"},
    ],
    "other": [
        {"attack":"Mixed Department Phishing", "vector":"fresh mixed vector", "persuasion":"workplace routine", "en":"new hospital workflow", "ar":"إجراء مستشفى جديد"},
        {"attack":"Vendor / HR / IT Scam", "vector":"link, attachment, reply, or QR", "persuasion":"authority or compliance", "en":"cross-department request", "ar":"طلب بين الأقسام"},
        {"attack":"Legitimate-Looking Internal Process", "vector":"contextual workflow", "persuasion":"routine", "en":"ordinary internal process abused", "ar":"استغلال إجراء داخلي عادي"},
    ]
}

LEGITIMATE_PLAYBOOK = {
    "clinical": [
        {"en":"shift schedule update", "ar":"تحديث جدول المناوبات", "sender":"Head Nurse / Clinical Operations"},
        {"en":"infection-control guideline notice", "ar":"إشعار إرشادات مكافحة العدوى", "sender":"Infection Control Team"},
        {"en":"CME workshop invitation", "ar":"دعوة ورشة تعليم طبي مستمر", "sender":"Medical Education"},
        {"en":"lab downtime notice", "ar":"إشعار توقف مؤقت للمختبر", "sender":"Laboratory Department"},
        {"en":"radiology protocol reminder", "ar":"تذكير ببروتوكول الأشعة", "sender":"Radiology Department"},
    ],
    "admin": [
        {"en":"policy acknowledgement with no link", "ar":"إقرار سياسة بدون رابط", "sender":"HR Department"},
        {"en":"procurement order confirmation", "ar":"تأكيد طلب مشتريات", "sender":"Procurement Department"},
        {"en":"insurance training session", "ar":"جلسة تدريب التأمين", "sender":"Training Department"},
        {"en":"meeting room change", "ar":"تغيير قاعة اجتماع", "sender":"Admin Office"},
        {"en":"accreditation visit reminder", "ar":"تذكير بزيارة الاعتماد", "sender":"Quality Department"},
    ],
    "it": [
        {"en":"scheduled maintenance notice", "ar":"إشعار صيانة مجدولة", "sender":"IT Department"},
        {"en":"helpdesk ticket closure", "ar":"إغلاق تذكرة دعم", "sender":"IT Helpdesk"},
        {"en":"approved patching window", "ar":"نافذة تحديث معتمدة", "sender":"Systems Team"},
        {"en":"VPN maintenance reminder", "ar":"تذكير صيانة VPN", "sender":"Network Team"},
        {"en":"security awareness training", "ar":"تدريب توعية أمنية", "sender":"Cybersecurity Office"},
    ],
    "other": [
        {"en":"general hospital announcement", "ar":"إعلان مستشفى عام", "sender":"Hospital Administration"},
        {"en":"training reminder", "ar":"تذكير تدريب", "sender":"Training Department"},
        {"en":"safe operational update", "ar":"تحديث تشغيلي آمن", "sender":"Operations Office"},
    ]
}

# =============================================================
# DIVERSITY EXPANSION — non-classic lure themes
# -------------------------------------------------------------
# EN: Added to EVERY role so learning/assessment examples are not
# always "IT/security"-flavoured. These are realistic lure types
# hospital staff actually receive: MOH-style public offers,
# restaurant/cafe staff discounts, prize/ad campaigns, and
# bank/telecom partner promos — used both as phishing pretexts and
# as safe legitimate look-alikes.
# AR: تُضاف لكل الأدوار حتى لا تكون الأمثلة دومًا بنكهة "تقنية/أمنية".
# هذه أنواع طُعم واقعية يستقبلها الموظفون فعليًا: عروض شبيهة بوزارة
# الصحة، خصومات موظفين بمطاعم/كوفيهات، حملات جوائز/إعلانات، وعروض
# شراكة بنكية/اتصالات — تُستخدم كذريعة تصيد وكذلك كرسائل شرعية آمنة.
# =============================================================
_EXTRA_ATTACK_THEMES = [
    {"attack": "Fake Government/MOH Offer", "vector": "benefits or wellness program link",
     "persuasion": "limited-time government benefit",
     "en": "MOH wellness program enrollment", "ar": "التسجيل في برنامج العافية بوزارة الصحة"},
    {"attack": "Restaurant/Retail Voucher Scam", "vector": "discount voucher link or QR",
     "persuasion": "exclusive staff discount",
     "en": "exclusive staff discount at a partner restaurant/cafe", "ar": "خصم حصري للموظفين في مطعم/كوفي شريك"},
    {"attack": "Prize/Reward Ad Scam", "vector": "promotional ad link or attachment",
     "persuasion": "limited-time prize or reward",
     "en": "hospital anniversary prize draw", "ar": "سحب جوائز بمناسبة ذكرى تأسيس المستشفى"},
    {"attack": "Telecom/Bank Promo Scam", "vector": "promotional offer link",
     "persuasion": "too-good-to-be-true financial offer",
     "en": "telecom/bank partnership offer for staff", "ar": "عرض شراكة بنكية/اتصالات للموظفين"},
]
_EXTRA_LEGIT_THEMES = [
    {"en": "official MOH public-health awareness bulletin", "ar": "نشرة توعوية صحية رسمية من وزارة الصحة",
     "sender": "Ministry of Health Communications"},
    {"en": "hospital cafeteria new menu / Ramadan timing notice", "ar": "إشعار قائمة الكافيتيريا الجديدة / مواعيد رمضان",
     "sender": "Hospital Facilities"},
    {"en": "staff wellness day announcement", "ar": "إعلان يوم العافية للموظفين",
     "sender": "HR Wellness Office"},
    {"en": "National Day / public holiday schedule notice", "ar": "إشعار جدول اليوم الوطني / العطلة الرسمية",
     "sender": "Hospital Administration"},
]
for _rt in list(ATTACK_PLAYBOOK.keys()):
    ATTACK_PLAYBOOK[_rt] = ATTACK_PLAYBOOK[_rt] + _EXTRA_ATTACK_THEMES
for _rt in list(LEGITIMATE_PLAYBOOK.keys()):
    LEGITIMATE_PLAYBOOK[_rt] = LEGITIMATE_PLAYBOOK[_rt] + _EXTRA_LEGIT_THEMES


def _choice_no_recent(items, memory_key, label_getter=lambda x: str(x)):
    recent = st.session_state.get(memory_key, [])
    pool = [x for x in items if label_getter(x) not in recent]
    if not pool:
        pool = list(items)
        recent = []
    item = random.choice(pool)
    label = label_getter(item)
    st.session_state[memory_key] = (recent + [label])[-8:]
    return item

def get_generation_plan(role_type, is_phishing=True, is_ar=False, phase="learn", difficulty="medium"):
    if is_phishing:
        items = ATTACK_PLAYBOOK.get(role_type, ATTACK_PLAYBOOK["other"])
        # --------------------------------------------------------
        # Difficulty-aware vector filtering (Axis 3 — Technical Elements):
        # Easy: NO attachment, NO QR (both strictly forbidden at this level).
        # Medium: NO QR (QR is exclusive to Advanced). Attachment (simple
        # generic PDF) is allowed at Medium, so it stays in the pool.
        # Hard: no exclusion needed here — QR is handled/injected separately
        # as mandatory, and attachment is enforced by the strict addon.
        # This prevents the prompt from telling the model to use a vector
        # ("attachment"/"QR") that a later instruction then forbids.
        # --------------------------------------------------------
        if difficulty == "easy":
            filtered = [x for x in items if "qr" not in x["vector"].lower() and "attachment" not in x["vector"].lower()]
            if filtered:
                items = filtered
        elif difficulty == "medium":
            filtered = [x for x in items if "qr" not in x["vector"].lower()]
            if filtered:
                items = filtered
        # --------------------------------------------------------
        # NEW: cap QR-vector scenarios to AT MOST 1 within any
        # rolling window of 6 learning picks (the QR vector kept
        # appearing too often once we added more QR-themed
        # diversity entries). Assessment (phase="assess") is left
        # uncapped since that pool legitimately needs QR coverage
        # too, just less repetitively within one learning session.
        # --------------------------------------------------------
        if phase == "learn":
            qr_window_key = f"qr_window_{role_type}"
            qr_window = st.session_state.get(qr_window_key, [])
            if sum(qr_window[-6:]) >= 1:
                non_qr_items = [x for x in items if "qr" not in x["vector"].lower()]
                if non_qr_items:
                    items = non_qr_items
        item = _choice_no_recent(
            items,
            f"plan_memory_{phase}_{role_type}_phish",
            lambda x: f"{x['attack']}|{x['vector']}|{x['persuasion']}"
        )
        if phase == "learn":
            qr_window = st.session_state.get(qr_window_key, [])
            qr_window.append(1 if "qr" in item["vector"].lower() else 0)
            st.session_state[qr_window_key] = qr_window[-6:]
        return {
            "attack_type": item["attack"],
            "scenario_seed": item["ar"] if is_ar else item["en"],
            "vector": item["vector"],
            "persuasion": item["persuasion"],
        }
    item = _choice_no_recent(
        LEGITIMATE_PLAYBOOK.get(role_type, LEGITIMATE_PLAYBOOK["other"]),
        f"plan_memory_{phase}_{role_type}_legit",
        lambda x: f"{x['en']}|{x['sender']}"
    )
    return {
        "attack_type": "Legitimate",
        "scenario_seed": item["ar"] if is_ar else item["en"],
        "vector": "safe internal communication",
        "persuasion": "normal workplace communication",
        "sender_hint": item["sender"],
    }

def get_role_unbounded_context(role_type, is_ar=False):
    if is_ar:
        return {
            "clinical": "الدور سريري داخل مستشفى سعودي. يمكن أن تشمل البيئة: طبيب، تمريض، صيدلة، مختبر، أشعة، عيادات، طوارئ، عناية مركزة، سجلات مرضى، PACS، LIS، EMR، تسليم مناوبات، بروتوكولات وزارة الصحة.",
            "admin": "الدور إداري داخل مستشفى سعودي. يمكن أن تشمل البيئة: استقبال، سكرتارية طبية، ملفات مرضى، تأمين، فوترة، مشتريات، موارد بشرية، اعتماد، عقود موردين، رواتب، اجتماعات إدارية.",
            "it": "الدور تقني داخل مستشفى سعودي. يمكن أن تشمل البيئة: VPN، شبكات، خوادم، EMR، نسخ احتياطي، Active Directory، MFA، شهادات، جدار ناري، تراخيص، مكتب مساعدة، أمن سيبراني.",
            "other": "الدور موظف عام داخل مستشفى سعودي. اختر قسمًا منطقيًا جديدًا في كل مرة: سريري، إداري، تقني، تشغيلي، دعم، تدريب، جودة.",
        }.get(role_type, "موظف في مستشفى سعودي.")
    return {
        "clinical": "Clinical role in a Saudi hospital. Possible context: doctor, nurse, pharmacy, laboratory, radiology, clinics, ER, ICU, patient records, PACS, LIS, EMR, handover, MOH clinical protocols.",
        "admin": "Administrative role in a Saudi hospital. Possible context: reception, medical secretary, patient records, insurance, billing, procurement, HR, accreditation, vendor contracts, payroll, administrative meetings.",
        "it": "IT/Informatics role in a Saudi hospital. Possible context: VPN, networks, servers, EMR, backups, Active Directory, MFA, certificates, firewall, licenses, helpdesk, cybersecurity.",
        "other": "General Saudi hospital employee. Choose a fresh logical department each time: clinical, administrative, technical, operations, support, training, quality.",
    }.get(role_type, "Saudi hospital employee.")

def get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=False):
    """Final 9-criterion difficulty engine with strong separation."""
    if is_ar:
        if is_phishing:
            rules = {
                "easy": """
مستوى مبتدئ — يجب أن يكون التصيد واضحًا جدًا عبر 9 معايير:
1) النطاق: مزيف بوضوح، غريب، وغير قريب من الرسمي.
2) الأخطاء: ضع بالضبط خطأين لغويين واضحين في جسم الرسالة.
3) الإلحاح: تهديد مباشر خلال ساعات أو اليوم.
4) التحية: عامة فقط مثل عزيزي الموظف/Dear Staff.
5) المرسل: قسم عام أو اسم غير دقيق.
6) الطلب: طلب واضح لكلمة مرور/بيانات دخول/تحديث حساب.
7) السياق الداخلي: عام وضعيف، بلا تفاصيل شخصية.
8) ناقل الهجوم: رابط واحد فقط أو مرفق واحد فقط.
9) التكتيك: خوف وإلحاح مباشر.
التحليل: اذكر Attack Type وRisk Level، واجعل أول سبب هو الطلب الحساس أو التهديد وليس الدومين دائمًا.
""",
                "medium": """
مستوى متوسط — يجب أن يكون خادعًا لكن قابلًا للاكتشاف عبر 9 معايير:
1) النطاق: مقبول ظاهريًا لكنه غير رسمي عند التدقيق.
2) الأخطاء: خطأ واحد خفيف فقط أو صياغة غريبة بسيطة.
3) الإلحاح: موعد مهني 24–72 ساعة بدون تهديد عدواني.
4) التحية: شبه مخصصة بالاسم الأول أو الدور.
5) المرسل: قسم/موظف يبدو معقولًا لكن توجد فجوة.
6) الطلب: طلب غير معتاد لكنه ممكن في العمل.
7) السياق الداخلي: تفصيل واحد منطقي مثل قسم/نظام/موعد.
8) ناقل الهجوم: رابط أو مرفق أو طلب رد، مع هندسة اجتماعية خفيفة.
9) التكتيك: مصداقية مهنية + ضغط زمني خفيف.
التحليل: نوّع المؤشرات؛ لا تبدأ دائمًا بـ"نطاق غير مألوف".
""",
                "hard": """
مستوى متقدم — يجب أن يبدو شبه شرعي وصعب الاكتشاف عبر 9 معايير:
1) النطاق: قريب بذكاء أو يبدو كخدمة عمل خارجية، لكن ليس رسميًا. لا تستخدم كلمات مكشوفة مثل secure/update/verify/login/reset/password/urgent/alert/account/emr.
2) الأخطاء: صفر أخطاء.
3) الإلحاح: مهذب وخفي، بدون تهديد مباشر أو حروف كبيرة.
4) التحية: مخصصة بالاسم والدور.
5) المرسل: شخص/قسم واقعي جدًا مع توقيع مهني.
6) الطلب: لا تطلب كلمة المرور مباشرة؛ اجعل الخطر في إجراء يبدو طبيعيًا: رد ببيانات، فتح مرفق، اعتماد MFA، مراجعة مستند، QR، أو بوابة.
7) السياق الداخلي: Spear phishing بسياق خفيف مثل اجتماع سابق/مناوبة/تذكرة/مراجعة.
8) ناقل الهجوم: ليس الرابط دائمًا؛ استخدم أحيانًا مرفق/QR/رد/مكالمة/اعتماد.
9) التكتيك: سلطة أو ثقة أو روتين مهني، وليس خوفًا.
التحليل: ركّز على السلوك والسياق والطلب، وليس الدومين فقط.
""",
            }
        else:
            rules = {
                "easy": "رسالة شرعية سهلة: نطاق رسمي فقط hospital.org أو moh.gov.sa، لا روابط خارجية، لا طلب بيانات حساسة، لا تهديد، سياق عمل بسيط وآمن.",
                "medium": "رسالة شرعية متوسطة: نطاق رسمي فقط، موعد أو إجراء طبيعي، تفاصيل عمل واقعية، يمكن ذكر الإنترانت أو رقم تحويلة، ولا يوجد طلب كلمة مرور أو بيانات حساسة.",
                "hard": "رسالة شرعية متقدمة: رسمية ومفصلة وقد تبدو مهمة أو عاجلة مهنيًا، لكنها آمنة تمامًا: نطاق رسمي، لا رابط خارجي مشبوه، لا بيانات حساسة، لا تهديد.",
            }
    else:
        if is_phishing:
            rules = {
                "easy": """
BEGINNER — make phishing obvious through 9 criteria:
1) Domain: clearly fake and not close to official.
2) Spelling: exactly two obvious mistakes in the body.
3) Urgency: direct threat within hours/today.
4) Greeting: generic only: Dear Staff/Dear Team.
5) Sender: vague department or generic name.
6) Request: obvious password/credential/account-update request.
7) Insider context: weak and generic.
8) Vector: one vector only: link OR attachment.
9) Psychology: blunt fear and urgency.
Analysis: include Attack Type and Risk Level. The first indicator should often be the sensitive request/threat, not always the domain.
""",
                "medium": """
INTERMEDIATE — mixed red flags through 9 criteria:
1) Domain: plausible but not official on inspection.
2) Spelling: exactly one subtle mistake OR one slightly odd phrase.
3) Urgency: professional 24–72 hour deadline, no aggressive threat.
4) Greeting: semi-personal: first name or role.
5) Sender: plausible but imperfect.
6) Request: unusual but possible in workplace context.
7) Insider context: one realistic department/system/deadline detail.
8) Vector: link, attachment, or reply request with light social engineering.
9) Psychology: professional credibility plus mild time pressure.
Analysis: vary indicator order; do not always start with Unfamiliar Domain.
""",
                "hard": """
ADVANCED — almost legitimate and hard to detect through 9 criteria:
1) Domain: intelligently close to a workplace/health service, but not official. Do NOT use obvious words: secure/update/verify/login/reset/password/urgent/alert/account/emr.
2) Spelling: zero mistakes.
3) Urgency: polite and subtle only; no threat, no all-caps.
4) Greeting: personalized with name and role/title.
5) Sender: realistic person/department with professional signature.
6) Request: never ask directly for a password; hide risk inside a normal-looking workflow: reply with data, open attachment, approve MFA, review document, scan QR, call number, or use portal.
7) Insider context: light spear-phishing context such as prior meeting, shift, ticket, case review, or department workflow.
8) Vector: not always link; sometimes attachment, QR, reply, phone call, MFA, or shared document.
9) Psychology: authority, trust, routine compliance, or professional responsibility.
Analysis: focus on behavior, context mismatch, role mismatch, data sensitivity, MFA/attachment risk—not domain only.
""",
            }
        else:
            rules = {
                "easy": "Legitimate Beginner: official hospital.org or moh.gov.sa only, no external links, no sensitive request, no threat, simple safe workplace purpose.",
                "medium": "Legitimate Intermediate: official domain only, normal deadline/process, realistic workflow detail, may mention intranet/extension, no credentials/payment request.",
                "hard": "Legitimate Advanced: official and detailed, may look important or professionally urgent, but safe: official domain, no suspicious external link, no sensitive request, no threat.",
            }
    return rules.get(difficulty, rules.get("medium"))

def get_analysis_contract(is_ar=False):
    if is_ar:
        return """
قواعد التحليل التعليمي:
- يجب أن تكون المؤشرات الثلاثة مختلفة فعلًا، ولا تبدأ دائمًا بالنطاق.
- رتب الأولوية هكذا عند وجودها: طلب بيانات/كلمة مرور، بيانات مرضى، اعتماد MFA/OTP، تحويل مالي/فاتورة، مرفق أو QR، انتحال سلطة، عدم توافق الدور، نطاق خارجي، إلحاح، أخطاء.
- يجب تضمين داخل النصوص: نوع الهجوم Attack Type، مستوى الخطورة Risk Level، وملاحظة مرتبطة بسياق الوظيفة.
- لا تجعل "خطأ إملائي" سببًا رئيسيًا إذا يوجد طلب بيانات أو مرفق أو MFA أو تحويل مالي.
"""
    return """
AI analysis rules:
- The 3 indicators must be genuinely different; do not always start with the domain.
- Prioritize indicators in this order when present: credential request, patient data, MFA/OTP abuse, financial/invoice request, attachment/QR risk, authority impersonation, role-context mismatch, external domain, urgency, spelling.
- Include Attack Type, Risk Level, and one role-context note inside the indicator descriptions / why_risky.
- Do not make spelling the main indicator when credentials, patient data, attachment, MFA, or financial risk exists.
"""

def build_prompt(role, index, language):
    is_ar = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    seed = random.randint(100000, 999999)
    plan = get_generation_plan(role_type, is_phishing=True, is_ar=is_ar, phase="learn", difficulty=difficulty)
    st.session_state["_last_learn_vector"] = plan.get("vector", "")
    recipient_email = get_recipient(role, index, language, phase="learn") if role_type != "other" else f"staff.{seed}@hospital.org"
    avoid_topics = get_avoid_list_text(role_type, "learn", is_ar)
    avoid_domains = get_used_domains_text(role_type, "learn", is_ar)
    role_context = get_role_unbounded_context(role_type, is_ar)
    diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=is_ar)
    analysis_contract = get_analysis_contract(is_ar)

    if is_ar:
        return f"""
أنت مولّد أمثلة تعلم للتوعية بالتصيد في بيئة مستشفى سعودي.

المطلوب: ولّد مثال تعلم واحد فقط. يجب أن يكون تصيدًا تعليميًا آمنًا ومحاكى.

خطة التنوع لهذه المحاولة:
- نوع الهجوم: {plan['attack_type']}
- فكرة جديدة مفتوحة: {plan['scenario_seed']}
- ناقل الهجوم المطلوب: {plan['vector']}
- أسلوب الإقناع: {plan['persuasion']}

قواعد إلزامية:
- لا تستخدم قالبًا ثابتًا ولا تعيد صياغة مثال سابق.
- لا تجعل كل شيء رابطًا؛ التزم بناقل الهجوم المطلوب إن كان مرفقًا/QR/ردًا/MFA/مكالمة.
- إذا كان الدور سريري أو إداري، لا تستخدم سيناريو عام لـMFA/OTP/تحديث كلمة المرور/مشاركة مستندات بشكل افتراضي — هذي سيناريوهات خاصة بقسم تقنية المعلومات. ابنِ السيناريو حول مهمة سريرية أو إدارية فعلية (رعاية مرضى، سجلات، فوترة، جدولة...) حتى لو كان ناقل الهجوم لا يزال رابطاً أو طلب بيانات دخول.
- اختر نطاقًا جديدًا واقعي الشكل. في المتقدم لا تستخدم كلمات مكشوفة في النطاق.
- لا تستخدم النص الحرفي suspicious_link داخل body.
- يجب أن يكون التحليل عميقًا ومتنوعًا، وليس Domain/Urgency/Spelling دائمًا.
- أخرج JSON فقط.
- في حقل injected_errors: اكتب قائمة بالكلمات المكتوبة بها خطأ إملائي/نحوي متعمد داخل body فقط (اكتب الكلمة الخاطئة كما وردت بالنص). العدد يجب أن يطابق مستوى الصعوبة بالضبط: سهل=كلمتان، متوسط=كلمة واحدة، صعب=قائمة فاضية.

السياق:
{role_context}
المستلم: {recipient_email}
رقم كسر التكرار: {seed}
{avoid_topics}{avoid_domains}
قواعد الصعوبة:
{diff_rule}
{analysis_contract}

أخرج JSON بهذا الشكل فقط:
{{
  "email_type": "نوع الهجوم المحدد",
  "attack_type": "{plan['attack_type']}",
  "risk_level": "Low/Medium/High/Critical",
  "confidence_score": "0-100%",
  "from": "اسم مرسل واقعي <email@invented-domain>",
  "to": "{recipient_email}",
  "subject": "عنوان الرسالة",
  "attachment": "اسم المرفق أو فراغ",
  "body": "نص البريد الكامل",
  "suspicious_text": "أخطر عبارة في الرسالة",
  "suspicious_link": "الرابط المشبوه أو فراغ",
  "injected_errors": ["الكلمة الخاطئة 1", "الكلمة الخاطئة 2"],
  "indicators": [
    {{"number": 1, "title": "نوع الهجوم / الطلب الخطر", "description": "شرح مرتبط بالسياق الوظيفي"}},
    {{"number": 2, "title": "مؤشر سلوكي أو تقني مختلف", "description": "شرح قصير"}},
    {{"number": 3, "title": "مؤشر ثالث مختلف", "description": "شرح قصير"}}
  ],
  "why_risky": "اذكر نوع الهجوم، مستوى الخطورة، ولماذا يهم هذا الدور داخل المستشفى",
  "learning_tip": "نصيحة عملية قصيرة"
}}
"""
    return f"""
You generate phishing-awareness learning examples for a Saudi hospital.

Task: Generate ONE simulated phishing learning email.

Diversity plan for this attempt:
- Attack Type: {plan['attack_type']}
- Fresh scenario seed: {plan['scenario_seed']}
- Required attack vector: {plan['vector']}
- Persuasion style: {plan['persuasion']}

Mandatory rules:
- Do not use a fixed template or paraphrase a previous example.
- Do not make every example a link; follow the required vector if it is attachment/QR/reply/MFA/phone/shared document.
- If the role is Clinical or Administrative, do NOT default to a generic MFA/OTP/login-credential-reset/document-sharing-portal scenario merely because it's an easy template — those are IT-department scenarios. Build the scenario around an actual clinical or administrative task instead (patient care workflow, records, billing, scheduling, etc.), even if the attack vector still happens to involve a link or credential request.
- Invent a new realistic-looking domain. For Advanced, avoid obvious domain words.
- Never write the literal placeholder suspicious_link inside body.
- The analysis must be varied and deep, not always Domain/Urgency/Spelling.
- Return JSON only.
- In the injected_errors field: list the exact misspelled words you deliberately placed inside body (the word as it appears, misspelled). The count MUST match the difficulty level exactly: Easy=two words, Intermediate=one word, Advanced=empty list.

Context:
{role_context}
Recipient: {recipient_email}
Anti-repeat seed: {seed}
{avoid_topics}{avoid_domains}
Difficulty rules:
{diff_rule}
{analysis_contract}

Return only this JSON structure:
{{
  "email_type": "specific attack type",
  "attack_type": "{plan['attack_type']}",
  "risk_level": "Low/Medium/High/Critical",
  "confidence_score": "0-100%",
  "from": "realistic sender name <email@invented-domain>",
  "to": "{recipient_email}",
  "subject": "email subject",
  "attachment": "filename or empty string",
  "body": "full email body",
  "suspicious_text": "most suspicious phrase",
  "suspicious_link": "suspicious URL or empty string",
  "injected_errors": ["misspelled word 1", "misspelled word 2"],
  "indicators": [
    {{"number": 1, "title": "Attack type / risky request", "description": "role-context explanation"}},
    {{"number": 2, "title": "different behavioral or technical clue", "description": "short explanation"}},
    {{"number": 3, "title": "third different clue", "description": "short explanation"}}
  ],
  "why_risky": "include attack type, risk level, and why it matters for this hospital role",
  "learning_tip": "short practical tip"
}}
"""

def build_assess_prompt(role, index, is_phishing, language):
    is_ar = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    seed = random.randint(100000, 999999)
    plan = get_generation_plan(role_type, is_phishing=is_phishing, is_ar=is_ar, phase="assess", difficulty=difficulty)
    st.session_state["_last_assess_vector"] = plan.get("vector", "")
    recipient_email = get_recipient(role, index, language, phase="assess") if role_type != "other" else f"staff.{seed}@hospital.org"
    suffix = f"assess_{is_phishing}"
    avoid_topics = get_avoid_list_text(role_type, suffix, is_ar)
    avoid_domains = get_used_domains_text(role_type, suffix, is_ar)
    role_context = get_role_unbounded_context(role_type, is_ar)
    diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=is_phishing, is_ar=is_ar)

    if is_ar:
        label = "تصيد" if is_phishing else "شرعي"
        legit_rule = "إذا كانت شرعية: استخدم hospital.org أو moh.gov.sa فقط، ولا تضع طلب بيانات حساسة أو تهديدًا أو رابطًا خارجيًا مشبوهًا."
        return f"""
أنت مولّد أسئلة اختبار للتوعية بالتصيد في بيئة مستشفى سعودي.

المطلوب: ولّد رسالة اختبار واحدة. التصنيف الصحيح يجب أن يكون: {label}.

خطة التنوع:
- النوع/الفكرة: {plan['attack_type']} — {plan['scenario_seed']}
- ناقل الرسالة: {plan['vector']}
- أسلوب الإقناع/السياق: {plan['persuasion']}

قواعد إلزامية:
- لا تستخدم قوالب ثابتة.
- لا تكرر نفس نوع السؤال أو نفس النطاق أو نفس أسلوب التحليل.
- أسئلة الاختبار يجب أن تحتوي مزيجًا واقعيًا: تصيد وروابط ومرفقات وQR وردود وMFA ورسائل شرعية آمنة.
- التفسير يجب أن يشرح التصنيف الصحيح بوضوح، مع ذكر نوع الهجوم أو سبب الشرعية.
- أخرج JSON فقط.
{legit_rule}

السياق:
{role_context}
المستلم: {recipient_email}
رقم كسر التكرار: {seed}
{avoid_topics}{avoid_domains}
قواعد الصعوبة:
{diff_rule}

أخرج JSON بهذا الشكل فقط:
{{
  "is_phishing": {str(is_phishing).lower()},
  "email_type": "نوع الرسالة",
  "attack_type": "{plan['attack_type']}",
  "from": "اسم مرسل واقعي <email@domain>",
  "to": "{recipient_email}",
  "subject": "عنوان الرسالة",
  "attachment": "اسم المرفق أو فراغ",
  "body": "نص البريد الكامل",
  "suspicious_text": "أخطر عبارة أو فراغ إذا شرعي",
  "suspicious_link": "الرابط المشبوه أو فراغ",
  "explanation": "شرح مختصر: هل هو تصيد أو شرعي، وما أقوى سببين للتصنيف"
}}
"""
    label = "PHISHING" if is_phishing else "LEGITIMATE"
    legit_rule = "If legitimate: use hospital.org or moh.gov.sa only; no sensitive-data request, no threat, no suspicious external link."
    return f"""
You generate assessment questions for phishing awareness in a Saudi hospital.

Task: Generate ONE assessment email. Correct label must be: {label}.

Diversity plan:
- Type/idea: {plan['attack_type']} — {plan['scenario_seed']}
- Message vector: {plan['vector']}
- Persuasion/context: {plan['persuasion']}

Mandatory rules:
- Do not use fixed templates.
- Do not repeat the same question type, domain, or analysis style.
- Assessment must include a realistic mix across runs: phishing, links, attachments, QR, reply requests, MFA, and safe legitimate messages.
- Explanation must clearly justify the correct label and mention the attack type or why it is safe.
- Return JSON only.
{legit_rule}

Context:
{role_context}
Recipient: {recipient_email}
Anti-repeat seed: {seed}
{avoid_topics}{avoid_domains}
Difficulty rules:
{diff_rule}

Return only this JSON structure:
{{
  "is_phishing": {str(is_phishing).lower()},
  "email_type": "new email type",
  "attack_type": "{plan['attack_type']}",
  "from": "realistic sender name <email@domain>",
  "to": "{recipient_email}",
  "subject": "email subject",
  "attachment": "filename or empty string",
  "body": "full email body",
  "suspicious_text": "most suspicious phrase or empty string if legitimate",
  "suspicious_link": "suspicious URL or empty string",
  "explanation": "brief explanation: phishing or legitimate, with the two strongest reasons"
}}
"""



# =============================================================
# STRICT ROLE + DIFFICULTY GUARDRAILS (patched)
# -------------------------------------------------------------
# These checks reject AI outputs that drift away from the selected
# job role (clinical/admin/IT) or violate the documented difficulty
# framework before the email is shown to the trainee.
# =============================================================
_ROLE_KEYWORDS = {
    "clinical": re.compile(r"\b(patient|clinical|emr|ehr|doctor|nurse|pharmac|medication|lab|radiology|pacs|icu|er|ward|handover|vitals|diagnostic|prescription|blood bank)\b|مريض|سريري|طبيب|ممرض|صيدل|دواء|مختبر|أشعة|مناوبة|قسم|بنك الدم|سجل طبي", re.I),
    "admin": re.compile(r"\b(invoice|procurement|vendor|supplier|payroll|insurance|billing|appointment|contract|leave|hr|administrative|records office)\b|فاتورة|مورد|مشتريات|رواتب|تأمين|فوترة|مواعيد|عقد|إجازات|إداري", re.I),
    "it": re.compile(r"\b(vpn|server|network|firewall|ssl|certificate|helpdesk|active directory|backup|database|endpoint|software|license|mfa|otp|cyber|it support)\b|شبكة|خادم|جدار ناري|شهادة|دعم تقني|نسخ احتياطي|قاعدة بيانات|ترخيص|تقنية|أمن سيبراني", re.I),
}
_ROLE_FORBIDDEN = {
    "clinical": re.compile(r"\b(payroll|invoice|vendor|procurement|leave balance|hr portal|vpn|ssl certificate|server|helpdesk|software license|records management team|document collaboration team|security team)\b|رواتب|فاتورة|مورد|مشتريات|إجازات|بوابة الموارد|دعم تقني|شبكة|خادم|فريق الأمن|فريق إدارة السجلات", re.I),
    "admin": re.compile(r"\b(emr|lab results|clinical handover|patient vitals|medication order|radiology image|vpn|ssl certificate|server|firewall)\b|تسليم سريري|نتائج مختبر|علامات حيوية|أمر دوائي|أشعة|شبكة|خادم", re.I),
    "it": re.compile(r"\b(lab results|clinical handover|patient vitals|payroll bank|supplier invoice|appointment booking)\b|نتائج مختبر|تسليم سريري|علامات حيوية|فاتورة مورد|رواتب|حجز مواعيد", re.I),
}

def _current_role_type_for_guardrail():
    try:
        role = st.session_state.get("role", "Clinical")
        return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]
    except Exception:
        return "clinical"

def _role_alignment_issues(result):
    role_type = _current_role_type_for_guardrail()
    if role_type == "other":
        return []
    text = " ".join(str(result.get(k, "")) for k in ["email_type", "from", "subject", "body", "attachment", "suspicious_text", "suspicious_link"])
    issues = []
    if not _ROLE_KEYWORDS[role_type].search(text):
        issues.append(f"email content is not clearly aligned with the selected {role_type} role")
    if _ROLE_FORBIDDEN[role_type].search(text):
        issues.append(f"email drifts into a different role instead of the selected {role_type} role")
    return issues

def _difficulty_structure_issues(result, difficulty, is_phishing=True):
    if not is_phishing or not isinstance(result, dict):
        return []
    body = str(result.get("body", ""))
    combined = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    issues = []
    has_qr = bool(re.search(r"\[\s*QR", body, re.I))
    has_attachment = bool(str(result.get("attachment", "")).strip())
    link = str(result.get("suspicious_link", "")).strip()
    if difficulty == "easy":
        if has_qr: issues.append("Beginner/Easy must not contain QR code")
        if has_attachment: issues.append("Beginner/Easy must not contain attachment")
        if link and link not in body: issues.append("Beginner/Easy must show the fake URL visibly in the body")
        if re.search(r"\[[^\]]+\]\(https?://", body): issues.append("Beginner/Easy must use plain visible URL, not a button")
    elif difficulty == "medium":
        if has_qr: issues.append("Intermediate must not contain QR code")
        if re.search(r"act now|today or|immediately or|account closes today|تصرف الآن|اليوم أو|فورًا وإلا", combined, re.I):
            issues.append("Intermediate urgency is too aggressive")
    elif difficulty == "hard":
        if not has_qr: issues.append("Advanced/Hard must include a QR code marker [QR: ...]")
        if re.search(r"https?://", body): issues.append("Advanced/Hard must not expose raw URL in body")
        if not has_attachment: issues.append("Advanced/Hard must include an official named attachment")
        if re.search(r"password|enter your login|provide your credentials|كلمة المرور|بيانات الدخول", combined, re.I):
            issues.append("Advanced/Hard must avoid direct credential requests")
    return issues

def get_generation_quality_issues(result, difficulty, is_phishing=True):
    if not isinstance(result, dict):
        return ["result is not a JSON object"]
    body = (result.get("body") or "") or ""
    subject = (result.get("subject") or "") or ""
    sender = (result.get("from") or "") or ""
    link = (result.get("suspicious_link") or "") or ""
    attach = (result.get("attachment") or "") or ""
    attack_type = (result.get("attack_type") or "") or (result.get("email_type") or "") or ""
    combined = " ".join([body, subject, sender, link, attach, attack_type])
    domains = extract_domains_from_result(result)
    non_official = [d for d in domains if _domain_root(d) not in {"hospital.org", "moh.gov.sa"}]
    issues = []

    if is_phishing:
        has_vector = bool(non_official or attach or re.search(r'\bQR\b|رمز\s*QR|reply|رد\s|OTP|MFA|اتصل|call\s', combined, re.I))
        if not has_vector:
            issues.append("phishing item needs a clear vector: fake domain/link, attachment, QR, reply request, MFA/OTP, or phone request")
        if difficulty == "easy":
            if not _has_generic_greeting(body):
                issues.append("Beginner must use a generic greeting")
            if not re.search(r'password|credential|login|verify|account|bank|IBAN|OTP|MFA|كلمة مرور|بيانات الدخول|تحقق|حساب|آيبان|رمز', combined, re.I):
                issues.append("Beginner needs an obvious sensitive request")
            if not re.search(r'urgent|immediately|today|suspended|terminated|locked|عاجل|فورًا|اليوم|تعليق|إيقاف', combined, re.I):
                issues.append("Beginner needs obvious urgency/threat")
        elif difficulty == "medium":
            if _contains_long_all_caps(combined):
                issues.append("Intermediate must not use aggressive all-caps")
            if re.search(r'permanent termination|within 1 hour|act now|account closed|إنهاء دائم|خلال ساعة|تصرف الآن', combined, re.I):
                issues.append("Intermediate threat is too aggressive")
        elif difficulty == "hard":
            if _has_generic_greeting(body):
                issues.append("Advanced must use a personalized greeting")
            if _contains_long_all_caps(combined):
                issues.append("Advanced must not contain all-caps")
            if re.search(r'act now|failure to comply|account will be closed|enter your password|full credentials|تصرف الآن|سيتم إغلاق|أدخل كلمة المرور|بيانات الدخول كاملة', combined, re.I):
                issues.append("Advanced contains beginner-style direct threat or password request")
            if any(_domain_has_obvious_advanced_word(d) for d in non_official):
                issues.append("Advanced fake domain is too obvious")
            # Advanced should not be only an obvious link-verification email unless the context is spear-phishing.
            if re.search(r'verify your account|account verification|تحقق من حسابك|تأكيد الحساب', combined, re.I) and not re.search(r'meeting|ticket|case|shift|review|اعتماد|اجتماع|تذكرة|مناوبة|مراجعة', combined, re.I):
                issues.append("Advanced needs contextual spear-phishing, not generic account verification")
    else:
        bad = [d for d in domains if _domain_root(d) not in {"hospital.org", "moh.gov.sa"}]
        if bad:
            issues.append("Legitimate item must not contain external or fake domains")
        if re.search(r'password|credential|verify your account|enter your login|OTP|MFA|كلمة مرور|بيانات الدخول|تحقق من حسابك|رمز تحقق', combined, re.I):
            issues.append("Legitimate item must not ask for credentials/MFA/account verification")
        if re.search(r'suspended|terminated|locked|account closed|تعليق|إيقاف|إغلاق الحساب', combined, re.I):
            issues.append("Legitimate item must not threaten account suspension")
    return issues

def get_system_prompt():
    difficulty = st.session_state.get("difficulty", "medium")
    role = st.session_state.get("role", "Clinical")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    role_context = get_role_unbounded_context(role_type, False)
    return f"""
You are a cybersecurity training content generator for a Saudi healthcare phishing-awareness study.
Role context: {role_context}
Current difficulty: {difficulty}.

Hard rules:
- Return valid JSON only.
- Keep all content safe, simulated, and educational.
- Do not use fixed templates or memorized demonstration domains.
- Generate varied healthcare workplace scenarios across links, attachments, QR, reply requests, MFA/OTP, phone requests, shared documents, invoice fraud, BEC, spear phishing, and legitimate internal messages.
- For legitimate emails, use only hospital.org or moh.gov.sa and include no sensitive-data request, no threat, and no suspicious external link.
- For phishing emails, use fake domains or simulated unsafe requests only. Do not impersonate a real private organization.
- Beginner, Intermediate, and Advanced must be visibly different.
- Arabic and English must have equal depth and quality.
""".strip()



# =============================================================
# STUDY 3 — FINAL 12-NOTE CONTENT OVERRIDE
# -------------------------------------------------------------
# Implements the final review notes collected from the five stages:
# 1) Tie the AI analysis directly to the generated attack type.
# 2) Keep Beginner / Intermediate / Advanced visibly different.
# 3) Keep assessment content aligned with the selected difficulty.
# 4) Improve Arabic and English parity.
# 5) Avoid repeated fixed templates and repeated example slots.
# 6) Diversify attack vectors: link, attachment, QR, reply, MFA/OTP, phone, shared document.
# 7) Diversify senders, recipients, departments, domains, and scenarios.
# 8) Do not over-focus analysis on Domain/Urgency/Spelling.
# 9) Prioritize sensitive requests and role-context risk in the analysis.
# 10) Keep legitimate assessment emails safe and clearly official.
# 11) Keep advanced phishing realistic and less obvious.
# 12) Reduce provider friction by using strict JSON, concise outputs, retries, and quality checks.
# =============================================================

ROLE_ATTACK_HINTS = {
    "clinical": "patient safety, EMR access, clinical workflow, handover, lab/radiology/pharmacy systems, or patient-data confidentiality",
    "admin": "billing, insurance, HR, procurement, patient-file administration, supplier payments, or executive workflow",
    "it": "network access, server administration, MFA, VPN, backups, security policy, identity management, or system availability",
    "other": "a realistic hospital workflow matching the selected mixed-role scenario",
}

ROLE_ATTACK_HINTS_AR = {
    "clinical": "سلامة المرضى، الوصول للسجلات الطبية، سير العمل السريري، التسليم، المختبر/الأشعة/الصيدلية، أو سرية بيانات المرضى",
    "admin": "الفوترة، التأمين، الموارد البشرية، المشتريات، ملفات المرضى، مدفوعات الموردين، أو سير العمل الإداري",
    "it": "وصول الشبكة، إدارة الخوادم، MFA، VPN، النسخ الاحتياطي، سياسات الأمن، إدارة الهوية، أو استمرارية الأنظمة",
    "other": "سير عمل مستشفى واقعي مطابق للسيناريو المختلط المختار",
}

ATTACK_ANALYSIS_PRIORITIES = [
    (r"OTP|MFA|رمز|مصادقة", "MFA / OTP Abuse", "طلب رمز تحقق أو اعتماد MFA"),
    (r"password|credential|login|username|IBAN|bank|كلمة مرور|بيانات الدخول|اسم المستخدم|آيبان|حساب بنكي", "Credential / Sensitive Data Request", "طلب بيانات دخول أو بيانات حساسة"),
    (r"invoice|payment|supplier|SAR|فاتورة|دفع|مورد|ريال", "Invoice / Payment Fraud", "احتيال فاتورة أو دفع"),
    (r"QR|رمز QR|scan", "QR Phishing", "تصيد عبر رمز QR"),
    (r"\.pdf|\.docx|\.xlsx|attachment|attached|مرفق|ملف", "Attachment-Based Attack", "هجوم عبر مرفق"),
    (r"reply|respond|send me|رد|أرسل", "Reply-Based Social Engineering", "هندسة اجتماعية عبر الرد"),
    (r"call|phone|extension|اتصل|هاتف|تحويلة", "Phone / Callback Phishing", "تصيد عبر اتصال أو رقم بديل"),
    (r"shared|document|drive|portal|مستند|مشترك|بوابة", "Cloud / Portal Phishing", "تصيد عبر مستند أو بوابة"),
]

def infer_attack_type_from_content(result, is_ar=False):
    combined = " ".join(str(result.get(k, "")) for k in ["email_type", "attack_type", "subject", "attachment", "body", "suspicious_text", "suspicious_link"])
    for pattern, en, ar in ATTACK_ANALYSIS_PRIORITIES:
        if re.search(pattern, combined, re.I):
            return ar if is_ar else en
    existing = result.get("attack_type") or result.get("email_type")
    return existing or ("تصيد موجه" if is_ar else "Spear Phishing")

def _place_link_in_body(body, link):
    """
    GUARANTEED link placement — removes link from anywhere it appears
    (including after the signature) and re-inserts it either:
      (a) immediately after the last cue sentence ('click the link below', etc.)
          that appears before the signature, OR
      (b) immediately before the closing signature block.
    This means the link can NEVER end up after 'Best regards / Regards / Sincerely'.
    """
    body = (body or "").strip()
    link = (link or "").strip()
    if not link or not body:
        return body

    # ── 1. Remove all occurrences of the link from the body ─────────────────
    lines = body.split("\n")
    cleaned = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == link or (link in stripped and len(stripped) <= len(link) + 8):
            # eat a trailing blank line too
            if i + 1 < len(lines) and not lines[i + 1].strip():
                i += 1
        else:
            cleaned.append(lines[i])
        i += 1
    body = "\n".join(cleaned).strip()

    # ── 2. Find the signature block start ───────────────────────────────────
    _SIG_RE = re.compile(
        r"^(regards|best regards|kind regards|warm regards|thank you|sincerely|"
        r"respectfully|yours truly|yours sincerely|best wishes|"
        r"شكرا|شكراً|مع تحياتي|تحياتي|أطيب التحيات|وتفضلوا)",
        re.I,
    )
    lines = body.split("\n")
    sig_start = None
    for idx, line in enumerate(lines):
        if _SIG_RE.match(line.strip()):
            sig_start = idx
            break

    body_above = lines[:sig_start] if sig_start is not None else lines[:]
    sig_block  = lines[sig_start:]  if sig_start is not None else []

    # ── Early-exit: link is already present above the signature ─────────────
    # This happens when the AI writes the link inline in a sentence
    # (e.g. "visit this link: http://..."). Step 1 only removes standalone
    # link lines, so inline occurrences survive. If the link is already
    # above the signature, we must NOT insert a second copy.
    if link in "\n".join(body_above):
        merged = body_above[:]
        if sig_block:
            if merged and merged[-1].strip():
                merged.append("")
            merged.extend(sig_block)
        return "\n".join(merged).strip()
    _CUE_RE = re.compile(
        r"(link below|following link|the link|click here|click the|visit the|"
        r"access.*link|download.*link|open.*link|survey.*link|"
        r"link:|here:|click:|visit:|download:|scan:|submit:|complete:|"
        r"الرابط|اضغط|افتح|امسح|انقر|حمّل|زيارة|هنا:|أكمل)",
        re.I,
    )
    cue_idx = None
    for idx in range(len(body_above) - 1, -1, -1):
        if _CUE_RE.search(body_above[idx]):
            cue_idx = idx
            break

    # ── 4. Insert the link ───────────────────────────────────────────────────
    if cue_idx is not None:
        insert_at = cue_idx + 1
        # Skip any blank lines that immediately follow the cue
        while insert_at < len(body_above) and not body_above[insert_at].strip():
            insert_at += 1
        body_above.insert(insert_at, link)
    else:
        # No cue sentence — append just before the signature
        body_above.append(link)

    # ── 5. Reassemble with exactly one blank line before signature ───────────
    merged = body_above[:]
    if sig_block:
        if merged and merged[-1].strip():
            merged.append("")
        merged.extend(sig_block)

    return "\n".join(merged).strip()


# Backward-compatible aliases so every existing call site works unchanged.
def _insert_before_signature(body, marker):
    return _place_link_in_body(body, marker)

_SALUTATION_LINE_RE = re.compile(
    r"^(regards|best regards|kind regards|warm regards|thank you|sincerely|"
    r"respectfully|شكرا|شكراً|مع تحياتي|تحياتي|أطيب التحيات|وتفضلوا بقبول)",
    re.I,
)

def _reposition_trailing_lone_link(body, link):
    return _place_link_in_body(body, link)

def _enforce_attack_vector(result, vector):
    """Many providers (Groq/Llama especially) acknowledge the requested
    attack vector in their reasoning but then default to inventing an
    attachment filename regardless, or skip writing the required
    [QR: ...] / [Label](url) marker in the body. This made every example
    look the same (always an attachment) even when the diversity plan
    asked for a link, QR code, or button. We post-process the raw result
    against the vector that was actually requested for this generation:
      - If the vector is NOT attachment-related, strip any attachment
        the model invented anyway.
      - If the vector IS attachment-related but the model gave a link
        instead, drop the stray link/QR markers and keep it attachment-only.
      - If the vector calls for a QR code and no [QR: ...] marker exists
        in the body, synthesize one.
      - If the vector calls for a link/button/portal and neither a QR nor
        a [Label](url) marker exists, synthesize a button.
    """
    if not isinstance(result, dict) or not vector:
        return result
    v = vector.lower()
    body = result.get("body")
    body = body if isinstance(body, str) else ""
    has_qr_marker  = bool(re.search(r'\[\s*QR', body, re.I))
    has_btn_marker = bool(re.search(r'\[[^\]]{1,80}\]\s*\(\s*https?://', body))

    wants_attachment = any(k in v for k in ["attachment", "pdf", "docx", "xlsm", "docm", "script"])
    wants_qr         = "qr" in v
    wants_link       = (not wants_attachment and not wants_qr and
                        any(k in v for k in ["link", "portal", "document", "url", "console", "enrollment", "share"]))

    def _as_str(v):
        return v if isinstance(v, str) else ("" if v is None else str(v))

    if not wants_attachment and _as_str(result.get("attachment")).strip():
        result["attachment"] = ""

    if wants_attachment and (has_qr_marker or has_btn_marker):
        # Vector says attachment but model produced a link/QR marker instead —
        # strip the markers (keep the plain sentence) so the email stays
        # attachment-only and doesn't show two vectors at once.
        body = re.sub(r'\[\s*QR(?:\s*Code)?\s*:?\s*[^\]]*\]', '', body, flags=re.I)
        body = re.sub(r'\[([^\]]{1,80})\]\s*\(\s*https?://[^\)\s]+\s*\)', r'\1', body)
        result["body"] = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body).strip()
        if not _as_str(result.get("attachment")).strip():
            if "docx" in v or "docm" in v:
                ext = ".docm" if "docm" in v else ".docx"
                stem = random.choice(["Patient_Report", "Handover_Notes", "Policy_Update", "Meeting_Minutes"])
            elif "xlsm" in v:
                ext, stem = ".xlsm", random.choice(["Backup_Verification", "Audit_Sheet", "Schedule_Update"])
            elif "script" in v:
                ext, stem = ".zip", random.choice(["Verification_Tool", "System_Update", "Backup_Script"])
            else:
                ext, stem = ".pdf", random.choice(["Patient_Report", "Invoice", "Policy_Notice", "Schedule_Update"])
            result["attachment"] = f"{stem}{ext}"

    elif wants_qr and not has_qr_marker:
        link = _as_str(result.get("suspicious_link")).strip() or "https://example-training-only.invalid/verify"
        result["suspicious_link"] = link
        result["body"] = _insert_before_signature(body, "[QR: Scan to continue]")

    elif wants_link and not has_qr_marker and not has_btn_marker:
        link = _as_str(result.get("suspicious_link")).strip() or "https://example-training-only.invalid/verify"
        result["suspicious_link"] = link
        result["body"] = _insert_before_signature(body, "[Open Link](" + link + ")")

    return result

def normalize_learning_analysis(result, role_type, difficulty, is_ar=False):
    """Post-processes model output so analysis is tied to the attack type and role context."""
    if not isinstance(result, dict):
        return result
    result = _recover_from_nested_email_blob(result)
    result = _enforce_attack_vector(result, st.session_state.get("_last_learn_vector", ""))
    # Defensive: trust what the model actually wrote over the requested
    # language flag. If the body/subject came back in Arabic script despite
    # English being requested (or vice versa), all of OUR appended sentences
    # below must match the model's actual language — otherwise we end up
    # gluing an English clause onto an Arabic sentence (or the reverse).
    is_ar = _detect_is_ar(result, is_ar)
    attack_type = result.get("attack_type") or infer_attack_type_from_content(result, is_ar)
    result["attack_type"] = attack_type
    if not result.get("email_type"):
        result["email_type"] = attack_type
    risk = result.get("risk_level") or ("Medium" if difficulty == "medium" else ("High" if difficulty == "hard" else "Low"))
    result["risk_level"] = risk
    role_hint = (ROLE_ATTACK_HINTS_AR if is_ar else ROLE_ATTACK_HINTS).get(role_type, ROLE_ATTACK_HINTS["other"])

    indicators = result.get("indicators")
    if not isinstance(indicators, list):
        indicators = []
    # Defensive: some providers occasionally return indicator items as plain
    # strings instead of {"number","title","description"} dicts, which
    # crashes any .get() call on them downstream. Normalize every item to a
    # dict shape before anything else touches it.
    indicators = [
        it if isinstance(it, dict) else {"number": i+1, "title": str(it or ""), "description": ""}
        for i, it in enumerate(indicators)
    ]
    while len(indicators) < 3:
        indicators.append({"number": len(indicators)+1, "title": "", "description": ""})

    # Indicator 1 must connect the attack type to the risky action — but we
    # only OVERWRITE the model's own wording when it's missing/empty or is a
    # generic placeholder. Otherwise we keep the model's real (varied,
    # non-repeating) analysis and merely make sure attack_type is mentioned
    # somewhere in it. Forcing the exact same template sentence every time
    # (regardless of provider/example) was the root cause of every "AI Tutor
    # Analysis" panel reading identically across different examples.
    def _is_placeholder(desc):
        d = (desc or "").strip()
        return (not d) or len(d) < 25

    if is_ar:
        d0 = (indicators[0].get("description") or "").strip()
        if _is_placeholder(d0):
            indicators[0] = {
                "number": 1,
                "title": indicators[0].get("title") or f"نوع الهجوم: {attack_type}",
                "description": f"الخطر الأساسي هنا مرتبط بـ {attack_type} داخل سياق {role_hint}."
            }
        elif attack_type not in d0:
            indicators[0]["description"] = f"{d0} (هذا يرتبط مباشرة بنوع هجوم {attack_type}.)"
        if _is_placeholder(indicators[1].get("description")) or re.search(r"^النطاق$|^Domain$", indicators[1].get("title", "").strip(), re.I):
            indicators[1] = {"number": 2, "title": indicators[1].get("title") or "طلب أو سلوك غير معتاد", "description": indicators[1].get("description") or "الرسالة تطلب إجراءً لا يتم عادة عبر بريد عادي في بيئة المستشفى."}
        if _is_placeholder(indicators[2].get("description")) or re.search(r"^إملاء$|^Spelling$", indicators[2].get("title", "").strip(), re.I):
            indicators[2] = {"number": 3, "title": indicators[2].get("title") or "عدم توافق السياق أو القناة", "description": indicators[2].get("description") or "القناة أو المرسل لا يطابقان طريقة التعامل الرسمية مع هذا النوع من الطلبات."}
        wr = (result.get("why_risky") or "").strip()
        if _is_placeholder(wr):
            wr = f"هذه رسالة {attack_type} بمستوى خطورة {risk}. قد تؤثر على {role_hint} إذا تم تنفيذ الطلب دون تحقق."
        elif attack_type not in wr:
            wr = f"{wr} (هذه رسالة {attack_type}.)"
        result["why_risky"] = wr
        tip = (result.get("learning_tip") or "").strip()
        if not tip:
            tip = "تحقق من الطلب عبر قناة المستشفى الرسمية قبل فتح رابط أو مرفق أو مشاركة أي بيانات."
        result["learning_tip"] = tip
    else:
        d0 = (indicators[0].get("description") or "").strip()
        if _is_placeholder(d0):
            indicators[0] = {
                "number": 1,
                "title": indicators[0].get("title") or f"Attack Type: {attack_type}",
                "description": f"The main risk is {attack_type} in a hospital role involving {role_hint}."
            }
        elif attack_type not in d0:
            indicators[0]["description"] = f"{d0} (This directly ties to the {attack_type} attack pattern.)"
        if _is_placeholder(indicators[1].get("description")) or re.search(r"^Domain$", indicators[1].get("title", "").strip(), re.I):
            indicators[1] = {"number": 2, "title": indicators[1].get("title") or "Unusual request or workflow", "description": indicators[1].get("description") or "The message asks for an action that should normally use an official hospital channel."}
        if _is_placeholder(indicators[2].get("description")) or re.search(r"^Spelling$", indicators[2].get("title", "").strip(), re.I):
            indicators[2] = {"number": 3, "title": indicators[2].get("title") or "Role-context or channel mismatch", "description": indicators[2].get("description") or "The sender or channel does not match how this workplace process should be handled."}
        wr = (result.get("why_risky") or "").strip()
        if _is_placeholder(wr):
            wr = f"This is a {attack_type} email with a {risk} risk level. It can affect {role_hint} if the recipient acts without verification."
        elif attack_type not in wr:
            wr = f"{wr} (This is a {attack_type} email.)"
        result["why_risky"] = wr
        tip = (result.get("learning_tip") or "").strip()
        if not tip:
            tip = "Verify the request through an official hospital channel before opening links, attachments, QR codes, or sharing data."
        result["learning_tip"] = tip

    # Keep exactly three indicators and correct numbering.
    result["indicators"] = [{**indicators[i], "number": i+1} for i in range(3)]
    return result

def clear_gsheet_data():
    """Wipe all data rows (keeping headers) from both synced tabs. Used by
    'Reset All Metrics' — now that load_runs() merges in the durable
    Google Sheet copy, a local-only reset would otherwise be silently
    undone on the next page load as old rows get pulled back in."""
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    if not client or not sheet_id:
        return
    for tab_name in ["Cycle Ratings", "Auto Metrics"]:
        try:
            sheet = client.open_by_key(sheet_id)
            ws = sheet.worksheet(tab_name)
            n_rows = ws.row_count
            if n_rows > 1:
                ws.delete_rows(2, n_rows)
        except Exception:
            pass
    st.session_state.pop("_gsheet_runs_cache", None)

def delete_run_from_gsheet(record):
    """Remove the matching row (by timestamp+provider+language) from the
    'Cycle Ratings' tab. Used by the Undo button so a removed cycle
    doesn't silently reappear on the next load_runs() merge — without
    this, Undo would only ever be temporary since Google Sheets is the
    durable source of truth and gets re-merged in on every page load."""
    client = _get_gsheet_client()
    sheet_id = _get_gsheet_id()
    if not client or not sheet_id or not record:
        return
    try:
        sheet = client.open_by_key(sheet_id)
        ws = sheet.worksheet("Cycle Ratings")
        all_vals = ws.get_all_values()
        if not all_vals:
            return
        headers = all_vals[0]
        try:
            ts_i = headers.index("timestamp")
            prov_i = headers.index("provider")
            lang_i = headers.index("language")
        except ValueError:
            return
        target = (str(record.get("timestamp", "")), str(record.get("provider", "")), str(record.get("language", "")))
        for row_idx in range(len(all_vals) - 1, 0, -1):
            row = all_vals[row_idx]
            if len(row) > max(ts_i, prov_i, lang_i):
                if (row[ts_i], row[prov_i], row[lang_i]) == target:
                    ws.delete_rows(row_idx + 1)  # +1: sheet rows are 1-indexed
                    break
        # Invalidate the cached pull so the next load_runs() reflects the deletion.
        st.session_state.pop("_gsheet_runs_cache", None)
    except Exception:
        pass

def _is_nonempty_str(v):
    return isinstance(v, str) and bool(v.strip())

def _detect_is_ar(result, fallback=False):
    """Decide the real language of a generated email from its own prose,
    ignoring URLs and our own [QR:...]/[Label](url) markup — those always
    contain Latin characters (and English words like "Button"/"QR") even
    inside a fully Arabic email, which previously made the script-based
    language check always see "mixed" content and silently fall back to
    the (unreliable) caller-provided flag. Stripping markup first means a
    genuinely Arabic email is correctly detected as Arabic even though its
    button/link markup is in Latin script."""
    subject = result.get("subject") or ""
    body = result.get("body") or ""
    sample = f"{subject} {body}"
    sample = re.sub(r"https?://\S+", " ", sample)
    sample = re.sub(r"\[[^\]]{1,80}\](?:\s*\([^)]*\))?", " ", sample)
    sample = re.sub(r"\b(Button|QR|Open|Link|Document|pdf|docx|xlsx|xlsm|docm)\b", " ", sample, flags=re.I)
    has_ar = bool(re.search(r"[\u0600-\u06FF]", sample))
    has_lat = bool(re.search(r"[A-Za-z]{4,}", sample))
    if has_ar and not has_lat:
        return True
    if has_lat and not has_ar:
        return False
    return fallback

def _is_substantial_str(v, min_len=15):
    return isinstance(v, str) and len(v.strip()) >= min_len

_GENERIC_HOSPITAL_NAME = {"en": "Riyadh Specialist Hospital", "ar": "مستشفى الرياض التخصصي"}

def _resolve_leftover_placeholders(result, is_ar=False):
    """Some providers occasionally leave an unfilled template placeholder
    inside the body instead of inventing real content, e.g. literally
    writing "[Hospital Name]" or "[اسم المستشفى]" instead of a name. These
    survive because they aren't QR/link markers, so the renderer doesn't
    touch them. Replace any leftover bracket placeholder that looks like a
    hospital-name slot with a real, consistent generic name; for any other
    leftover bracket placeholder, just drop the brackets and keep the
    (likely still-readable) text inside, since a literal bracket label
    looks far more obviously broken than blending it into the sentence."""
    body = result.get("body")
    if not isinstance(body, str) or "[" not in body:
        return result
    hospital_name = _GENERIC_HOSPITAL_NAME["ar" if is_ar else "en"]
    body = re.sub(r"\[\s*(?:اسم\s*المستشفى|hospital\s*name)\s*\]", hospital_name, body, flags=re.I)
    # Any other still-unresolved bracket placeholder that isn't one of our
    # own QR/link/button markers (those are handled separately at render
    # time) — drop the brackets but keep the label so it doesn't look like
    # raw template syntax leaked into a "real" email.
    body = re.sub(r"\[([^\]]{1,40})\](?!\s*\()", lambda m: m.group(1) if not re.match(r"^\s*QR", m.group(1), re.I) else m.group(0), body)
    result["body"] = body
    return result

def normalize_assessment_email(result, role_type, difficulty, is_phishing, is_ar=False):
    """Keeps assessment focused on email content + difficulty, without adding learning-analysis sections."""
    if not isinstance(result, dict):
        return result
    result = _recover_from_nested_email_blob(result)
    is_ar = _detect_is_ar(result, is_ar)
    if is_phishing:
        result = _enforce_attack_vector(result, st.session_state.get("_last_assess_vector", ""))
    if "error" not in result:
        core_missing = not (_is_nonempty_str(result.get("from")) and
                             _is_nonempty_str(result.get("subject")) and
                             _is_substantial_str(result.get("body")))
        if core_missing:
            debug_keys = {k: (str(v)[:400] if v else v) for k, v in result.items()}
            return {"error": {"code": "incomplete_content", "message": "incomplete_generation", "debug": debug_keys}}
    result = _resolve_leftover_placeholders(result, is_ar)
    result["is_phishing"] = bool(is_phishing)
    if is_phishing:
        result["attack_type"] = result.get("attack_type") or infer_attack_type_from_content(result, is_ar)
        explanation = (result.get("explanation") or "").strip()
        attack_type = result["attack_type"]
        if attack_type and attack_type not in explanation:
            prefix = (f"التصنيف تصيد لأن نوع الهجوم هو {attack_type}. " if is_ar else f"This is phishing because the attack type is {attack_type}. ")
            result["explanation"] = prefix + explanation
    else:
        result["attack_type"] = "Legitimate"
        # A legitimate assessment item must stay clean.
        result["suspicious_text"] = ""
        result["suspicious_link"] = ""
    return result

# Stronger difficulty contract: short, provider-friendly, and visibly different.
def get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=False):
    if is_ar:
        if is_phishing:
            return {
                "easy": "مبتدئ: تصيد واضح جدًا جدًا. تحية عامة فقط بدون أي اسم شخصي، مرسل عام (فريق/قسم) بدون اسم أو لقب وظيفي لشخص محدد، نطاق مزيف واضح تمامًا لا يشبه أي جهة رسمية حقيقية (ممنوع أي تشابه مع moh.gov.sa أو hospital.org)، طلب مباشر وصريح لكلمة المرور أو بيانات الدخول، رابط نصي مكشوف فقط، ممنوع أي مرفق نهائيًا، تهديد مباشر وصريح، وخطأان إملائيان واضحان إلزاميًا.",
                "medium": "متوسط: تصيد مقنع جزئيًا. تفاصيل عمل واقعية، تحية شبه شخصية بالاسم الأول أو المسمى الوظيفي فقط (ليست عامة وليست كاملة)، نطاق شبه رسمي يشبه الرسمي بفارق بسيط، إلحاح مهني خفيف 24-72 ساعة، مؤشران واضحان فقط، ولا تستخدم تهديدًا مبالغًا أو كلمات إنجليزية كثيرة.",
                "hard": "متقدم: تصيد قريب من الشرعي. تحية بالاسم الكامل + اللقب الوظيفي الدقيق، مرسل بتوقيع مهني كامل، نطاق شبه رسمي بذكاء، لا أخطاء إملائية، لا تهديد مباشر، لا طلب كلمة مرور صريح، يجب أن يتضمن مرفقًا رسميًا مسمّى دائماً. رمز QR اختياري ومتنوّع بين الأمثلة (ضعه فقط إذا يخدم السيناريو، وليس بكل مرة)، واستخدم سياقًا داخليًا واقعيًا (EMR/بروتوكول/رقم تذكرة) واسم زميل أو قسم محدد."
            }.get(difficulty, "متوسط")
        return {
            "easy": "شرعي مبتدئ: رسمي وواضح من hospital.org أو moh.gov.sa، لا رابط خارجي، لا بيانات حساسة، لا تهديد.",
            "medium": "شرعي متوسط: رسمي مع تفاصيل عمل واقعية وموعد طبيعي، قد يذكر الإنترانت أو التحويلة، دون طلب بيانات حساسة.",
            "hard": "شرعي متقدم: يبدو مهمًا ومهنيًا لكنه آمن؛ نطاق رسمي، تفاصيل دقيقة، لا رابط مشبوه، لا تهديد، لا بيانات دخول."
        }.get(difficulty, "شرعي متوسط")
    if is_phishing:
        return {
            "easy": "Beginner: extremely obvious phishing. Generic greeting only, no personal name anywhere; generic sender (a team/department, NOT a named person with a title); domain must be completely and obviously fake, unrelated to any real organization (must NOT resemble moh.gov.sa or hospital.org); direct explicit password/credential request; a plain visible link only; NO attachment of any kind; direct explicit threat; and EXACTLY two obvious spelling/grammar mistakes.",
            "medium": "Intermediate: partly convincing phishing. Realistic workplace detail, semi-personal greeting using first name or job title only (not generic, not full name+title), look-alike domain resembling the real one with a small detectable difference, mild professional urgency of 24-72 hours, only two clear red flags, no extreme threat or heavy all-caps.",
            "hard": "Advanced: near-legitimate phishing. Personalized greeting with full name and precise job title, sender with a complete professional signature, near-official domain, no spelling mistakes, no direct password request, no blunt threat. MUST include an officially named attachment. A QR code is OPTIONAL and should vary across examples (include it only when it fits the scenario, not in every single one), plus a realistic internal context (EMR/clinical protocol/ticket number) and a specific colleague or department name."
        }.get(difficulty, "Intermediate")
    return {
        "easy": "Legitimate Beginner: official hospital.org or moh.gov.sa only, simple safe purpose, no external link, no sensitive request, no threat.",
        "medium": "Legitimate Intermediate: official domain, realistic workplace detail, normal deadline or intranet/extension reference, no credentials/payment request.",
        "hard": "Legitimate Advanced: important and detailed but safe; official domain, no suspicious external link, no sensitive-data request, no threat."
    }.get(difficulty, "Legitimate Intermediate")

# Tighten prompts once more so providers know the analysis must mention the attack type.
_OLD_BUILD_PROMPT_STUDY3 = build_prompt
_OLD_BUILD_ASSESS_PROMPT_STUDY3 = build_assess_prompt

def build_prompt(role, index, language):
    base = _OLD_BUILD_PROMPT_STUDY3(role, index, language)
    is_ar = (language == "Arabic")
    extra = """

قاعدة نهائية مهمة:
- يجب أن يرتبط التحليل مباشرة بنوع الهجوم المكتوب في attack_type.
- أول مؤشر في indicators يجب أن يبدأ بنوع الهجوم، وليس النطاق دائمًا.
- اربط كل سبب بسياق الدور الوظيفي والمستشفى.
- لا تجعل التحليل أطول من اللازم.
""" if is_ar else """

Final important rule:
- The AI analysis must directly connect to the attack_type field.
- The first indicator must start from the attack type, not always the domain.
- Link each reason to the role context and hospital workflow.
- Keep the analysis concise.
"""
    return base + extra

def build_assess_prompt(role, index, is_phishing, language):
    base = _OLD_BUILD_ASSESS_PROMPT_STUDY3(role, index, is_phishing, language)
    is_ar = (language == "Arabic")
    extra = """

قاعدة الاختبار النهائية:
- ركّز فقط على محتوى البريد وتصنيفه وصعوبته.
- لا تضف تحليل تعليمي طويل داخل سؤال الاختبار.
- يجب أن يكون البريد مناسبًا تمامًا لمستوى الصعوبة المختار.
""" if is_ar else """

Final assessment rule:
- Focus only on the email content, correct label, and selected difficulty.
- Do not add long learning analysis inside assessment questions.
- The email must clearly fit the selected difficulty level.
"""
    return base + extra

# Override generators to apply the final normalization after each provider response.
_OLD_GENERATE_EMAIL_STUDY3 = generate_email
_OLD_GENERATE_ASSESS_EMAIL_STUDY3 = generate_assess_email
_OLD_GENERATE_OTHER_EMAIL_STUDY3 = generate_other_email
_OLD_GENERATE_OTHER_ASSESS_EMAIL_STUDY3 = generate_other_assess_email

_DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_log.json")

def _store_debug(stage, err):
    """Keep technical diagnostics out of the user-facing UI; persist them to
    disk (not just session_state) so the admin/debug panel still shows the
    real error even if the page was reloaded or the admin panel was opened
    in a fresh session — which was making the log look empty right when it
    was needed most."""
    entry = {"stage": stage, "error": err, "ts": __import__("time").time()}
    try:
        log = st.session_state.setdefault("_debug_log", [])
        log.append(entry)
        st.session_state["_debug_log"] = log[-20:]
    except Exception:
        pass
    try:
        with open(_DEBUG_LOG_PATH, "r", encoding="utf-8") as f:
            disk_log = json.load(f)
        if not isinstance(disk_log, list):
            disk_log = []
    except Exception:
        disk_log = []
    disk_log.append(entry)
    disk_log = disk_log[-20:]
    try:
        with open(_DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(disk_log, f, ensure_ascii=False, default=str)
    except Exception:
        pass

def _load_debug_log():
    try:
        with open(_DEBUG_LOG_PATH, "r", encoding="utf-8") as f:
            disk_log = json.load(f)
        if isinstance(disk_log, list):
            return disk_log
    except Exception:
        pass
    return []

def _friendly_generation_error(language):
    if language == "Arabic":
        return "تعذّر توليد هذا المثال حالياً بعد عدة محاولات. يرجى الضغط على (حاول مرة أخرى)."
    return "We couldn't generate this example after several attempts. Please tap Try Again."

def _is_incomplete_error(result):
    return isinstance(result, dict) and "error" in result

def _is_fatal_error(result):
    """A handful of error types where retrying is pointless (e.g. missing/
    invalid API key) — fail fast instead of burning 3 attempts."""
    if not (isinstance(result, dict) and "error" in result):
        return False
    err = result.get("error")
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return bool(re.search(r"api[_\s]?key|unauthorized|401|403|invalid x-api-key", msg, re.I))

_ERROR_CATEGORY_PATTERNS = [
    (r"language_mismatch", "wrong-language output, retried"),
    (r"JSON parse error|invalid JSON|Cannot parse JSON", "JSON parsing failed"),
    (r"quality checks", "repeated quality-check rejection"),
    (r"incomplete_generation|empty from/subject/body", "incomplete model output"),
    (r"Unexpected API response", "unexpected API response shape"),
    (r"no text content", "empty model response"),
    (r"503|UNAVAILABLE|overloaded|high demand", "provider overloaded"),
    (r"rate limit|429", "rate limited"),
    (r"timeout|timed out|deadline", "request timed out"),
    (r"Unknown provider", "unknown provider configured"),
]

def _short_hint(result):
    if not (isinstance(result, dict) and "error" in result):
        return ""
    err = result.get("error")
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    msg = str(msg or "").strip()
    if not msg:
        return " [no details returned]"
    for pattern, label in _ERROR_CATEGORY_PATTERNS:
        if re.search(pattern, msg, re.I):
            return f" [{label}]"
    # Always show the real message, truncated to a safe length, instead of
    # a vague placeholder — the debug-log tab has proven unreliable across
    # different hosting setups (ephemeral/multi-instance filesystems), so
    # the on-screen hint is currently the only dependable diagnostic.
    flat = re.sub(r"\s+", " ", msg)[:300]
    return f" [{flat}]"

def _language_clearly_mismatched(result, requested_is_ar):
    """Hard language gate: reject a generation outright if its OWN prose
    clearly does not match the requested language. This is intentionally
    stricter/different from _detect_is_ar (used for cosmetic text-merging
    decisions) — here an ambiguous/mixed case is NOT rejected (it might be
    a legitimately bilingual proper noun), but a CLEAR mismatch is:
      - Arabic was requested but the prose contains no Arabic at all
        (i.e. the model answered fully in English).
      - English was requested but the prose contains any Arabic script
        at all (i.e. the model answered fully or partly in Arabic).
    """
    if not isinstance(result, dict):
        return False
    subject = result.get("subject")
    subject = subject if isinstance(subject, str) else ""
    body = result.get("body")
    body = body if isinstance(body, str) else ""
    sample = f"{subject} {body}"
    sample = re.sub(r"https?://\S+", " ", sample)
    sample = re.sub(r"\[[^\]]{1,80}\](?:\s*\([^)]*\))?", " ", sample)
    sample = re.sub(r"\b(Button|QR|Open|Link|Document|pdf|docx|xlsx|xlsm|docm)\b", " ", sample, flags=re.I)
    has_ar = bool(re.search(r"[\u0600-\u06FF]", sample))
    has_lat = bool(re.search(r"[A-Za-z]{4,}", sample))
    if requested_is_ar:
        return has_lat and not has_ar
    else:
        return has_ar

def generate_email(role, index, language, difficulty="medium"):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    requested_is_ar = (language == "Arabic")
    last_err = None
    for attempt in range(3):
        result = _OLD_GENERATE_EMAIL_STUDY3(role, index, language, difficulty)
        if isinstance(result, dict) and "error" not in result:
            result = normalize_learning_analysis(result, role_type, difficulty, requested_is_ar)
        if isinstance(result, dict) and "error" not in result:
            if _language_clearly_mismatched(result, requested_is_ar):
                _store_debug("generate_email", {"message": "language_mismatch", "requested_ar": requested_is_ar, "body": str(result.get("body"))[:300]})
                last_err = {"error": {"message": "language_mismatch"}}
                continue
            evaluate_and_log_auto_scores(result, difficulty, language, is_phishing=True)
            return result
        if _is_fatal_error(result):
            return result
        if _is_incomplete_error(result):
            _store_debug("generate_email", result.get("error"))
            last_err = result
            continue
        return result
    return {"error": {"message": _friendly_generation_error(language) + _short_hint(last_err)}}

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    requested_is_ar = (language == "Arabic")
    last_err = None
    for attempt in range(3):
        result = _OLD_GENERATE_ASSESS_EMAIL_STUDY3(role, index, is_phishing, language, difficulty)
        if isinstance(result, dict) and "error" not in result:
            result = normalize_assessment_email(result, role_type, difficulty, is_phishing, requested_is_ar)
        if isinstance(result, dict) and "error" not in result:
            if _language_clearly_mismatched(result, requested_is_ar):
                _store_debug("generate_assess_email", {"message": "language_mismatch", "requested_ar": requested_is_ar, "body": str(result.get("body"))[:300]})
                last_err = {"error": {"message": "language_mismatch"}}
                continue
            evaluate_and_log_auto_scores(result, difficulty, language, is_phishing=is_phishing)
            return result
        if _is_fatal_error(result):
            return result
        if _is_incomplete_error(result):
            _store_debug("generate_assess_email", result.get("error"))
            last_err = result
            continue
        return result
    return {"error": {"message": _friendly_generation_error(language) + _short_hint(last_err)}}

def generate_other_email(index, language, difficulty):
    requested_is_ar = (language == "Arabic")
    last_err = None
    for attempt in range(3):
        result = _OLD_GENERATE_OTHER_EMAIL_STUDY3(index, language, difficulty)
        if isinstance(result, dict) and "error" not in result:
            result = normalize_learning_analysis(result, "other", difficulty, requested_is_ar)
        if isinstance(result, dict) and "error" not in result:
            if _language_clearly_mismatched(result, requested_is_ar):
                _store_debug("generate_other_email", {"message": "language_mismatch", "requested_ar": requested_is_ar, "body": str(result.get("body"))[:300]})
                last_err = {"error": {"message": "language_mismatch"}}
                continue
            evaluate_and_log_auto_scores(result, difficulty, language, is_phishing=True)
            return result
        if _is_fatal_error(result):
            return result
        if _is_incomplete_error(result):
            _store_debug("generate_other_email", result.get("error"))
            last_err = result
            continue
        return result
    return {"error": {"message": _friendly_generation_error(language) + _short_hint(last_err)}}

def generate_other_assess_email(index, is_phishing, language, difficulty):
    requested_is_ar = (language == "Arabic")
    last_err = None
    for attempt in range(3):
        result = _OLD_GENERATE_OTHER_ASSESS_EMAIL_STUDY3(index, is_phishing, language, difficulty)
        if isinstance(result, dict) and "error" not in result:
            result = normalize_assessment_email(result, "other", difficulty, is_phishing, requested_is_ar)
        if isinstance(result, dict) and "error" not in result:
            if _language_clearly_mismatched(result, requested_is_ar):
                _store_debug("generate_other_assess_email", {"message": "language_mismatch", "requested_ar": requested_is_ar, "body": str(result.get("body"))[:300]})
                last_err = {"error": {"message": "language_mismatch"}}
                continue
            evaluate_and_log_auto_scores(result, difficulty, language, is_phishing=is_phishing)
            return result
        if _is_fatal_error(result):
            return result
        if _is_incomplete_error(result):
            _store_debug("generate_other_assess_email", result.get("error"))
            last_err = result
            continue
        return result
    return {"error": {"message": _friendly_generation_error(language) + _short_hint(last_err)}}

# Make the final system prompt shorter and more explicit for all providers.
def get_system_prompt():
    difficulty = st.session_state.get("difficulty", "medium")
    role = st.session_state.get("role", "Clinical")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    role_context = get_role_unbounded_context(role_type, False)
    return f"""
You are a safe cybersecurity training content generator for a Saudi healthcare phishing-awareness study.
Role context: {role_context}
Difficulty: {difficulty}.
Return valid JSON only. No markdown.
Learning emails must include attack_type, risk_level, indicators, why_risky, and learning_tip.
Assessment emails must stay concise and contain only the assessment JSON fields.
Phishing content must be simulated, educational, and use invented non-real domains.
Legitimate content must use hospital.org or moh.gov.sa only and must not request credentials, payment, MFA, OTP, bank details, or urgent account verification.
Beginner, Intermediate, and Advanced must be visibly different.
Analysis must mention the attack type and role-context risk, not only Domain/Urgency/Spelling.
Arabic and English must have equal depth and quality.
""".strip()

# =============================================================
# END FINAL DIVERSITY + DIFFICULTY OVERRIDE
# =============================================================

# ══════════════════════════════════════════════════════════════


# =============================================================
# FINAL PATCH — Stability, diversity, and analysis-quality fixes
# =============================================================
# Notes covered:
# - Stronger provider retries for Gemini 503/high-demand and slow providers.
# - Claude/Gemini get shorter prompts and larger timeout/token safety.
# - Learning analysis always links attack_type to the explanation.
# - Placeholder indicator titles are replaced with meaningful titles.
# - Beginner/Intermediate/Advanced are kept visibly different.
# - Assessment stays focused on email difficulty and classification only.

import time as _time_patch

_PROVIDER_RETRYABLE_PATTERNS = re.compile(
    r"503|UNAVAILABLE|high demand|temporar|try again|rate limit|overloaded|timeout|timed out|deadline",
    re.I,
)

# Keep the original network caller, then wrap it with provider-aware retry.
_BASE_CALL_AI_FINAL = call_ai

def _is_retryable_ai_error(data):
    if not isinstance(data, dict) or "error" not in data:
        return False
    err = data.get("error")
    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return bool(_PROVIDER_RETRYABLE_PATTERNS.search(msg))

def _compact_prompt_for_slow_provider(prompt):
    """Shorten long prompts for Claude/Gemini without removing the JSON contract."""
    if not isinstance(prompt, str) or len(prompt) < 3500:
        return prompt
    # Preserve the most important parts: top task + JSON contract + final rules.
    head = prompt[:2200]
    tail = prompt[-1800:]
    return head + "\n\n[Prompt shortened: keep all rules above; avoid repetition; return valid JSON only.]\n\n" + tail

def call_ai(prompt, max_tokens=1600):
    provider = st.session_state.get("ai_provider", "groq")
    attempts = 3 if provider in {"gemini", "anthropic", "groq"} else 2
    last = None
    for attempt in range(attempts):
        use_prompt = _compact_prompt_for_slow_provider(prompt) if provider in {"gemini", "anthropic"} else prompt
        data = _BASE_CALL_AI_FINAL(use_prompt, max_tokens=max(max_tokens, 3500 if provider in {"gemini", "anthropic"} else max_tokens))
        if not _is_retryable_ai_error(data):
            return data
        last = data
        _time_patch.sleep(1.2 * (attempt + 1))
    return last or {"error": {"message": "AI provider failed after retries."}}

def call_groq(prompt, max_tokens=1600):
    return call_ai(prompt, max_tokens)

# Stronger prompt wording that avoids placeholder-like indicator titles.
_BASE_BUILD_PROMPT_FINAL = build_prompt
_BASE_BUILD_ASSESS_PROMPT_FINAL = build_assess_prompt

def build_prompt(role, index, language):
    base = _BASE_BUILD_PROMPT_FINAL(role, index, language)
    if language == "Arabic":
        return base + """

تعليمات جودة نهائية:
- لا تكتب عناوين عامة مثل "مؤشر ثالث" أو "مؤشر مختلف".
- اكتب أسماء مؤشرات واضحة مثل: طلب بيانات حساسة، إساءة MFA/OTP، مرفق مشبوه، QR غير موثوق، انتحال سلطة، عدم توافق سير العمل.
- يجب أن يوضح أول مؤشر: نوع الهجوم + الإجراء الخطر داخل البريد.
- يجب أن تكون أمثلة المبتدئ واضحة، والمتوسط مقنعة جزئياً، والمتقدم قريب من الشرعي لكن لا يزال خطراً.
- اجعل كل مثال جديد مختلفاً في: المرسل، النطاق، القسم، الطلب، وناقل الهجوم.
"""
    return base + """

Final quality instructions:
- Do not write generic placeholder titles like "different clue" or "third different clue".
- Use concrete indicator titles such as: Sensitive data request, MFA/OTP abuse, Suspicious attachment, Untrusted QR code, Authority impersonation, Workflow mismatch.
- Indicator 1 must state the attack type plus the risky action inside the email.
- Beginner must be obvious, Intermediate partly convincing, Advanced near-legitimate but still risky.
- Make every example differ in sender, domain, department, request, and attack vector.
"""

def build_assess_prompt(role, index, is_phishing, language):
    base = _BASE_BUILD_ASSESS_PROMPT_FINAL(role, index, is_phishing, language)
    if language == "Arabic":
        return base + """

تعليمات اختبار نهائية:
- لا تضف تحليل المعلم داخل الاختبار.
- البريد نفسه فقط يجب أن يثبت مستوى الصعوبة المختار.
- إذا كان شرعياً: لا روابط خارجية، لا تهديد، لا بيانات حساسة، ولا نطاق غير رسمي.
- إذا كان تصيداً: اجعل العلامات مناسبة للصعوبة، وليس كلها واضحة في المتقدم.
"""
    return base + """

Final assessment instructions:
- Do not include tutor-analysis sections in assessment questions.
- The email content itself must prove the selected difficulty level.
- If legitimate: no external links, no threats, no sensitive requests, and no unofficial domain.
- If phishing: red flags must match the difficulty; Advanced must not look Beginner-level obvious.
"""

# =============================================================
# UX REALISM PATCH — QR/link rendering contract + professional tone
# -------------------------------------------------------------
# EN: Adds two things on top of every previous prompt layer:
#  1) A strict, render-friendly contract for how a QR code and a
#     clickable link must be written inside `body`, so the UI can
#     turn them into a REAL scannable QR image / a REAL clickable
#     button instead of literal bracket text like "[QR Code: ...]".
#  2) An instruction to make the wording read like a genuinely
#     professional, polished email — not generic training copy.
# AR: تضيف طبقتين فوق كل طبقات التعليمات السابقة:
#  1) صيغة ثابتة وصارمة لكتابة رمز QR والرابط داخل body، حتى تقدر
#     الواجهة تحوّلها لصورة QR فعلية قابلة للمسح / زر فعلي قابل
#     للنقر، بدل نص بين أقواس لا يظهر كباركود حقيقي.
#  2) تعليمة لجعل صياغة الرسالة تقرأ كبريد احترافي حقيقي، وليس
#     نصًا تدريبيًا عامًا.
# =============================================================
_BASE_BUILD_PROMPT_UX = build_prompt
_BASE_BUILD_ASSESS_PROMPT_UX = build_assess_prompt

_UX_CONTRACT_AR = """

تنسيق إلزامي: QR → اكتب مرة واحدة فقط [QR: نص قصير]، والرابط الفعلي في suspicious_link فقط لا تكرره بالنص أو التوقيع.
زر → اكتب مرة واحدة فقط [نص الزر](الرابط)، ولا تكرر الرابط بعدها.
اجعل الصياغة كاملة تقرأ كبريد احترافي حقيقي، لا حشو تدريبي.
مهم جدًا: أعد القيم بالحقول المباشرة بالمستوى الأول فقط كما هي بالمخطط تمامًا، بدون أي حقول إضافية أو متداخلة غير المطلوبة. استخدم نفس أسماء الحقول المحددة بالمخطط حرفيًا، بدون أي تسمية بديلة.
اكتب المحتوى بلغة واحدة فقط هي اللغة المطلوبة بهذا الطلب. لا تضف نسخة ثانية بلغة أخرى بأي مكان من الرد.
"""
_UX_CONTRACT_EN = """

Mandatory format: QR -> write ONCE [QR: short label]; real URL only in suspicious_link, never repeated in body/signature.
Button -> write ONCE [Button label](url); never print that URL again afterward.
Write the whole message like a real professional email, no generic training filler.
CRITICAL: Return values directly as the top-level fields defined in the schema only — no extra or nested fields beyond what is shown. Use the exact field names given in the schema, character for character — do not substitute alternate names.
Write the content in ONLY the single language requested for this request. Do not include a second-language version anywhere in the response.
"""

def build_prompt(role, index, language):
    base = _BASE_BUILD_PROMPT_UX(role, index, language)
    base = base + (_UX_CONTRACT_AR if language == "Arabic" else _UX_CONTRACT_EN)
    if language == "Arabic":
        return base + "\n\nتعليمة لغة نهائية صارمة: كل حقل نصي (subject, body, indicators, why_risky, learning_tip) يجب يكون بالعربية الفصحى بالكامل. ممنوع أي كلمة أو جملة إنجليزية إلا أسماء العلامات التجارية أو الاختصارات التقنية الشائعة (مثل MFA، OTP، PACS). أي رد يحتوي فقرة كاملة بالإنجليزية يعتبر خطأ ويجب رفضه.\n"
    return base + "\n\nFINAL strict language directive: every text field (subject, body, indicators, why_risky, learning_tip) must be written entirely in English. Do not include any Arabic word or sentence anywhere in the response.\n"

def build_assess_prompt(role, index, is_phishing, language):
    base = _BASE_BUILD_ASSESS_PROMPT_UX(role, index, is_phishing, language)
    base = base + (_UX_CONTRACT_AR if language == "Arabic" else _UX_CONTRACT_EN)
    if language == "Arabic":
        return base + "\n\nتعليمة لغة نهائية صارمة: كل حقل نصي يجب يكون بالعربية الفصحى بالكامل. ممنوع أي جملة إنجليزية كاملة إلا الاختصارات التقنية الشائعة.\n"
    return base + "\n\nFINAL strict language directive: every text field must be written entirely in English. Do not include any Arabic word or sentence anywhere in the response.\n"
# =============================================================
# END UX REALISM PATCH
# =============================================================

# =============================================================
# DIFFICULTY ENFORCEMENT PATCH
# Enforces the 4-axis framework strictly at prompt level:
# - EASY: generic greeting, plain text URL, NO QR, NO button, 2 spelling errors
# - MEDIUM: semi-personal greeting, simple button, NO QR, 1 spelling error
# - HARD: full name+title, QR mandatory, professional button, zero errors
# Also post-processes the AI output to remove QR from easy/medium
# =============================================================
_BASE_BUILD_PROMPT_DIFF = build_prompt
_BASE_BUILD_ASSESS_PROMPT_DIFF = build_assess_prompt

_DIFF_ADDON_EASY_AR = """

⚠️ تعليمات صارمة جداً لمستوى السهل — يجب الالتزام بها حرفياً:
1. التحية: "عزيزي الموظف" أو "عزيزي الزميل" فقط — ممنوع منعاً باتاً أي اسم شخصي.
2. هوية المرسل (from): يجب أن تكون جهة عامة أو اسم قسم فقط (مثل "فريق الدعم الفني" أو "قسم الموارد البشرية") — ممنوع تماماً أن يكون المرسل شخصاً باسمه الشخصي أو بلقب وظيفي (ممنوع كتابة "د." أو "Dr." أو اسم كامل + منصب) حتى في التوقيع بنهاية الرسالة.
3. النطاق: يجب أن يكون واضح التزوير تماماً ولا يشبه أي جهة حقيقية إطلاقاً — ممنوع استخدام كلمات مثل gov أو moh أو board أو ministry أو أي تركيبة تحاكي moh.gov.sa أو hospital.org.
4. الأخطاء: ضع بالضبط خطأين إملائيين أو نحويين واضحين في جسم الرسالة — هذا إلزامي ومطلوب.
5. الرابط: ضع الرابط كنص خام مرئي فقط في جسم الرسالة (مثل: http://fake-hospital.com/update) — ممنوع استخدام زر أو markdown.
6. QR: محظور تماماً — لا تكتب [QR:...] أبداً.
7. المرفق: محظور تماماً — الحقل attachment يجب أن يبقى فارغاً تماماً.
8. الإلحاح: صريح ومباشر ("الآن فوراً" أو "سيُغلق حسابك اليوم").
"""

_DIFF_ADDON_EASY_EN = """

⚠️ STRICT EASY LEVEL RULES — follow these literally or the output is invalid:
1. Greeting: MUST be "Dear Employee" or "Dear Staff" — ANY personal name is FORBIDDEN.
2. Sender identity (from): MUST be a generic team or department only (e.g. "IT Support Team" or "HR Department") — a named individual with a personal title (e.g. "Dr.", a full first+last name, a job title in the signature) is STRICTLY FORBIDDEN, including in the closing signature.
3. Domain: must be completely and obviously fake, with NO resemblance to any real organization — do NOT use words like gov, moh, board, ministry, or any pattern that mimics moh.gov.sa or hospital.org.
4. Errors: place EXACTLY TWO obvious spelling/grammar mistakes in the body — this is REQUIRED.
5. Link: place the URL as RAW VISIBLE PLAIN TEXT in the body (e.g. http://fake-hospital.com/update) — NO button, NO markdown link.
6. QR: COMPLETELY FORBIDDEN — do NOT write [QR:...] anywhere.
7. Attachment: FORBIDDEN — the "attachment" field MUST be left completely empty.
8. Urgency: direct and explicit ("Act NOW", "your account closes TODAY").
"""

_DIFF_ADDON_MEDIUM_AR = """

⚠️ تعليمات صارمة جداً لمستوى المتوسط — يجب الالتزام بها حرفياً:
1. التحية: استخدم الاسم الأول أو المسمى الوظيفي فقط (مثل "عزيزي د. أحمد") — ممنوع التحية العامة الكاملة وممنوع الاسم الكامل + اللقب الدقيق.
2. النطاق: مشابه للرسمي مع فرق بسيط يمكن ملاحظته عند التدقيق (مثل hospital-it.net) — ليس واضح التزوير تماماً كمستوى السهل، وليس شبه رسمي بذكاء كمستوى الصعب.
3. الأخطاء: ضع بالضبط خطأً إملائياً واحداً خفيفاً في جسم الرسالة — هذا إلزامي.
4. الرابط: استخدم زراً بسيطاً [نص](رابط) — مسموح. مرفق PDF بسيط وعام مسموح أيضاً (اختياري).
5. QR: محظور تماماً — لا تكتب [QR:...] أبداً.
6. الارتباط الوظيفي: يجب ذكر اسم القسم الحقيقي (سريري/إداري/تقني) داخل الرسالة.
"""

_DIFF_ADDON_MEDIUM_EN = """

⚠️ STRICT INTERMEDIATE LEVEL RULES — follow these literally:
1. Greeting: use first name or job title ONLY (e.g. "Dear Dr. Ahmed") — no fully generic greeting and no full name + precise title.
2. Domain: look-alike to the real one with a small detectable difference (e.g. hospital-it.net) — not obviously fake like Easy, and not near-official like Advanced.
3. Errors: place EXACTLY ONE subtle spelling/grammar mistake in the body — this is REQUIRED.
4. Link: a simple button [label](url) is allowed. A simple generic PDF attachment is also allowed (optional).
5. QR: COMPLETELY FORBIDDEN — do NOT write [QR:...] anywhere.
6. Role alignment: mention the real department name (Clinical/Administrative/IT) inside the email.
"""

_DIFF_ADDON_HARD_AR = """

⚠️ تعليمات صارمة جداً لمستوى الصعب — يجب الالتزام بها حرفياً (الكل إلزامي معاً):
1. التحية: الاسم الكامل + اللقب الوظيفي الدقيق (مثل "عزيزتي د. نورة العتيبي، استشارية الأمراض الداخلية").
2. النطاق: شبه رسمي بذكاء (مثل moh-staff.net) — ممنوع كلمات: secure, update, verify, login, reset.
3. المرفق: إلزامي ومطلوب دائماً — يجب أن يحتوي الحقل attachment على اسم مستند رسمي واقعي (مثل Compliance_Protocol_2024.pdf).
4. QR: اختياري ومتنوّع — لا تضعه في كل إيميل. ضعه فقط إذا كان يخدم السيناريو فعلاً (مثل تحقق عبر جهاز موبايل أو تسجيل جهاز)، واكتبه [QR: نص قصير وصفي]. إذا لم يكن مناسباً للسيناريو، لا تكتب أي رمز QR إطلاقاً واعتمد على الزر أو الرابط فقط.
5. الزر: استخدم زراً رسمياً باسم وصفي واضح (ليس "Open Link" أو "اضغط هنا").
6. الأخطاء: صفر أخطاء — لغة احترافية كاملة.
7. الإلحاح: خفيف ومهذب فقط ("إجراء روتيني") — ممنوع أي تهديد.
8. التوقيع: اسم كامل + المنصب + القسم + رقم تحويلة داخلية حقيقي (ليس XX).
9. الارتباط الوظيفي: يجب ربط الرسالة بمهمة يومية محددة جداً للدور المختار، وذكر اسم زميل أو قسم داخلي محدد (وليس عاماً).
"""

_DIFF_ADDON_HARD_EN = """

⚠️ STRICT ADVANCED LEVEL RULES — follow these literally (ALL are mandatory together):
1. Greeting: FULL NAME + precise job title (e.g. "Dear Dr. Noura Al-Otaibi, Internal Medicine Consultant").
2. Domain: near-official but not matching (e.g. moh-staff.net) — FORBIDDEN words: secure, update, verify, login, reset.
3. Attachment: MANDATORY AND REQUIRED — the "attachment" field MUST contain a realistic, officially named document (e.g. Compliance_Protocol_2024.pdf).
4. QR: OPTIONAL and VARIED — do NOT include it in every email. Only include [QR: short descriptive label] when it genuinely fits the scenario (e.g. mobile device check-in, device enrollment). If it doesn't fit naturally, omit it completely and rely on the button/link instead.
5. Button: use a professionally descriptive label (NOT "Open Link" or "Click Here").
6. Errors: ZERO spelling or grammar errors.
7. Urgency: polite and subtle ONLY ("routine procedure") — NO threats.
8. Signature: full name + title + department + real internal extension (no XX placeholders).
9. Role alignment: tie the email to a very specific daily task of the selected role, and name a specific colleague or internal department (not generic).
"""

def build_prompt(role, index, language):
    base = _BASE_BUILD_PROMPT_DIFF(role, index, language)
    difficulty = st.session_state.get("difficulty", "medium")
    is_ar = (language == "Arabic")
    if difficulty == "easy":
        base += _DIFF_ADDON_EASY_AR if is_ar else _DIFF_ADDON_EASY_EN
    elif difficulty == "medium":
        base += _DIFF_ADDON_MEDIUM_AR if is_ar else _DIFF_ADDON_MEDIUM_EN
    elif difficulty in ("hard", "advanced"):
        base += _DIFF_ADDON_HARD_AR if is_ar else _DIFF_ADDON_HARD_EN
    return base

def build_assess_prompt(role, index, is_phishing, language):
    base = _BASE_BUILD_ASSESS_PROMPT_DIFF(role, index, is_phishing, language)
    difficulty = st.session_state.get("difficulty", "medium")
    is_ar = (language == "Arabic")
    if difficulty == "easy":
        base += _DIFF_ADDON_EASY_AR if is_ar else _DIFF_ADDON_EASY_EN
    elif difficulty == "medium":
        base += _DIFF_ADDON_MEDIUM_AR if is_ar else _DIFF_ADDON_MEDIUM_EN
    elif difficulty in ("hard", "advanced"):
        base += _DIFF_ADDON_HARD_AR if is_ar else _DIFF_ADDON_HARD_EN
    return base

# =============================================================
# END DIFFICULTY ENFORCEMENT PATCH
# =============================================================

_BAD_INDICATOR_TITLES = re.compile(
    r"^(different behavioral or technical clue|third different clue|indicator\s*\d+|مؤشر\s*\d+|مؤشر ثالث|مؤشر سلوكي أو تقني مختلف)$",
    re.I,
)

def _clean_indicator_title(title, attack_type, n, is_ar=False):
    t0 = (title or "").strip()
    if t0 and not _BAD_INDICATOR_TITLES.match(t0):
        return t0
    if is_ar:
        defaults = [f"نوع الهجوم: {attack_type}", "طلب أو سير عمل غير معتاد", "عدم توافق القناة أو الدور"]
    else:
        defaults = [f"Attack Type: {attack_type}", "Unusual request or workflow", "Role-context or channel mismatch"]
    return defaults[min(max(n-1,0),2)]

_BASE_NORMALIZE_LEARNING_FINAL = normalize_learning_analysis

_FIELD_SYNONYMS = {
    "from":             ["from", "sender", "sender_email", "sender_address", "from_email", "from_address", "sender_display"],
    "to":               ["to", "recipient", "recipient_email", "to_email", "to_address"],
    "subject":          ["subject", "email_subject", "title"],
    "body":             ["body", "body_html", "body_text", "message", "message_body", "content", "email_body"],
    "suspicious_link":  ["suspicious_link", "link", "malicious_link", "phishing_link", "url"],
    "attachment":       ["attachment", "attachment_name", "file_attachment"],
}
_CANONICAL_FIELDS = list(_FIELD_SYNONYMS.keys())

def _strip_html(text):
    """Quick plain-text fallback for a body field that came back as
    body_html (e.g. '<p>Dear Doctor</p>') instead of plain text."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"</p>|</div>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def _try_parse_blob(value):
    """If `value` is a string that LOOKS like a stringified dict (Python-
    repr style, single-quoted — not valid JSON), parse it into a real
    dict. Tries the exact/clean parse first, then a couple of common
    real-world breakages (stray backslashes) before giving up."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    for candidate in (s, re.sub(r'\\+(?=["\'])', '', s)):
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None

def _deep_flatten(value, pool, depth=0):
    """Walk `value` (dict, or a string that might itself be a stringified
    dict) and collect EVERY scalar key/value pair it contains into `pool`,
    at any nesting depth, under any key name the model invented — so it
    does not matter whether a provider nests content under 'email',
    'email_content', 'analysis', or anything else not seen before.
    Shallower keys win: once a key is in the pool, deeper duplicates of
    the same key name are ignored, so top-level fields are never
    overwritten by something buried inside a sub-blob."""
    if depth > 6:
        return
    if isinstance(value, dict):
        # First pass at this level: capture plain scalars so this level's
        # own keys take priority over anything nested deeper inside it.
        # An EMPTY placeholder (e.g. top-level "subject": "") must NOT
        # block a real, non-empty value found deeper from winning.
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)):
                if k not in pool or (not pool[k] and v):
                    pool[k] = v
        # Second pass: recurse into nested dicts / stringified blobs.
        for k, v in value.items():
            if isinstance(v, dict):
                _deep_flatten(v, pool, depth + 1)
            elif isinstance(v, str):
                parsed = _try_parse_blob(v)
                if parsed is not None:
                    _deep_flatten(parsed, pool, depth + 1)
            elif isinstance(v, list):
                for item in v:
                    _deep_flatten(item, pool, depth + 1)

def _recover_from_nested_email_blob(result):
    """Universal recovery: regardless of WHICH unexpected shape a provider
    used (wrong field names, content nested one or more levels deep under
    an arbitrary wrapper key, HTML body instead of plain text, etc.), this
    rebuilds the canonical top-level fields (from/to/subject/body/
    suspicious_link/attachment) from whatever the model actually produced,
    anywhere in the response. This intentionally does NOT special-case any
    specific wrapper key name (like 'email' or 'analysis') — new naming
    variants are handled automatically without needing a code change.
    Wrapped in a try/except as a whole: this is glue code patching up
    inconsistent provider output, so a bug in it must never crash the whole
    generation pipeline — worst case, it just leaves the original result
    untouched and the existing core_missing check handles it normally."""
    if not isinstance(result, dict):
        return result
    try:
        pool = {}
        _deep_flatten(result, pool)

        for canonical in _CANONICAL_FIELDS:
            cur = result.get(canonical)
            if isinstance(cur, str) and cur.strip():
                continue  # already has a real value, leave it alone
            for alt in _FIELD_SYNONYMS[canonical]:
                val = pool.get(alt)
                if isinstance(val, str) and val.strip():
                    result[canonical] = val
                    break

        # body_html (or any HTML-flavoured body) needs tag-stripping to be
        # readable plain text in the email window.
        body = result.get("body")
        if isinstance(body, str) and re.search(r"<[a-zA-Z]+[^>]*>", body):
            result["body"] = _strip_html(body)

        # Final type safety net: nothing downstream expects these fields to
        # ever be anything but a plain string. If recovery still leaves a
        # non-string (e.g. a dict slipped through), drop it to "" rather
        # than let it crash re.sub()/.strip() elsewhere.
        for canonical in _CANONICAL_FIELDS:
            if canonical in result and not isinstance(result[canonical], str):
                result[canonical] = "" if result[canonical] is None else str(result[canonical])
    except Exception:
        pass
    return result

def normalize_learning_analysis(result, role_type, difficulty, is_ar=False):
    result = _BASE_NORMALIZE_LEARNING_FINAL(result, role_type, difficulty, is_ar)
    if not isinstance(result, dict):
        return result
    result = _recover_from_nested_email_blob(result)
    is_ar = _detect_is_ar(result, is_ar)
    # SAFETY NET: if the core fields are still empty after every parsing /
    # repair step, this generation effectively failed (most likely a
    # response that got cut off before the JSON closed). Surface this as a
    # clear error + retry button instead of silently rendering a blank
    # email window, which was confusing and hard to diagnose.
    if "error" not in result:
        core_missing = not (_is_nonempty_str(result.get("from")) and
                             _is_nonempty_str(result.get("subject")) and
                             _is_substantial_str(result.get("body")))
        if core_missing:
            debug_keys = {k: (str(v)[:400] if v else v) for k, v in result.items()}
            return {"error": {"code": "incomplete_content", "message": "incomplete_generation", "debug": debug_keys}}
    result = _resolve_leftover_placeholders(result, is_ar)
    attack_type = result.get("attack_type") or infer_attack_type_from_content(result, is_ar)
    result["attack_type"] = attack_type
    indicators = result.get("indicators") if isinstance(result.get("indicators"), list) else []
    while len(indicators) < 3:
        indicators.append({"number": len(indicators)+1, "title": "", "description": ""})
    for i in range(3):
        indicators[i]["number"] = i + 1
        indicators[i]["title"] = _clean_indicator_title(indicators[i].get("title"), attack_type, i+1, is_ar)
        desc = (indicators[i].get("description") or "").strip()
        if i == 0 and attack_type not in desc:
            if is_ar:
                desc = f"هذا المؤشر يوضح {attack_type} لأنه يرتبط مباشرة بالإجراء الخطر المطلوب في البريد. " + desc
            else:
                desc = f"This shows {attack_type} because it directly matches the risky action requested in the email. " + desc
        indicators[i]["description"] = desc.strip()
    result["indicators"] = indicators[:3]
    wr = (result.get("why_risky") or "").strip()
    if attack_type and attack_type not in wr:
        role_hint = (ROLE_ATTACK_HINTS_AR if is_ar else ROLE_ATTACK_HINTS).get(role_type, ROLE_ATTACK_HINTS["other"])
        prefix = f"هذه رسالة {attack_type} وتؤثر على {role_hint}. " if is_ar else f"This is a {attack_type} email that affects {role_hint}. "
        result["why_risky"] = prefix + wr
    return result

_BASE_GENERATION_ISSUES_FINAL = get_generation_quality_issues



# =============================================================
# STRICT ROLE + DIFFICULTY GUARDRAILS (patched)
# -------------------------------------------------------------
# These checks reject AI outputs that drift away from the selected
# job role (clinical/admin/IT) or violate the documented difficulty
# framework before the email is shown to the trainee.
# =============================================================
_ROLE_KEYWORDS = {
    "clinical": re.compile(r"\b(patient|clinical|emr|ehr|doctor|nurse|pharmac|medication|lab|radiology|pacs|icu|er|ward|handover|vitals|diagnostic|prescription|blood bank)\b|مريض|سريري|طبيب|ممرض|صيدل|دواء|مختبر|أشعة|مناوبة|قسم|بنك الدم|سجل طبي", re.I),
    "admin": re.compile(r"\b(invoice|procurement|vendor|supplier|payroll|insurance|billing|appointment|contract|leave|hr|administrative|records office)\b|فاتورة|مورد|مشتريات|رواتب|تأمين|فوترة|مواعيد|عقد|إجازات|إداري", re.I),
    "it": re.compile(r"\b(vpn|server|network|firewall|ssl|certificate|helpdesk|active directory|backup|database|endpoint|software|license|mfa|otp|cyber|it support)\b|شبكة|خادم|جدار ناري|شهادة|دعم تقني|نسخ احتياطي|قاعدة بيانات|ترخيص|تقنية|أمن سيبراني", re.I),
}
_ROLE_FORBIDDEN = {
    "clinical": re.compile(r"\b(payroll|invoice|vendor|procurement|leave balance|hr portal|vpn|ssl certificate|server|helpdesk|software license|records management team|document collaboration team|security team)\b|رواتب|فاتورة|مورد|مشتريات|إجازات|بوابة الموارد|دعم تقني|شبكة|خادم|فريق الأمن|فريق إدارة السجلات", re.I),
    "admin": re.compile(r"\b(emr|lab results|clinical handover|patient vitals|medication order|radiology image|vpn|ssl certificate|server|firewall)\b|تسليم سريري|نتائج مختبر|علامات حيوية|أمر دوائي|أشعة|شبكة|خادم", re.I),
    "it": re.compile(r"\b(lab results|clinical handover|patient vitals|payroll bank|supplier invoice|appointment booking)\b|نتائج مختبر|تسليم سريري|علامات حيوية|فاتورة مورد|رواتب|حجز مواعيد", re.I),
}

def _current_role_type_for_guardrail():
    try:
        role = st.session_state.get("role", "Clinical")
        return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]
    except Exception:
        return "clinical"

def _role_alignment_issues(result):
    role_type = _current_role_type_for_guardrail()
    if role_type == "other":
        return []
    text = " ".join(str(result.get(k, "")) for k in ["email_type", "from", "subject", "body", "attachment", "suspicious_text", "suspicious_link"])
    issues = []
    if not _ROLE_KEYWORDS[role_type].search(text):
        issues.append(f"email content is not clearly aligned with the selected {role_type} role")
    if _ROLE_FORBIDDEN[role_type].search(text):
        issues.append(f"email drifts into a different role instead of the selected {role_type} role")
    return issues

def _difficulty_structure_issues(result, difficulty, is_phishing=True):
    if not is_phishing or not isinstance(result, dict):
        return []
    body = str(result.get("body", ""))
    combined = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    issues = []
    has_qr = bool(re.search(r"\[\s*QR", body, re.I))
    has_attachment = bool(str(result.get("attachment", "")).strip())
    link = str(result.get("suspicious_link", "")).strip()
    if difficulty == "easy":
        if has_qr: issues.append("Beginner/Easy must not contain QR code")
        if has_attachment: issues.append("Beginner/Easy must not contain attachment")
        if link and link not in body: issues.append("Beginner/Easy must show the fake URL visibly in the body")
        if re.search(r"\[[^\]]+\]\(https?://", body): issues.append("Beginner/Easy must use plain visible URL, not a button")
    elif difficulty == "medium":
        if has_qr: issues.append("Intermediate must not contain QR code")
        if re.search(r"act now|today or|immediately or|account closes today|تصرف الآن|اليوم أو|فورًا وإلا", combined, re.I):
            issues.append("Intermediate urgency is too aggressive")
    elif difficulty == "hard":
        if not has_qr: issues.append("Advanced/Hard must include a QR code marker [QR: ...]")
        if re.search(r"https?://", body): issues.append("Advanced/Hard must not expose raw URL in body")
        if not has_attachment: issues.append("Advanced/Hard must include an official named attachment")
        if re.search(r"password|enter your login|provide your credentials|كلمة المرور|بيانات الدخول", combined, re.I):
            issues.append("Advanced/Hard must avoid direct credential requests")
    return issues

def get_generation_quality_issues(result, difficulty, is_phishing=True):
    issues = _BASE_GENERATION_ISSUES_FINAL(result, difficulty, is_phishing)
    if not isinstance(result, dict):
        return issues
    if is_phishing:
        # Prevent model-copy artifacts that appeared during testing.
        titles = " ".join(str((x.get("title") or "")) for x in result.get("indicators", []) if isinstance(x, dict))
        if _BAD_INDICATOR_TITLES.search(titles):
            issues.append("indicator titles must be concrete, not placeholders")
        # The analysis must mention the attack type in learning outputs.
        attack_type = result.get("attack_type") or result.get("email_type") or ""
        analysis_text = " ".join(str(result.get(k, "")) for k in ["why_risky", "learning_tip"])
        analysis_text += " " + " ".join(str((x.get("description") or "")) for x in result.get("indicators", []) if isinstance(x, dict))
        if attack_type and attack_type not in analysis_text:
            issues.append("analysis must connect to attack_type")
    return issues

# =============================================================
# FRAMEWORK COMPLIANCE PATCH — Difficulty_Framework.docx enforcement
# Adds deterministic (non-LLM) gating on top of every previous prompt
# layer. Covers, per the accumulated findings across Easy/Medium/Hard
# testing rounds:
#   1) Verifiable spelling-error count (Easy=2, Medium=1, Hard=0) using
#      the model-reported injected_errors field, with a regex fallback
#      list of common misspellings when that field is missing.
#   2) Sender identity per Axis 1: no personal name/title at Easy,
#      no FULL name+title (Advanced-style) leaking into Medium, and a
#      required personal name+title at Hard. Domain sophistication:
#      Easy must be obviously fake, not government-like.
#   3) Axis 4 role alignment, deepened: reject generic commercial/
#      prize/marketing-themed phishing for role-specific roles even if
#      a role keyword is superficially mentioned once.
#   4) Axis 3: NO attachment at Easy; MANDATORY named attachment at
#      Hard (QR is now OPTIONAL/varied at Hard, not mandatory); and a
#      consistency check so the body never claims an attachment exists
#      when the attachment field is actually empty.
#   5) Recipient/greeting name consistency: the greeting must not name
#      someone entirely unrelated to the "to" address.
# This wraps the same shared function used by build_prompt's retry
# loop, so it automatically applies across all 4 AI providers and
# both languages (Arabic/English) without touching each provider path.
# =============================================================
_BASE_GENERATION_ISSUES_FRAMEWORK = get_generation_quality_issues

_PERSONAL_TITLE_RE = re.compile(
    r'\b(Dr\.?|Prof\.?|Professor|Doctor)\s+[A-Z][a-zA-Z\-]+(\s+[A-Z][a-zA-Z\-]+)?'
    r'|(د\.|دكتور|دكتورة|الدكتور|الدكتورة|أ\.د)\s*[\u0600-\u06FF]{2,}',
    re.I,
)
# Stricter: title + TWO capitalized name words (full first+last name) —
# this is the Advanced-level sender signature and must not leak into
# Easy or Medium (Medium may only use a single first name/title).
_FULL_NAME_TITLE_RE = re.compile(
    r'\b(Dr\.?|Prof\.?|Professor|Doctor)\s+[A-Z][a-zA-Z\-]+\s+[A-Z][a-zA-Z\-]+'
    r'|(د\.|دكتور|دكتورة|الدكتور|الدكتورة|أ\.د)\s*[\u0600-\u06FF]{2,}\s+[\u0600-\u06FF]{2,}',
    re.I,
)
_LOOKALIKE_GOV_DOMAIN_RE = re.compile(r'\b(gov|moh|ministry|board)\b', re.I)

_COMMON_MISSPELLINGS = [
    "acess", "informatin", "comunity", "recieve", "seperate", "occured", "untill",
    "goverment", "priviledge", "enviroment", "maintainance", "noticable",
    "reccommend", "adress", "begining", "calender", "definately", "embarass",
    "harrass", "independant", "occassion", "reccomend", "succesful", "tommorow",
    "wich", "teh", "hte", "loosing", "patint", "guidlines", "aknowledge",
    "acknowlegde", "requiered", "immediatly", "urgant", "pleaes", "thier",
]
_MISSPELLING_RE = re.compile(r'\b(' + '|'.join(_COMMON_MISSPELLINGS) + r')\b', re.I)

_COMMERCIAL_THEME_RE = re.compile(
    r'\bprize\b|\breward\b|\bcash\s*back\b|\bdiscount\b|\bpromotion\b|\banniversary\b|'
    r'\bwellness program\b|\btelecom\b|\bbank offer\b|\bloan\b|\bvoucher\b|\bgift\b|'
    r'\bwin(?:ner|ning)?\b|\braffle\b|\blucky draw\b|'
    r'جائزة|خصم|عرض تجاري|كاش باك|مكافأة|قرض|هدية|رابح|سحب',
    re.I,
)

_ROLE_KEYWORD_RE = {
    "clinical": re.compile(
        r'\b(patient|emr|lab result|medication|ward round|nurse|physician|diagnosis|radiology|icu)\w*'
        r'|مريض|عيادة|مختبر|دواء|تمريض|تشخيص|أشعة|طوارئ|عناية\s*مركزة',
        re.I,
    ),
    "admin": re.compile(
        r'\b(invoice|contract|billing|insurance|procurement|hr|schedule|vendor|accreditation)\w*'
        r'|فاتورة|عقد|تأمين|مشتريات|موارد\s*بشرية|جدولة|مورد|اعتماد',
        re.I,
    ),
    "it": re.compile(
        r'\b(network|vpn|server|firewall|backup|active directory|certificate|license|helpdesk|cybersecurity)\w*'
        r'|شبكة|خادم|جدار\s*ناري|نسخ\s*احتياطي|شهادة|ترخيص|الدعم\s*الفني',
        re.I,
    ),
}

# Generic commercial/prize themes are never role-specific regardless of any
# incidental keyword match (handled separately via _COMMERCIAL_THEME_RE).
# NOTE: MFA/OTP is intentionally NOT treated as an automatic violation here
# anymore — "MFA / OTP Abuse" tied to clinical-system access is one of the
# legitimately pre-approved Attack Playbook vectors for the clinical role
# (see ATTACK_PLAYBOOK["clinical"]), so rejecting it outright just caused
# repeated failed retries on a perfectly valid scenario. We only flag a
# login/credential theme as a role-mismatch when it is otherwise completely
# generic (no role keyword at all) — which the primary keyword-missing
# check below already covers.
_GENERIC_IT_HR_THEME_RE = re.compile(
    r'document sharing|log in now',
    re.I,
)

_ATTACHMENT_MENTION_RE = re.compile(r'\battach(?:ed|ment)?\b|مرفق|مرفقة|المرفق', re.I)

def _extract_name_tokens_from_email(addr):
    local = (addr or "").split("@")[0]
    parts = re.split(r'[.\-_]', local)
    return [p for p in parts if len(p) >= 3 and not p.isdigit()]



# =============================================================
# STRICT ROLE + DIFFICULTY GUARDRAILS (patched)
# -------------------------------------------------------------
# These checks reject AI outputs that drift away from the selected
# job role (clinical/admin/IT) or violate the documented difficulty
# framework before the email is shown to the trainee.
# =============================================================
_ROLE_KEYWORDS = {
    "clinical": re.compile(r"\b(patient|clinical|emr|ehr|doctor|nurse|pharmac|medication|lab|radiology|pacs|icu|er|ward|handover|vitals|diagnostic|prescription|blood bank)\b|مريض|سريري|طبيب|ممرض|صيدل|دواء|مختبر|أشعة|مناوبة|قسم|بنك الدم|سجل طبي", re.I),
    "admin": re.compile(r"\b(invoice|procurement|vendor|supplier|payroll|insurance|billing|appointment|contract|leave|hr|administrative|records office)\b|فاتورة|مورد|مشتريات|رواتب|تأمين|فوترة|مواعيد|عقد|إجازات|إداري", re.I),
    "it": re.compile(r"\b(vpn|server|network|firewall|ssl|certificate|helpdesk|active directory|backup|database|endpoint|software|license|mfa|otp|cyber|it support)\b|شبكة|خادم|جدار ناري|شهادة|دعم تقني|نسخ احتياطي|قاعدة بيانات|ترخيص|تقنية|أمن سيبراني", re.I),
}
_ROLE_FORBIDDEN = {
    "clinical": re.compile(r"\b(payroll|invoice|vendor|procurement|leave balance|hr portal|vpn|ssl certificate|server|helpdesk|software license|records management team|document collaboration team|security team)\b|رواتب|فاتورة|مورد|مشتريات|إجازات|بوابة الموارد|دعم تقني|شبكة|خادم|فريق الأمن|فريق إدارة السجلات", re.I),
    "admin": re.compile(r"\b(emr|lab results|clinical handover|patient vitals|medication order|radiology image|vpn|ssl certificate|server|firewall)\b|تسليم سريري|نتائج مختبر|علامات حيوية|أمر دوائي|أشعة|شبكة|خادم", re.I),
    "it": re.compile(r"\b(lab results|clinical handover|patient vitals|payroll bank|supplier invoice|appointment booking)\b|نتائج مختبر|تسليم سريري|علامات حيوية|فاتورة مورد|رواتب|حجز مواعيد", re.I),
}

def _current_role_type_for_guardrail():
    try:
        role = st.session_state.get("role", "Clinical")
        return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]
    except Exception:
        return "clinical"

def _role_alignment_issues(result):
    role_type = _current_role_type_for_guardrail()
    if role_type == "other":
        return []
    text = " ".join(str(result.get(k, "")) for k in ["email_type", "from", "subject", "body", "attachment", "suspicious_text", "suspicious_link"])
    issues = []
    if not _ROLE_KEYWORDS[role_type].search(text):
        issues.append(f"email content is not clearly aligned with the selected {role_type} role")
    if _ROLE_FORBIDDEN[role_type].search(text):
        issues.append(f"email drifts into a different role instead of the selected {role_type} role")
    return issues

def _difficulty_structure_issues(result, difficulty, is_phishing=True):
    if not is_phishing or not isinstance(result, dict):
        return []
    body = str(result.get("body", ""))
    combined = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    issues = []
    has_qr = bool(re.search(r"\[\s*QR", body, re.I))
    has_attachment = bool(str(result.get("attachment", "")).strip())
    link = str(result.get("suspicious_link", "")).strip()
    if difficulty == "easy":
        if has_qr: issues.append("Beginner/Easy must not contain QR code")
        if has_attachment: issues.append("Beginner/Easy must not contain attachment")
        if link and link not in body: issues.append("Beginner/Easy must show the fake URL visibly in the body")
        if re.search(r"\[[^\]]+\]\(https?://", body): issues.append("Beginner/Easy must use plain visible URL, not a button")
    elif difficulty == "medium":
        if has_qr: issues.append("Intermediate must not contain QR code")
        if re.search(r"act now|today or|immediately or|account closes today|تصرف الآن|اليوم أو|فورًا وإلا", combined, re.I):
            issues.append("Intermediate urgency is too aggressive")
    elif difficulty == "hard":
        if not has_qr: issues.append("Advanced/Hard must include a QR code marker [QR: ...]")
        if re.search(r"https?://", body): issues.append("Advanced/Hard must not expose raw URL in body")
        if not has_attachment: issues.append("Advanced/Hard must include an official named attachment")
        if re.search(r"password|enter your login|provide your credentials|كلمة المرور|بيانات الدخول", combined, re.I):
            issues.append("Advanced/Hard must avoid direct credential requests")
    return issues

def get_generation_quality_issues(result, difficulty, is_phishing=True):
    issues = _BASE_GENERATION_ISSUES_FRAMEWORK(result, difficulty, is_phishing)
    if not isinstance(result, dict) or not is_phishing:
        return issues

    body = str(result.get("body") or "")
    subject = str(result.get("subject") or "")
    sender = str(result.get("from") or "")
    to_addr = str(result.get("to") or "")
    attachment = str(result.get("attachment") or "").strip()
    combined_lower = f"{subject} {body}".lower()

    domain_match = re.search(r"@([\w.-]+)>?", sender)
    domain = (domain_match.group(1) if domain_match else "")

    # --- Axis 2 (verifiable spelling-error count) ---
    injected = result.get("injected_errors")
    if isinstance(injected, list):
        err_count = len([e for e in injected if str(e).strip()])
    else:
        err_count = len(_MISSPELLING_RE.findall(body))
    expected = {"easy": 2, "medium": 1, "hard": 0}.get(difficulty)
    if expected is not None and err_count != expected:
        issues.append(f"{difficulty} must contain exactly {expected} spelling/grammar mistake(s) in body (found {err_count})")

    # --- Axis 1 (Sender Identity) + Axis 3 (Technical Elements) ---
    if difficulty == "easy":
        if attachment:
            issues.append("Easy must have NO attachment (attachment field must be empty)")
        if _PERSONAL_TITLE_RE.search(sender):
            issues.append("Easy sender must be generic (team/department) — no personal name or title in 'from'")
        if _LOOKALIKE_GOV_DOMAIN_RE.search(domain):
            issues.append("Easy domain must not resemble a real government/hospital domain (avoid gov/moh/board/ministry)")
    elif difficulty == "medium":
        if _has_generic_greeting(body):
            issues.append("Intermediate greeting must be semi-personal (first name or title), not fully generic")
        if _FULL_NAME_TITLE_RE.search(sender) or _FULL_NAME_TITLE_RE.search(body[:200]):
            issues.append("Intermediate sender must NOT use a full name + job title (that is an Advanced-level signature) — first name/title only")
    elif difficulty == "hard":
        if not attachment:
            issues.append("Advanced must include a mandatory named attachment (attachment field cannot be empty)")
        if not _PERSONAL_TITLE_RE.search(sender) and not _PERSONAL_TITLE_RE.search(body[:200]):
            issues.append("Advanced sender must use a full personal name and job title")
        # QR is intentionally OPTIONAL/varied at Hard now — no check here.

    # --- Axis 3 consistency: body must not claim an attachment that doesn't exist ---
    if _ATTACHMENT_MENTION_RE.search(body) and not attachment:
        issues.append("body mentions an attachment but the attachment field is empty — keep them consistent")

    # --- Axis 4 (Role & Healthcare Context alignment), deepened ---
    role = st.session_state.get("role", "Clinical")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    role_type = role_info[2] if role_info else "clinical"
    role_kw_re = _ROLE_KEYWORD_RE.get(role_type)
    if role_kw_re:
        has_role_keyword = bool(role_kw_re.search(combined_lower))
        if not has_role_keyword:
            issues.append(f"email content must clearly reflect the '{role_type}' role context (missing role-specific keywords)")
        elif _COMMERCIAL_THEME_RE.search(combined_lower):
            issues.append(f"generic commercial/prize/marketing-themed phishing is not allowed for the '{role_type}' role — the scenario must revolve around an actual {role_type} work task, not a superficial keyword mention")
        elif role_type in ("clinical", "admin") and _GENERIC_IT_HR_THEME_RE.search(combined_lower):
            issues.append(f"generic MFA/login/document-sharing phishing is not allowed for the '{role_type}' role — that is an IT-themed scenario; the email must revolve around an actual {role_type} task instead")

    # --- Recipient/greeting name consistency (English only — Arabic
    # transliteration vs Latin email-derived tokens can't be reliably
    # matched, and skip for intentionally generic Easy greetings) ---
    if not _has_generic_greeting(body) and not re.search(r'[\u0600-\u06FF]', body[:120]):
        name_tokens = _extract_name_tokens_from_email(to_addr)
        if name_tokens:
            greeting_zone = body[:120].lower()
            if not any(tok.lower() in greeting_zone for tok in name_tokens):
                issues.append("greeting name does not match the 'to' recipient address — keep them consistent")

    return issues
# =============================================================
# END FRAMEWORK COMPLIANCE PATCH
# =============================================================

# Regeneration helper: when Try Again is clicked, also clear the used topic/domain
# memory for the current phase enough to allow a truly fresh attempt.
def clear_generation_memory_for_current(role_type=None, phase_suffix="learn"):
    if role_type is None:
        role = st.session_state.get("role", "Clinical")
        role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
        _, _, role_type = role_info
    for prefix in ["used_topics", "used_domains"]:
        for key in list(st.session_state.keys()):
            if key.startswith(f"{prefix}_{role_type}_{phase_suffix}"):
                st.session_state[key] = []

# More explicit system prompt for all providers.
def get_system_prompt():
    difficulty = st.session_state.get("difficulty", "medium")
    role = st.session_state.get("role", "Clinical")
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    role_context = get_role_unbounded_context(role_type, False)
    return f"""
You are a safe cybersecurity training content generator for a Saudi healthcare phishing-awareness study.
Role context: {role_context}
Difficulty: {difficulty}.
Return valid JSON only. No markdown, no comments, no prose outside JSON.
Do not use fixed templates, repeated domains, or placeholder indicator titles.
Learning email JSON must include attack_type, risk_level, indicators, why_risky, and learning_tip.
Assessment email JSON must stay concise and contain assessment fields only.
Phishing content must be simulated and use invented non-real domains.
Legitimate content must use hospital.org or moh.gov.sa only and must not request credentials, payment, MFA, OTP, bank details, or urgent account verification.
Beginner = obvious; Intermediate = partly convincing; Advanced = near-legitimate and subtle.
Analysis must link attack_type to role-context risk and should not rely only on Domain/Urgency/Spelling.
Arabic and English must have equal depth and quality.
""".strip()

# =============================================================
# END FINAL PATCH
# =============================================================

# =============================================================
# NOTE: Superseded generation-engine iterations removed (2026-07 cleanup)
# -------------------------------------------------------------
# The following abandoned engine iterations were deleted here because
# they were 100% dead code: every name they defined (generate_email,
# generate_assess_email, generate_other_email, generate_other_assess_email,
# get_generation_quality_issues, etc.) was re-defined again later in the
# file, so Python only ever executed the LAST definition (the v18 engine
# below). Removed blocks, in order: ROOT-CONTROLLED GENERATION ENGINE,
# ENHANCED HYBRID SCENARIO ENGINE v2, SCENARIO ENGINE v3, SCENARIO ENGINE v4,
# SCENARIO CONTENT ENGINE v5, EMAIL GENERATION ENGINE v6,
# API-FIRST DYNAMIC GENERATION ENGINE v10, FINAL REVIEW PATCH v11,
# FINAL STABLE PATCH v12, FINAL RESEARCH PATCH v15 (~3,850 lines total).
# None of this removal changes app behaviour: it was unreachable before
# the cleanup too.
# =============================================================
# =============================================================
# FINAL RESEARCH ENGINE v18 — ROLE × DIFFICULTY × HIGH-DIVERSITY
# -------------------------------------------------------------
# This engine is intentionally defined BEFORE Streamlit routing.
# It replaces the previous fixed-template patches with:
#   1) strict role alignment (clinical/admin/it/other mix),
#   2) strict progressive difficulty contracts,
#   3) combinatorial scenario generation (hundreds of blueprints),
#   4) multi-structure email composition,
#   5) cross-session anti-repeat history,
#   6) API wording generation with deterministic safe fallback.
# =============================================================

_V18_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_history_v18.json")

V18_ROLE_BANKS = {
    "clinical": {
        "easy": [
            ("Clinical Staff Portal", "basic account activation"), ("Shift Portal", "shift access check"),
            ("Hospital App", "mobile app access update"), ("Training Portal", "mandatory training login"),
            ("Clinic Account", "basic profile verification"), ("EMR Access", "simple account reset"),
            ("Staff Badge", "badge access confirmation"), ("Patient Portal", "basic sign-in verification"),
            ("Ward Schedule", "schedule access confirmation"), ("Clinical Email", "mailbox storage warning"),
            ("On-call Portal", "on-call access update"), ("Nursing Dashboard", "dashboard password reset"),
        ],
        "medium": [
            ("Emergency Department", "triage dashboard review"), ("Radiology", "PACS access verification"),
            ("Laboratory", "LIS result-routing update"), ("Pharmacy", "dispensing workflow review"),
            ("Outpatient Clinics", "appointment workflow revision"), ("ICU", "clinical handover update"),
            ("Infection Control", "policy acknowledgement"), ("Patient Safety", "incident portal review"),
            ("Blood Bank", "inventory access confirmation"), ("Operating Theatre", "theatre list review"),
            ("Dialysis Unit", "treatment schedule update"), ("Medical Records", "record access review"),
            ("Cardiology", "ECG workflow update"), ("Oncology", "treatment-plan portal review"),
            ("Respiratory Therapy", "device allocation review"), ("Endoscopy", "procedure-list confirmation"),
        ],
        "hard": [
            ("Blood Bank Transfusion Committee", "blood product release approval"),
            ("Radiology Informatics", "PACS reconciliation exception"),
            ("Clinical Governance", "sentinel event evidence review"),
            ("Antimicrobial Stewardship", "restricted-antibiotic approval"),
            ("Mortality and Morbidity Committee", "case review pack acknowledgement"),
            ("Research Ethics Office", "protocol deviation response"),
            ("Medication Safety Committee", "high-alert medication audit"),
            ("Operating Room Governance", "surgical safety checklist variance"),
            ("Pathology Quality Unit", "critical-result escalation audit"),
            ("Accreditation Office", "JCI tracer evidence submission"),
            ("Clinical Coding", "DRG case validation request"),
            ("Medical Affairs", "privileging document review"),
            ("Radiation Safety", "dosimetry compliance attestation"),
            ("Infection Prevention", "outbreak line-list review"),
            ("Transplant Coordination", "donor-case document verification"),
            ("Patient Experience", "formal complaint case response"),
        ],
    },
    "admin": {
        "easy": [
            ("Employee Portal", "profile update"), ("Payroll Portal", "salary account confirmation"),
            ("Leave System", "leave balance verification"), ("HR Account", "password reset"),
            ("Benefits Portal", "benefit enrollment check"), ("Training System", "course login update"),
            ("Attendance Portal", "timesheet verification"), ("Staff Directory", "contact details update"),
            ("Expense Portal", "basic reimbursement login"), ("Recruitment Portal", "candidate account update"),
            ("Policy Portal", "policy account activation"), ("Employee App", "mobile access confirmation"),
        ],
        "medium": [
            ("Human Resources", "annual appraisal workflow"), ("Finance", "expense claim review"),
            ("Procurement", "supplier onboarding verification"), ("Insurance Office", "coverage data review"),
            ("Medical Records Administration", "records retention acknowledgement"),
            ("Quality Office", "department audit submission"), ("Training and Development", "CME registration review"),
            ("Facilities", "work-order portal update"), ("Executive Office", "meeting action tracker"),
            ("Patient Relations", "complaint routing update"), ("Corporate Communications", "staff survey review"),
            ("Legal Affairs", "contract document acknowledgement"), ("Revenue Cycle", "billing exception review"),
            ("Admissions", "patient registration workflow"), ("Scheduling Office", "clinic capacity review"),
            ("Compliance", "annual declaration submission"),
        ],
        "hard": [
            ("Chief Executive Office", "confidential board pack review"),
            ("Internal Audit", "evidence request for control testing"),
            ("Procurement Governance", "single-source justification approval"),
            ("Finance Operations", "high-value payment exception"),
            ("Legal Affairs", "litigation hold acknowledgement"),
            ("Accreditation Program Office", "survey evidence submission"),
            ("Corporate Compliance", "conflict-of-interest attestation"),
            ("Revenue Integrity", "coding variance investigation"),
            ("Vendor Management", "banking detail change validation"),
            ("Board Secretariat", "committee resolution signature"),
            ("Risk Management", "enterprise risk register update"),
            ("Insurance Contracting", "payer agreement amendment"),
            ("Human Capital", "executive succession file review"),
            ("Strategic Planning", "restricted KPI dashboard access"),
            ("Data Protection Office", "privacy incident response pack"),
            ("Supply Chain", "critical shortage allocation approval"),
        ],
    },
    "it": {
        "easy": [
            ("IT Helpdesk", "password reset"), ("Email Support", "mailbox reactivation"),
            ("VPN Portal", "VPN sign-in verification"), ("Wi-Fi Portal", "wireless access update"),
            ("MFA Service", "MFA enrollment"), ("Account Support", "basic account reactivation"),
            ("Device Portal", "device login update"), ("Remote Access", "remote access confirmation"),
            ("Software Center", "basic software account update"), ("Service Desk", "ticket portal login"),
            ("Cloud Storage", "storage account verification"), ("Network Account", "password expiry notice"),
        ],
        "medium": [
            ("Enterprise Applications", "application access recertification"), ("Network Operations", "VPN profile migration"),
            ("Identity Management", "MFA token re-registration"), ("Microsoft 365 Support", "shared mailbox review"),
            ("Service Desk", "remote-support session approval"), ("Cybersecurity", "suspicious sign-in review"),
            ("Cloud Operations", "backup console verification"), ("Database Services", "maintenance-window acknowledgement"),
            ("Endpoint Management", "device compliance review"), ("Telephony", "softphone profile update"),
            ("PACS Support", "clinical imaging account review"), ("Integration Team", "interface monitoring update"),
            ("Data Center", "server access recertification"), ("Application Support", "license renewal review"),
            ("Digital Health", "portal migration confirmation"), ("Information Security", "policy attestation"),
        ],
        "hard": [
            ("Security Operations Center", "privileged access anomaly review"),
            ("Identity and Access Management", "emergency role-elevation approval"),
            ("Network Security", "firewall exception validation"),
            ("Cloud Security", "conditional-access policy exception"),
            ("Database Administration", "production credential rotation"),
            ("Infrastructure Architecture", "certificate chain remediation"),
            ("Incident Response", "forensic evidence handoff"),
            ("Enterprise Integration", "HL7 interface certificate renewal"),
            ("PACS Infrastructure", "DICOM routing exception"),
            ("Business Continuity", "disaster recovery failover approval"),
            ("Privileged Access Management", "vault checkout reconciliation"),
            ("Security Governance", "third-party risk exception"),
            ("Endpoint Detection", "quarantine release authorization"),
            ("Cloud Platform", "tenant federation change review"),
            ("Data Protection", "encryption key escrow verification"),
            ("Change Advisory Board", "emergency production change approval"),
        ],
    },
}

V18_ATTACKS = {
    "easy": ["credential_request", "password_reset", "pin_verification", "account_reactivation", "fake_login"],
    "medium": ["credential_request", "fake_survey", "shared_document", "reply_information", "pdf_review", "login_portal"],
    "hard": ["sharepoint", "microsoft365", "qr", "pdf", "excel", "reply_chain", "docusign", "calendar_invite"],
}

V18_STRUCTURES = {
    "easy": ["problem_action_link", "warning_link_consequence", "notice_action_deadline", "short_alert", "support_request", "account_status"],
    "medium": ["context_request_deadline", "department_notice", "workflow_exception", "review_and_confirm", "service_change", "audit_followup", "two_paragraph_professional"],
    "hard": ["internal_memo", "reply_thread", "committee_request", "document_share", "calendar_followup", "audit_evidence", "executive_request", "system_notification"],
}

V18_SUBJECT_PREFIX = {
    "easy": ["Urgent:", "Important:", "Action required:", "Account alert:", "Final notice:", "Immediate action:"],
    "medium": ["Reminder:", "Workflow notice:", "Review required:", "Department update:", "Access review:", "Follow-up:"],
    "hard": ["Re:", "Fwd:", "For review:", "Action item:", "Committee follow-up:", "Document shared:", "Meeting follow-up:", "Exception review:"],
}

V18_TIME_OPTIONS = {
    "easy": ["Today, 8:15 AM", "Today, 10:42 AM", "Today, 2:31 PM", "Yesterday, 4:05 PM", "Monday, 7:03 AM", "Thursday, 4:07 PM"],
    "medium": ["Monday, 9:18 AM", "Tuesday, 1:26 PM", "Wednesday, 11:04 AM", "Thursday, 8:47 AM", "Friday, 2:12 PM", "Yesterday, 3:39 PM"],
    "hard": ["Mon 8:13 AM", "Tue 3:47 PM", "Wed 10:06 AM", "Thu 4:22 PM", "Fri 7:56 AM", "Yesterday 5:14 PM", "Last Friday 2:08 PM"],
}

V18_DOMAINS = {
    "easy": ["secure-staff-verify.net", "hospital-access-check.com", "portal-verify-now.net", "account-update-care.org", "staff-login-update.co", "secure-login-alert.net"],
    "medium": ["hospital-services.org.co", "staffportal-health.net", "clinical-workflow.org", "healthsystems-support.com", "hospital-share.net", "care-portal-services.com"],
    "hard": ["hospitalorg-support.com", "m365-hospital.org", "sharepoint-hospital.net", "hospital-docs.co", "secure-hospitaloffice.com", "hospital-cloud.org"],
}

# =============================================================
# LIBRARY-DRIVEN DIVERSITY SYSTEM v19
# -------------------------------------------------------------
# Design agreed with the researcher: instead of handing the API a
# finished seed sentence and asking it to "reword it" (which kept
# collapsing onto the same handful of shapes), each generated email
# is now assembled from FIVE independent seed libraries — subject,
# opening angle, content angle (V18_ROLE_BANKS, already existed),
# urgency reason, and the existing sender/recipient/signature pools.
# Every library is split by role_type x difficulty (a strict grid —
# picking for "clinical/easy" can never reach into "it/hard"), and
# every pick is drawn through the session-scoped anti-repeat helper
# (_v18_no_repeat_choice) so the SAME user never sees the same seed
# twice in one session. The API receives the seeds as creative
# DIRECTION (not literal text) and is instructed to write the actual
# subject/body wording itself, constrained by V19_DIFFICULTY_CONTRACT
# below — so the AI still does the real writing (per the project's
# core idea), it just starts from a much richer, role-and-difficulty
# -correct starting point instead of one fixed template per cell.
# =============================================================

# Full difficulty contract (matches the researcher's own table):
# link visibility, greeting type, language register, how directly
# credentials are requested, the urgency style, indicator-count
# range, attachment policy, QR policy, MS365/Outlook usage, and how
# close the fake login page is to the real one. Used both to build
# the API prompt's rules section and to validate/repair the result.
V19_DIFFICULTY_CONTRACT = {
    "easy": {
        "link_style": "very obvious fake link (unrelated or misspelled domain)",
        "greeting": "generic (\"Dear Staff\" style, no name)",
        "language": "direct and simple",
        "credential_request": "direct — asks for the password/PIN/OTP outright",
        "urgency_style": "blunt words like \"Immediately\" or \"Today\"",
        "indicator_range": (4, 5),
        "attachment": "never",
        "qr": "never",
        "ms365_outlook": "never",
        "login_page_realism": "simple, does not resemble the real system",
    },
    "medium": {
        "link_style": "looks fairly official, close to the real domain",
        "greeting": "role/department based (e.g. \"Dear Radiology Team\")",
        "language": "more professional",
        "credential_request": "asks the user to sign in / verify — not a bare password request",
        "urgency_style": "a reasonable deadline, not a bare command",
        "indicator_range": (3, 4),
        "attachment": "sometimes a plain PDF",
        "qr": "never",
        "ms365_outlook": "never",
        "login_page_realism": "resembles the real system's look",
    },
    "hard": {
        "link_style": "very close to the official one, may use a shortener",
        "greeting": "the real person's name",
        "language": "very natural, reads like an internal message",
        "credential_request": "review/approve a document instead of a direct login",
        "urgency_style": "a logical business reason (audit, policy, patient safety), not a threat",
        "indicator_range": (1, 2),
        "attachment": "PDF, Excel, or SharePoint depending on the scenario",
        "qr": "sometimes, when the scenario calls for it",
        "ms365_outlook": "sometimes",
        "login_page_realism": "near-identical copy of the real one",
    },
}

# --- Opening-angle seeds: the DIRECTION for the first 1-2 sentences,
# tailored per role (clinical/admin/it) AND per difficulty. These are
# not final sentences — the API rewrites them into real prose in the
# requested language; the deterministic fallback path turns them into
# a simple sentence directly when the API is unavailable.
V19_OPENING_SEEDS = {
    "clinical": {
        "easy": [
            "a routine system check flagged a problem with the account",
            "access was automatically paused pending a quick check",
            "the mobile app logged an unusual sign-in attempt",
            "a scheduled maintenance step needs the user's confirmation",
            "the account was flagged for a missing verification step",
        ],
        "medium": [
            "a scheduled review of clinical system access identified an item needing confirmation",
            "the department's monthly access audit found an outstanding item",
            "a recent system upgrade requires staff to reconfirm their access",
            "clinical governance requested a routine credential check for this unit",
            "the on-call roster system flagged an inconsistency that needs review",
        ],
        "hard": [
            "following up on the item discussed at this week's unit meeting",
            "the accreditation team asked for confirmation on this file before the tracer visit",
            "sharing the file we discussed for your sign-off before the committee meets",
            "a minor exception was logged during the last audit cycle and needs your input",
            "the case reference from the last handover still needs a confirmation from your side",
        ],
    },
    "admin": {
        "easy": [
            "the payroll system flagged an incomplete profile step",
            "a routine HR check found a missing confirmation on file",
            "the leave-management app needs a quick account check",
            "the directory sync job flagged an outdated profile entry",
            "a scheduled account cleanup needs the user to confirm details",
        ],
        "medium": [
            "the finance team's quarterly access review found an item needing confirmation",
            "a recent policy update requires staff to reconfirm their account details",
            "the procurement portal migration needs a short confirmation from your side",
            "HR's routine audit flagged an outstanding item on this account",
            "the department's compliance check found a pending action item",
        ],
        "hard": [
            "following up on the file shared during this week's management meeting",
            "internal audit asked for a quick confirmation before closing this item",
            "sharing the document we discussed for your review before the board pack is finalised",
            "a minor exception surfaced during the procurement review and needs your input",
            "the finance exception noted last week still needs a confirmation from your office",
        ],
    },
    "it": {
        "easy": [
            "the security system flagged an unusual sign-in on the account",
            "a routine password-policy check found an expired credential",
            "the VPN service logged a failed authentication attempt",
            "the helpdesk queue has a pending action item on this account",
            "the mailbox quota system flagged the account for review",
        ],
        "medium": [
            "the quarterly access-recertification review found an item needing confirmation",
            "a scheduled identity-management migration requires staff to reconfirm access",
            "the security team's routine review flagged an outstanding item on this account",
            "a recent policy change requires re-authentication for privileged accounts",
            "the monitoring system logged an anomaly that needs a quick confirmation",
        ],
        "hard": [
            "following up on the change request discussed in this week's CAB meeting",
            "security operations asked for confirmation before closing this ticket",
            "sharing the access-review file we discussed for your sign-off",
            "a minor exception was logged during the last security audit and needs your input",
            "the incident reference from last week still needs a confirmation from your side",
        ],
    },
}

# --- Urgency-reason seeds: mirrors the researcher's own contract —
# Easy is a blunt command word, Medium is a believable deadline,
# Hard is a logical business reason (never a bare threat).
V19_URGENCY_SEEDS = {
    "clinical": {
        "easy": ["immediately", "today", "within the hour", "before your next shift"],
        "medium": ["within 24 hours", "before Thursday's shift handover", "by the end of the business day", "before the next audit cycle"],
        "hard": ["ahead of the accreditation tracer visit", "per the patient-safety review timeline", "before the committee finalises this cycle's file", "as part of the routine governance schedule"],
    },
    "admin": {
        "easy": ["immediately", "today", "within the hour", "before close of business"],
        "medium": ["within 24 hours", "before the next payroll run", "by the end of the business day", "before the quarterly close"],
        "hard": ["ahead of the internal audit deadline", "per this quarter's compliance timeline", "before the board pack is finalised", "as part of the routine procurement schedule"],
    },
    "it": {
        "easy": ["immediately", "today", "within the hour", "before your account locks"],
        "medium": ["within 24 hours", "before the scheduled migration window", "by the end of the business day", "before the next recertification cycle"],
        "hard": ["ahead of the change-advisory-board deadline", "per this quarter's security review timeline", "before the ticket is auto-closed", "as part of the routine access-audit schedule"],
    },
}

# --- Subject-line templates: kept difficulty-specific rather than a
# 3x3x3 grid, since the {area}/{topic} substitution (already fully
# role-specific via V18_ROLE_BANKS) is what actually varies the
# wording per role; the TEMPLATE SHAPE only needs to vary by
# difficulty (easy = blunt, medium = professional, hard = internal).
V19_SUBJECT_TEMPLATES = {
    "easy": [
        "Immediate action needed: {topic}",
        "Your {topic} is on hold",
        "We could not verify your {topic}",
        "{topic} — action required today",
        "Reminder: {topic} pending",
        "Access alert: {topic}",
    ],
    "medium": [
        "{area}: {topic} requires your review",
        "Update on {topic} — {area}",
        "Follow-up needed: {topic}",
        "{area} notice regarding {topic}",
        "Please confirm: {topic}",
        "Workflow change — {topic}",
    ],
    "hard": [
        "Re: {topic}",
        "{topic} — evidence requested",
        "Follow-up on our {area} discussion",
        "{area}: outstanding item — {topic}",
        "For your review: {topic}",
        "{topic} — sign-off needed",
    ],
}

# Arabic urgency phrases for the deterministic fallback path (used
# when the API is unavailable and the language is Arabic). Kept
# role-shared to control scope, but still difficulty-correct: Easy is
# a blunt command, Medium a believable deadline, Hard a logical
# business reason rather than a threat.
V19_URGENCY_SEEDS_AR = {
    "easy": ["فورًا", "اليوم", "خلال الساعة القادمة", "قبل نهاية الدوام"],
    "medium": ["خلال 24 ساعة", "قبل نهاية يوم العمل", "قبل نهاية الأسبوع", "قبل دورة المراجعة القادمة"],
    "hard": ["قبل زيارة الاعتماد القادمة", "ضمن الجدول الزمني لمراجعة السلامة", "قبل إغلاق الملف من اللجنة", "ضمن الجدول الدوري المعتاد للحوكمة"],
}

# Arabic subject-line templates (fallback path). NOTE: {topic}/{area}
# still come from V18_ROLE_BANKS, which is English-only, so the topic
# text itself stays in English inside the Arabic sentence — the same
# accepted trade-off already used throughout the Arabic body templates
# below. Fully bilingual role banks would remove this, but that is a
# separate, larger content task.
V19_SUBJECT_TEMPLATES_AR = {
    "easy": [
        "إجراء فوري مطلوب: {topic}",
        "حسابك بخصوص {topic} معلّق",
        "تعذر التحقق من {topic}",
        "{topic} — مطلوب إجراء اليوم",
        "تذكير: {topic} بانتظار الإجراء",
    ],
    "medium": [
        "{area}: {topic} بحاجة لمراجعتكم",
        "تحديث بخصوص {topic} — {area}",
        "متابعة مطلوبة: {topic}",
        "إشعار من {area} بخصوص {topic}",
        "يرجى التأكيد: {topic}",
    ],
    "hard": [
        "رد: {topic}",
        "{topic} — مطلوب دليل",
        "متابعة لنقاشنا في {area}",
        "{area}: بند معلّق — {topic}",
        "للمراجعة: {topic}",
    ],
}

# FIXED: why_risky / learning_tip used to be ONE hardcoded sentence per
# language, reused verbatim on every single fallback email — exactly
# the repetition the researcher flagged in the "AI Tutor Analysis"
# panel across otherwise-different emails. Each difficulty now has its
# own small pool, picked with the same anti-repeat helper as everything
# else.
V19_WHY_RISKY_EN = {
    "easy": [
        "This training phishing email uses an obviously unofficial domain and a direct credential request to pressure a quick click.",
        "The message combines a fake sender domain with blunt urgency, a classic pattern used to rush the reader past normal checks.",
        "This is a simulated phishing email: the sender domain, the direct data request, and the same-day deadline are all textbook red flags.",
        "The generic greeting, the obvious domain, and the direct password request together make this an easy-to-spot simulated phish.",
        "This training email relies on fear of losing access rather than a believable scenario — a common beginner-level tactic.",
        "Nothing here matches how the real hospital system communicates: the domain, the urgency, and the direct request all stand out.",
    ],
    "medium": [
        "This training phishing email uses role-specific context and a plausible-but-unofficial domain to make the request feel routine.",
        "The message mimics a normal workflow request, which is what makes it more convincing than an obvious scam attempt.",
        "This simulated email relies on a believable deadline and department-style wording rather than an outright threat.",
        "The professional tone and department framing are designed to lower the reader's guard compared to a blunt phishing attempt.",
        "This training email avoids asking for a password directly, relying instead on a sign-in page to capture credentials.",
        "The domain looks close to legitimate, which is exactly what makes medium-difficulty phishing harder to catch at a glance.",
    ],
    "hard": [
        "This training phishing email is deliberately realistic: the channel and destination are the only things that differ from the real internal process.",
        "The message uses credible internal context and a near-official domain, which is what makes advanced phishing hard to catch.",
        "This simulated email mirrors a genuine internal request closely — the sender's channel is the main giveaway here.",
        "The personalised greeting and internal framing make this training email nearly indistinguishable from a real message.",
        "This is designed to test attention to detail: the request itself is plausible, and only the destination is wrong.",
        "The logical business reason given here (audit, policy, or safety) is what makes advanced phishing convincing rather than alarming.",
    ],
}
V19_WHY_RISKY_AR = {
    "easy": [
        "هذه رسالة تصيد تدريبية تستخدم نطاقاً غير رسمي واضحاً وطلباً مباشراً للبيانات للضغط على المستخدم للتصرف بسرعة.",
        "تجمع الرسالة بين نطاق مرسل مزيف وإلحاح مباشر، وهو نمط كلاسيكي يُستخدم لتجاوز الفحص الطبيعي للرسالة.",
        "هذه رسالة تصيد تدريبية: نطاق المرسل والطلب المباشر للبيانات والمهلة الزمنية لنفس اليوم كلها مؤشرات واضحة.",
        "التحية العامة والنطاق الواضح والطلب المباشر لكلمة المرور تجعل هذه الرسالة سهلة الاكتشاف كتصيد تدريبي.",
        "تعتمد هذه الرسالة التدريبية على الخوف من فقدان الوصول بدل سيناريو مقنع، وهو أسلوب شائع بالمستوى المبتدئ.",
        "لا شيء هنا يطابق طريقة تواصل النظام الحقيقي بالمستشفى: النطاق والإلحاح والطلب المباشر كلها بارزة.",
    ],
    "medium": [
        "تستخدم هذه الرسالة التدريبية سياقاً مرتبطاً بالوظيفة ونطاقاً يبدو مقبولاً لكنه غير رسمي، مما يجعل الطلب يبدو روتينياً.",
        "تحاكي الرسالة طلب عمل اعتيادي، وهذا بالضبط ما يجعلها أكثر إقناعاً من محاولة احتيال واضحة.",
        "تعتمد هذه الرسالة التدريبية على مهلة زمنية معقولة وصياغة شبيهة بمراسلات القسم بدل التهديد المباشر.",
        "النبرة المهنية والإطار القسمي مصممان لتقليل حذر القارئ مقارنة بمحاولة تصيد مباشرة وواضحة.",
        "تتجنب هذه الرسالة التدريبية طلب كلمة المرور مباشرة، وتعتمد بدلاً من ذلك على صفحة تسجيل دخول لالتقاط البيانات.",
        "النطاق يبدو قريباً من الرسمي، وهذا بالضبط ما يجعل تصيد المستوى المتوسط أصعب اكتشافاً من النظرة الأولى.",
    ],
    "hard": [
        "هذه رسالة تصيد تدريبية واقعية عمداً: الفرق الوحيد عن الإجراء الداخلي الحقيقي هو القناة والوجهة.",
        "تستخدم الرسالة سياقاً داخلياً موثوقاً ونطاقاً قريباً جداً من الرسمي، وهذا ما يصعّب اكتشاف التصيد المتقدم.",
        "تحاكي هذه الرسالة التدريبية طلباً داخلياً حقيقياً بدقة عالية — قناة المرسل هي المؤشر الأساسي هنا.",
        "التحية الشخصية والإطار الداخلي يجعلان هذه الرسالة التدريبية شبه مطابقة لرسالة حقيقية.",
        "صُممت هذه الرسالة لاختبار الانتباه للتفاصيل: الطلب نفسه معقول، والوجهة فقط هي الخاطئة.",
        "السبب المنطقي المذكور هنا (تدقيق، سياسة، أو سلامة) هو ما يجعل التصيد المتقدم مقنعاً بدل مثير للريبة.",
    ],
}
V19_LEARNING_TIP_EN = {
    "easy": [
        "Verify the request through official channels and avoid using links in unexpected messages.",
        "Check the sender's domain carefully before clicking anything, and never enter your password from an email link.",
        "When a message pressures you to act 'today' or 'immediately', pause and verify through a known official channel first.",
        "Legitimate systems rarely ask for your password directly by email — treat any such request as suspicious.",
        "If an email creates panic about losing access, that pressure itself is a warning sign worth pausing on.",
        "Compare the sender's address letter by letter with the real hospital domain before trusting the message.",
    ],
    "medium": [
        "Even professional-looking requests should be verified directly with the department before you act on them.",
        "A plausible deadline is not proof of legitimacy — confirm unusual requests through the official directory.",
        "Hover over links before clicking, and compare the domain carefully against the organization's real one.",
        "A sign-in page reached through an email link should never be trusted over the official bookmarked portal.",
        "Department-style wording can still be faked — call the department directly if a request feels unusual.",
        "Take a moment to check whether this kind of request is normally made by email at all before acting.",
    ],
    "hard": [
        "Realistic internal wording is not enough — always confirm the channel matches how that request is normally made.",
        "For sensitive requests, verify directly with the person or department through a known number or system, not by replying to the email.",
        "The more convincing a message looks, the more important it is to double-check the actual link destination.",
        "A logical-sounding business reason (audit, policy, safety) is not proof of legitimacy on its own.",
        "If a message references a real meeting or file, confirm it through that same channel rather than the email link.",
        "Advanced phishing relies on trust in context — when in doubt, verify out-of-band rather than by replying.",
    ],
}
V19_LEARNING_TIP_AR = {
    "easy": [
        "تحقق من الطلب عبر القنوات الرسمية ولا تستخدم الروابط الواردة في الرسائل غير المتوقعة.",
        "تفحّص نطاق المرسل بعناية قبل الضغط على أي شيء، ولا تُدخل كلمة المرور أبداً من رابط داخل بريد إلكتروني.",
        "عندما تضغط عليك الرسالة للتصرف \"اليوم\" أو \"فوراً\"، توقف وتحقق عبر قناة رسمية معروفة أولاً.",
        "الأنظمة الشرعية نادراً ما تطلب كلمة المرور مباشرة عبر البريد — تعامل مع أي طلب كهذا بحذر.",
        "لو رسالة تسبب لك قلقاً من فقدان الوصول، هذا الضغط بحد ذاته إشارة تستحق التوقف عندها.",
        "قارن عنوان المرسل حرفاً بحرف مع نطاق المستشفى الحقيقي قبل الوثوق بالرسالة.",
    ],
    "medium": [
        "حتى الطلبات التي تبدو مهنية يجب التحقق منها مباشرة مع القسم قبل تنفيذها.",
        "المهلة الزمنية المعقولة ليست دليلاً على الشرعية — تحقق من الطلبات غير المعتادة عبر الدليل الرسمي.",
        "مرّر المؤشر فوق الرابط قبل الضغط عليه، وقارن النطاق بعناية مع نطاق الجهة الحقيقي.",
        "صفحة تسجيل الدخول المفتوحة من رابط بريد إلكتروني لا يجب الوثوق بها أبداً مقارنة بالبوابة الرسمية المحفوظة.",
        "الصياغة الشبيهة بمراسلات القسم يمكن تقليدها — اتصل بالقسم مباشرة لو الطلب بدا غير معتاد.",
        "خذ لحظة للتفكير هل هذا النوع من الطلبات يتم عادة عبر البريد الإلكتروني أصلاً قبل التصرف.",
    ],
    "hard": [
        "الصياغة الداخلية الواقعية لا تكفي — تأكد دائماً أن القناة تطابق الطريقة المعتادة لهذا النوع من الطلبات.",
        "بالنسبة للطلبات الحساسة، تحقق مباشرة مع الشخص أو القسم عبر رقم أو نظام معروف، لا بالرد على البريد.",
        "كلما بدت الرسالة مقنعة أكثر، زادت أهمية التحقق من وجهة الرابط الفعلية.",
        "السبب المنطقي الظاهري (تدقيق، سياسة، سلامة) ليس دليلاً كافياً على الشرعية بمفرده.",
        "لو الرسالة تشير لاجتماع أو ملف حقيقي، تحقق منه عبر نفس القناة بدل رابط البريد.",
        "التصيد المتقدم يعتمد على الثقة بالسياق — عند الشك، تحقق من قناة أخرى بدل الرد على الرسالة.",
    ],
}


def _v18_load_history():
    try:
        with open(_V18_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _v18_save_history(items):
    try:
        with open(_V18_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(items[-2000:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _v18_role_type(role):
    return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]


def _v18_effective_role(role, index, phase):
    rt = _v18_role_type(role)
    if rt != "other":
        return rt
    key = f"v18_other_roles_{phase}_{st.session_state.get('v18_cycle_id', 0)}"
    if key not in st.session_state:
        seq = ["clinical", "admin", "it", "clinical", "admin", "it"]
        random.shuffle(seq)
        st.session_state[key] = seq
    seq = st.session_state[key]
    return seq[index % len(seq)]


def _v18_rng(index, role_type, difficulty, phase):
    nonce = st.session_state.get("v18_cycle_id")
    if nonce is None:
        nonce = random.randint(100000, 999999)
        st.session_state["v18_cycle_id"] = nonce
    seed = f"{nonce}|{index}|{role_type}|{difficulty}|{phase}|{random.random()}"
    return random.Random(seed)


def _v18_no_repeat_choice(items, memory_key, rng=None):
    """FIXED: shared, session-guaranteed anti-repeat picker. Streamlit's
    st.session_state always works within a browser session (unlike writing
    to a file next to the script, which silently fails on read-only/ephemeral
    deployments such as Streamlit Community Cloud — see the note on
    _v18_pick_blueprint below). Avoids repeating a value seen recently in
    THIS session before falling back to allowing repeats once every option
    has been used.
    """
    recent = st.session_state.get(memory_key, [])
    pool = [x for x in items if x not in recent]
    if not pool:
        pool = list(items)
        recent = []
    choice = (rng or random).choice(pool)
    st.session_state[memory_key] = (recent + [choice])[-(max(1, len(items) - 1)):]
    return choice


def _v18_pick_blueprint(role, index, language, difficulty, phase, is_phishing=True):
    difficulty = str(difficulty or "medium").lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    role_type = _v18_effective_role(role, index, phase)
    rng = _v18_rng(index, role_type, difficulty, phase)
    bank = V18_ROLE_BANKS[role_type][difficulty]

    # FIXED (root cause of repeated scenarios/domains across a 6-example
    # batch): this used to rely ONLY on a JSON file written next to the
    # script (scenario_history_v18.json) to avoid repeats. On Streamlit
    # Community Cloud that directory is frequently read-only or reset on
    # restart, so _v18_save_history() was failing silently every time
    # (wrapped in try/except: pass) and _v18_load_history() always came
    # back empty — meaning the "used" set was ALWAYS empty and every pick
    # was fully random with no memory at all. That is exactly why the same
    # domain and the same scenario could appear two or three times in one
    # six-example session. We now track "used" primarily in
    # st.session_state (guaranteed to work, scoped to this cycle), and keep
    # the on-disk file only as a secondary best-effort layer for repeat
    # visits across sessions on setups where the disk is actually writable.
    cycle_key = f"v18_used_signatures_{st.session_state.get('v18_cycle_id', 0)}"
    used_session = set(st.session_state.get(cycle_key, []))
    history = _v18_load_history()
    used_file = {x.get("signature") for x in history[-600:] if isinstance(x, dict)}
    used = used_session | used_file

    candidates = []
    for area, topic in bank:
        for attack in V18_ATTACKS[difficulty]:
            for structure in V18_STRUCTURES[difficulty]:
                sig = f"{role_type}|{difficulty}|{area}|{topic}|{attack}|{structure}"
                candidates.append((sig, area, topic, attack, structure))
    rng.shuffle(candidates)
    chosen = next((c for c in candidates if c[0] not in used), candidates[0])
    sig, area, topic, attack, structure = chosen

    st.session_state[cycle_key] = list(used_session | {sig})
    history.append({"signature": sig, "role": role_type, "difficulty": difficulty, "phase": phase})
    _v18_save_history(history)

    # FIXED: domain / subject prefix / display time previously had NO
    # anti-repeat protection at all (plain rng.choice every time) — with
    # pools of only 6 options and 6 examples per session, repeats were
    # statistically expected, not a fluke.
    domain = _v18_no_repeat_choice(V18_DOMAINS[difficulty], f"v18_recent_domain_{difficulty}_{st.session_state.get('v18_cycle_id', 0)}", rng)
    prefix = _v18_no_repeat_choice(V18_SUBJECT_PREFIX[difficulty], f"v18_recent_prefix_{difficulty}_{st.session_state.get('v18_cycle_id', 0)}", rng)
    display_time = _v18_no_repeat_choice(V18_TIME_OPTIONS[difficulty], f"v18_recent_time_{difficulty}_{st.session_state.get('v18_cycle_id', 0)}", rng)

    # v19 library-driven seeds — each pick is strictly scoped to
    # role_type x difficulty (a separate list per cell, never mixed
    # across roles) and anti-repeat protected within this session.
    cid = st.session_state.get('v18_cycle_id', 0)
    subject_template = _v18_no_repeat_choice(
        V19_SUBJECT_TEMPLATES[difficulty], f"v19_recent_subject_{difficulty}_{cid}", rng)
    subject_template_ar = _v18_no_repeat_choice(
        V19_SUBJECT_TEMPLATES_AR[difficulty], f"v19_recent_subject_ar_{difficulty}_{cid}", rng)
    opening_seed = _v18_no_repeat_choice(
        V19_OPENING_SEEDS[role_type][difficulty], f"v19_recent_opening_{role_type}_{difficulty}_{cid}", rng)
    urgency_seed = _v18_no_repeat_choice(
        V19_URGENCY_SEEDS[role_type][difficulty], f"v19_recent_urgency_{role_type}_{difficulty}_{cid}", rng)
    urgency_seed_ar = _v18_no_repeat_choice(
        V19_URGENCY_SEEDS_AR[difficulty], f"v19_recent_urgency_ar_{difficulty}_{cid}", rng)

    return {
        "signature": sig, "role_type": role_type, "difficulty": difficulty,
        "area": area, "topic": topic, "attack": attack, "structure": structure,
        "domain": domain, "subject_prefix": prefix, "display_time": display_time,
        "subject_template": subject_template, "subject_template_ar": subject_template_ar,
        "opening_seed": opening_seed,
        "urgency_seed": urgency_seed, "urgency_seed_ar": urgency_seed_ar,
        "language": language, "is_phishing": bool(is_phishing), "phase": phase,
    }


def _v18_recipient(role_type, index, phase):
    pool = RECIPIENT_POOLS.get(role_type) or RECIPIENT_POOLS.get("clinical") or []
    if not pool:
        return {"en": "Dr. Sara Almutairi", "ar": "د. سارة المطيري", "email": "dr.sara.almutairi@hospital.org"}
    key = f"v18_recipient_{role_type}_{phase}_{st.session_state.get('v18_cycle_id', 0)}"
    order = get_session_random_order(len(pool), key)
    return pool[order[index % len(order)]]


def _v18_sender(area, domain, difficulty, rng):
    local_easy = ["support", "alerts", "verify", "account-update", "secure-login", "staff-support"]
    local_med = ["notifications", "workflow", "service-desk", "no-reply", "department-updates", "portal"]
    local_hard = ["operations", "governance", "office", "coordinator", "secretariat", "systems"]
    pool = local_easy if difficulty == "easy" else local_med if difficulty == "medium" else local_hard
    # FIXED: same missing anti-repeat issue as domain/prefix/time above.
    local = _v18_no_repeat_choice(pool, f"v18_recent_sender_{difficulty}_{st.session_state.get('v18_cycle_id', 0)}", rng)
    return f"{area} <{local}@{domain}>"


def _v18_link(bp, index):
    slug = re.sub(r"[^a-z0-9]+", "-", bp["topic"].lower()).strip("-")
    if bp["difficulty"] == "hard" and bp["attack"] == "sharepoint":
        return f"https://{bp['domain']}/sites/{slug}/Shared%20Documents/{100+index}"
    if bp["difficulty"] == "hard" and bp["attack"] == "microsoft365":
        return f"https://{bp['domain']}/m365/{slug}?ref={700+index}"
    return f"http://{bp['domain']}/{slug}/{100+index}"


# FIXED: the easy-difficulty greeting pool used to have only 3 fixed
# strings, AND (see _v18_enforce below) the enforcement step forced the
# body's first line to be exactly the first of those 3 strings almost every
# time — which is why every screenshot showed "Dear Staff," no matter what.
# The pool is now bigger, and both the deterministic path (_v18_greeting)
# and the AI-enforcement path pull from the SAME pool using the SAME
# session-scoped anti-repeat memory, so they can no longer disagree or
# collapse onto one value.
V18_EASY_GREETINGS_EN = ["Dear Staff,", "Dear Healthcare Team,", "Dear Employee,", "Dear Colleague,", "Dear Team Member,", "Hello Staff,"]
V18_EASY_GREETINGS_AR = ["عزيزي الموظف،", "فريق العمل العزيز،", "الزملاء الأعزاء،", "عزيزي الزميل،", "أعضاء الفريق الأعزاء،", "مرحباً بكم،"]


def _v18_greeting(bp, person, rng):
    is_ar = bp["language"] == "Arabic"
    if bp["difficulty"] == "easy":
        pool = V18_EASY_GREETINGS_AR if is_ar else V18_EASY_GREETINGS_EN
        key = f"v18_recent_greeting_easy_{bp['language']}_{st.session_state.get('v18_cycle_id', 0)}"
        return _v18_no_repeat_choice(pool, key, rng)
    if bp["difficulty"] == "medium":
        return f"عزيزي فريق {bp['area']}،" if is_ar else f"Dear {bp['area']} Team,"
    name = person.get("ar") if is_ar else person.get("en")
    return f"عزيزي/عزيزتي {name}،" if is_ar else f"Dear {name},"


def _v18_indicator_objects(bp, link, is_phishing=True):
    is_ar = bp["language"] == "Arabic"
    if not is_phishing:
        return []
    # Indicator COUNT now varies within the contracted range (easy 4-5,
    # medium 3-4, hard 1-2) instead of always being the same fixed
    # number every time, per the researcher's difficulty table. The
    # coin flip is seeded off this email's own signature so it's
    # reproducible for a given seed but varies across different emails.
    extra = random.Random(bp.get("signature", "") + "_indicator_count").random() < 0.5
    if bp["difficulty"] == "easy":
        rows = [
            ("نطاق مرسل مزيف وواضح", "المرسل لا يستخدم نطاق المستشفى الرسمي."),
            ("طلب مباشر لبيانات الدخول", "تطلب الرسالة كلمة المرور أو الرمز أو بيانات الحساب مباشرة."),
            ("إلحاح أو تهديد واضح", "تضغط الرسالة للتصرف اليوم أو خلال ساعات."),
            ("رابط خارجي ظاهر", f"الرابط يقود إلى نطاق غير رسمي: {link}"),
            ("تحية عامة غير شخصية", "الرسالة لا تخاطبك باسمك، بل بصيغة عامة تدل على إرسال جماعي."),
        ] if is_ar else [
            ("Obvious fake sender domain", "The sender does not use the hospital's official domain."),
            ("Direct credential request", "The message directly asks for a password, PIN, OTP, or login details."),
            ("Strong urgency or threat", "The email pressures the user to act today or within hours."),
            ("Visible external link", f"The link points to a non-official domain: {link}"),
            ("Generic, impersonal greeting", "The message does not address you by name, suggesting a mass mailing."),
        ]
        rows = rows[:4] + (rows[4:5] if extra else [])
    elif bp["difficulty"] == "medium":
        rows = [
            ("نطاق يبدو رسميًا لكنه غير مطابق", "النطاق قريب من نطاق المستشفى لكنه ليس النطاق المعتمد."),
            ("طلب غير معتاد ضمن سير العمل", "الطلب يبدو مهنيًا لكنه يطلب إجراءً لا يتم عادة عبر البريد."),
            ("مهلة زمنية مقنعة", "تستخدم الرسالة موعدًا معقولًا للضغط دون تهديد مباشر."),
            ("رابط لا يطابق الوجهة المتوقعة", "الرابط الظاهر لا يقود لنفس النظام الذي تتحدث عنه الرسالة."),
        ] if is_ar else [
            ("Plausible but incorrect domain", "The domain looks professional but does not match the approved hospital domain."),
            ("Unusual workflow request", "The request sounds legitimate but asks for an action not normally completed by email."),
            ("Plausible deadline pressure", "The message uses a believable deadline rather than an obvious threat."),
            ("Link destination mismatch", "The visible link does not point to the same system the email refers to."),
        ]
        rows = rows[:3] + (rows[3:4] if extra else [])
    else:
        rows = [
            ("اختلاف دقيق في القناة", "الطلب مهني جدًا لكن القناة أو الرابط لا يطابقان الإجراء الداخلي المعتاد."),
            ("طلب حساس بصياغة مقنعة", "تستخدم الرسالة سياقًا داخليًا لإقناع المستلم بتنفيذ إجراء حساس."),
        ] if is_ar else [
            ("Subtle channel mismatch", "The message is highly realistic, but the channel or destination differs from the normal internal process."),
            ("Sensitive request in a convincing context", "The email uses credible internal context to encourage a sensitive action."),
        ]
        rows = rows[:1] if not extra else rows[:2]
    return [{"number": i+1, "title": t, "description": d} for i, (t, d) in enumerate(rows)]


def _v18_fallback_phishing(bp, role, index):
    rng = _v18_rng(index, bp["role_type"], bp["difficulty"], bp["phase"] + "_fallback")
    person = _v18_recipient(bp["role_type"], index, bp["phase"])
    greeting = _v18_greeting(bp, person, rng)
    link = _v18_link(bp, index)
    sender = _v18_sender(bp["area"], bp["domain"], bp["difficulty"], rng)
    is_ar = bp["language"] == "Arabic"
    subject = (bp["subject_template_ar"] if is_ar else bp["subject_template"]).format(area=bp["area"], topic=bp["topic"])
    area, topic = bp["area"], bp["topic"]
    structure = bp["structure"]

    if bp["difficulty"] == "easy":
        action_en = rng.choice(["verify your password today", "confirm your staff PIN immediately", "update your login credentials now", "enter your OTP to restore access"])
        action_ar = rng.choice(["تحقق من كلمة المرور اليوم", "أكّد الرقم السري للموظف فورًا", "حدّث بيانات الدخول الآن", "أدخل رمز التحقق لاستعادة الوصول"])
        action = action_ar if is_ar else action_en
        if is_ar:
            variants = {
                "problem_action_link": f"{greeting}\n\nتم اكتشاف مشكلة في {topic}. يجب أن {action} لتجنب فقدان الوصول إلى حساب المستشفى.\n\nاستخدم الرابط التالي وأدخل بيانات الدخول المطلوبة:\n{link}\n\nعدم الإكمال اليوم قد يؤدي إلى تعليق الوصول خلال ساعات.\n\nشكرًا،\nفريق دعم الحسابات",
                "warning_link_consequence": f"{greeting}\n\nتنبيه عاجل بخصوص {topic}. لن يستمر الوصول إلى الخدمة ما لم {action}.\n\nرابط التحقق: {link}\n\nسيتم إيقاف الحساب اليوم إذا لم يتم اتخاذ الإجراء.\n\nمع التحية،\nالدعم الفني",
                "notice_action_deadline": f"{greeting}\n\nهذا إشعار مهم من {area}. يرجى {action} قبل نهاية اليوم.\n\nابدأ من هنا: {link}\n\nالتأخير سيؤدي إلى فقدان الوصول.\n\nشكرًا،\nدعم بوابة الموظفين",
                "short_alert": f"{greeting}\n\nإجراء فوري مطلوب لـ {topic}: {action}.\n{link}\n\nستنتهي صلاحية الوصول اليوم.\n\nفريق الدعم",
                "support_request": f"{greeting}\n\nيطلب فريق الدعم تأكيد {topic}. يرجى {action} عبر الرابط التالي:\n{link}\n\nيجب إكمال الخطوة خلال ساعات.\n\nشكرًا لتعاونكم،\nمكتب الدعم",
                "account_status": f"{greeting}\n\nحالة الحساب: معلّقة بسبب {topic}. لإعادة التفعيل، {action}.\n\n{link}\n\nسيتم إغلاق الطلب اليوم.\n\nقسم الحسابات",
            }
        else:
            variants = {
                "problem_action_link": f"{greeting}\n\nA problem was detected with your {topic}. You must {action} to avoid losing access to your hospital account.\n\nPlease use this link and enter the requested login details:\n{link}\n\nFailure to complete this today may suspend your access within hours.\n\nThank you,\nHospital Account Support",
                "warning_link_consequence": f"{greeting}\n\nUrgent warning regarding {topic}. Your access will not continue unless you {action}.\n\nVerification link: {link}\n\nYour account will be disabled today if no action is taken.\n\nRegards,\nTechnical Support",
                "notice_action_deadline": f"{greeting}\n\nThis is an important notice from {area}. Please {action} before the end of today.\n\nStart here: {link}\n\nDelay will result in loss of access.\n\nThank you,\nStaff Portal Support",
                "short_alert": f"{greeting}\n\nImmediate action is required for {topic}: {action}.\n{link}\n\nAccess expires today.\n\nSupport Team",
                "support_request": f"{greeting}\n\nThe support team needs you to confirm {topic}. Please {action} using the following link:\n{link}\n\nComplete this step within hours.\n\nThank you for your cooperation,\nHelpdesk",
                "account_status": f"{greeting}\n\nAccount status: pending because of {topic}. To reactivate it, {action}.\n\n{link}\n\nThe request closes today.\n\nAccount Services",
            }
        body = variants.get(structure, next(iter(variants.values())))
        attack_type = "Credential Phishing"
        attachment = ""
    elif bp["difficulty"] == "medium":
        # FIXED: previously every "medium" structure (context_request_deadline,
        # department_notice, workflow_exception, review_and_confirm,
        # service_change, audit_followup, two_paragraph_professional) fell
        # through to one single fixed body, silently defeating point 4 of the
        # research design ("different writing structures") whenever the
        # deterministic fallback was used instead of the API. Each structure
        # now has its own distinct opening/framing/closing.
        deadline = bp["urgency_seed_ar"] if is_ar else bp["urgency_seed"]
        if is_ar:
            variants = {
                "context_request_deadline": f"{greeting}\n\nيرجى مراجعة {topic} الخاصة بقسم {area}. أظهر التحديث الأخير وجود عنصر يحتاج إلى تأكيد قبل اكتمال سير العمل.\n\nاستخدم بوابة المراجعة التالية وأكمل التحقق {deadline}:\n{link}\n\nلأي استفسار، تواصل مع القسم عبر القنوات الرسمية.\n\nمع التحية،\n{area}",
                "department_notice": f"{greeting}\n\nإشعار من قسم {area} بخصوص {topic}. نود إبلاغكم بتحديث يتطلب مراجعتكم ضمن سير العمل المعتاد.\n\nرابط الإشعار الكامل:\n{link}\n\nيرجى الاطلاع {deadline}.\n\nتحياتنا،\nقسم {area}",
                "workflow_exception": f"{greeting}\n\nتم رصد استثناء في سير عمل {topic} ضمن {area} يتطلب تأكيدكم قبل المتابعة.\n\nراجعوا تفاصيل الاستثناء من هنا:\n{link}\n\nإكمال المراجعة {deadline} يضمن استمرار سير العمل دون تأخير.\n\nمع الشكر،\n{area}",
                "review_and_confirm": f"{greeting}\n\nنطلب مراجعتكم وتأكيدكم على {topic} المرتبط بـ{area} قبل اعتماده بشكل نهائي.\n\nرابط المراجعة والتأكيد:\n{link}\n\nيرجى إتمام ذلك {deadline}.\n\nمع التقدير،\n{area}",
                "service_change": f"{greeting}\n\nنود إعلامكم بتغيير في خدمة {topic} يخص {area}، ويتطلب هذا التغيير إجراءً بسيطاً من جانبكم.\n\nتفاصيل التغيير والإجراء المطلوب:\n{link}\n\nالموعد النهائي للتنفيذ: {deadline}.\n\nمع التحية،\nفريق {area}",
                "audit_followup": f"{greeting}\n\nمتابعة لمراجعة سابقة تخص {topic} في {area}، لوحظ عنصر يحتاج تأكيدكم لإغلاق الملف.\n\nرابط ملف المتابعة:\n{link}\n\nنأمل إكمال ذلك {deadline}.\n\nمع الشكر،\nفريق التدقيق - {area}",
                "two_paragraph_professional": f"{greeting}\n\nبخصوص {topic} في {area}: راجع فريقنا آخر التحديثات ووجد عنصراً يستدعي تأكيدكم قبل استكمال الإجراء.\n\nيرجى استخدام الرابط التالي لإتمام التحقق {deadline}: {link}. لأي استفسار، تواصلوا مع القسم عبر القنوات الرسمية المعتمدة.\n\nمع التحية،\n{area}",
            }
        else:
            variants = {
                "context_request_deadline": f"{greeting}\n\nPlease review the {topic} for {area}. The latest update identified an item that requires confirmation before the workflow can be completed.\n\nUse the review portal below and complete the verification {deadline}:\n{link}\n\nFor questions, contact the department through the official directory.\n\nRegards,\n{area}",
                "department_notice": f"{greeting}\n\nThis is a notice from {area} regarding {topic}. There is an update that requires your review as part of the standard workflow.\n\nFull notice link:\n{link}\n\nPlease review this {deadline}.\n\nRegards,\n{area}",
                "workflow_exception": f"{greeting}\n\nA workflow exception was flagged for {topic} within {area} and needs your confirmation before it can proceed.\n\nReview the exception details here:\n{link}\n\nCompleting this {deadline} keeps the workflow on schedule.\n\nThank you,\n{area}",
                "review_and_confirm": f"{greeting}\n\nWe need your review and confirmation on the {topic} associated with {area} before it can be finalized.\n\nReview and confirmation link:\n{link}\n\nPlease complete this {deadline}.\n\nRegards,\n{area}",
                "service_change": f"{greeting}\n\nWe are notifying you of a change to the {topic} service affecting {area}, which requires a brief action on your part.\n\nChange details and required action:\n{link}\n\nDeadline for action: {deadline}.\n\nRegards,\n{area} Team",
                "audit_followup": f"{greeting}\n\nFollowing up on a recent review of {topic} in {area}, one item still needs your confirmation to close the file.\n\nFollow-up file link:\n{link}\n\nWe would appreciate this being completed {deadline}.\n\nThank you,\n{area} Audit Team",
                "two_paragraph_professional": f"{greeting}\n\nRegarding the {topic} for {area}: our team reviewed the latest update and found an item that requires your confirmation before the process can continue.\n\nPlease use the link below to complete verification {deadline}: {link}. For any questions, contact the department through the official directory.\n\nRegards,\n{area}",
            }
        body = variants.get(bp["structure"], next(iter(variants.values())))
        attack_type = "Spear Phishing"
        attachment = f"{re.sub(r'[^a-z0-9]+','_',topic.lower()).strip('_')}.pdf" if bp["attack"] == "pdf_review" else ""
    else:
        # FIXED: same structural-diversity gap as medium above, now resolved
        # for all 8 "hard" structures (internal_memo, reply_thread,
        # committee_request, document_share, calendar_followup,
        # audit_evidence, executive_request, system_notification).
        channel = bp["attack"]
        if is_ar:
            variants = {
                "internal_memo": f"{greeting}\n\nمذكرة داخلية بخصوص {topic} ضمن {area}. يرجى مراجعة المرفق/المرجع المرتبط وتأكيد الملاحظة قبل الاجتماع القادم.\n\n{link}\n\nفي حال وجود اختلاف، يرجى الرد على نفس السلسلة مع رقم الحالة فقط.\n\nمع التقدير،\n{area}",
                "reply_thread": f"{greeting}\n\nبالإشارة إلى مراسلاتنا السابقة حول {topic}، يرجى تأكيد النقطة المتبقية أدناه ضمن نفس السياق.\n\n{link}\n\nالرد على هذه الرسالة برقم الحالة فقط يكفي لإغلاق الملف.\n\nمع التقدير،\n{area}",
                "committee_request": f"{greeting}\n\nتطلب لجنة {area} مراجعتكم لملف {topic} قبل الجلسة القادمة.\n\nرابط ملف اللجنة:\n{link}\n\nيرجى تأكيد الاستلام قبل الموعد المحدد.\n\nمع التقدير،\nأمانة لجنة {area}",
                "document_share": f"{greeting}\n\nتمت مشاركة مستند يخص {topic} من {area} لمراجعتكم واعتماده.\n\nرابط المستند المشترك:\n{link}\n\nيرجى المراجعة في أقرب وقت ممكن.\n\nمع التحية،\n{area}",
                "calendar_followup": f"{greeting}\n\nمتابعة لاجتماع {topic} الأخير مع {area}، يرجى تأكيد البند المعلّق أدناه.\n\n{link}\n\nنأمل إغلاق هذا البند قبل الاجتماع القادم.\n\nمع التحية،\n{area}",
                "audit_evidence": f"{greeting}\n\nكجزء من مراجعة {topic} الجارية في {area}، نطلب منكم تزويدنا بالدليل المرتبط عبر الرابط التالي.\n\n{link}\n\nيرجى الرد بالرقم المرجعي للحالة فقط.\n\nمع التقدير،\nفريق المراجعة - {area}",
                "executive_request": f"{greeting}\n\nبتوجيه من إدارة {area}، يرجى مراجعة واعتماد ملف {topic} المرفق.\n\n{link}\n\nنأمل الإفادة في أقرب وقت ممكن نظراً لحساسية الموضوع.\n\nمع التقدير،\nمكتب {area}",
                "system_notification": f"{greeting}\n\nإشعار آلي من نظام {area} بخصوص {topic} يتطلب مراجعتكم.\n\n{link}\n\nهذا إشعار تلقائي، وللاستفسار يرجى التواصل مع {area} عبر القنوات الرسمية.\n\n{area}",
            }
        else:
            variants = {
                "internal_memo": f"{greeting}\n\nInternal memo regarding {topic} within {area}. Please review the linked reference and confirm the noted exception before the next meeting.\n\n{link}\n\nIf anything differs from your records, reply to this thread with the case reference only.\n\nKind regards,\n{area}",
                "reply_thread": f"{greeting}\n\nFollowing up on our earlier thread about {topic}, please confirm the remaining item below within the same context.\n\n{link}\n\nReplying with the case reference alone is enough to close this out.\n\nKind regards,\n{area}",
                "committee_request": f"{greeting}\n\nThe {area} committee requests your review of the {topic} file before the next session.\n\nCommittee file link:\n{link}\n\nPlease confirm receipt before the scheduled date.\n\nKind regards,\n{area} Committee Secretariat",
                "document_share": f"{greeting}\n\nA document regarding {topic} has been shared by {area} for your review and sign-off.\n\nShared document link:\n{link}\n\nPlease review at your earliest convenience.\n\nRegards,\n{area}",
                "calendar_followup": f"{greeting}\n\nFollowing up on the recent {topic} meeting with {area}, please confirm the pending item below.\n\n{link}\n\nWe would like this closed before the next meeting.\n\nRegards,\n{area}",
                "audit_evidence": f"{greeting}\n\nAs part of the ongoing {topic} review in {area}, we need you to provide the related evidence through the link below.\n\n{link}\n\nPlease reply with the case reference only.\n\nKind regards,\n{area} Audit Team",
                "executive_request": f"{greeting}\n\nAt the direction of {area} leadership, please review and approve the attached {topic} file.\n\n{link}\n\nGiven the sensitivity of this matter, a prompt response would be appreciated.\n\nKind regards,\n{area} Office",
                "system_notification": f"{greeting}\n\nAutomated notification from the {area} system regarding {topic} requires your review.\n\n{link}\n\nThis is an automated message; for questions, contact {area} through official channels.\n\n{area}",
            }
        body = variants.get(bp["structure"], next(iter(variants.values())))
        attack_type = "Advanced Spear Phishing"
        attachment = ""
        if channel == "pdf": attachment = f"{re.sub(r'[^a-z0-9]+','_',topic.lower()).strip('_')}_review.pdf"
        elif channel == "excel": attachment = f"{re.sub(r'[^a-z0-9]+','_',topic.lower()).strip('_')}_tracker.xlsx"
        elif channel == "qr": body += "\n\n[QR Code: Review the internal reference]" if not is_ar else "\n\n[رمز QR: مراجعة المرجع الداخلي]"

    suspicious_text = "" if not body else next((x for x in ["verify your password today", "confirm your staff PIN immediately", "within 24 hours", "please review the item"] if x.lower() in body.lower()), "review the request")
    indicators = _v18_indicator_objects(bp, link, True)
    cid = st.session_state.get('v18_cycle_id', 0)
    why = _v18_no_repeat_choice(
        (V19_WHY_RISKY_AR if is_ar else V19_WHY_RISKY_EN)[bp["difficulty"]],
        f"v19_recent_why_{bp['difficulty']}_{is_ar}_{cid}")
    tip = _v18_no_repeat_choice(
        (V19_LEARNING_TIP_AR if is_ar else V19_LEARNING_TIP_EN)[bp["difficulty"]],
        f"v19_recent_tip_{bp['difficulty']}_{is_ar}_{cid}")
    return {
        "email_type": topic, "attack_type": attack_type, "from": sender,
        "to": person.get("email", "employee@hospital.org"), "subject": subject,
        "attachment": attachment, "body": body, "suspicious_text": suspicious_text,
        "suspicious_link": link, "injected_errors": [], "indicators": indicators,
        "why_risky": why, "learning_tip": tip,
        "risk_level": "High" if bp["difficulty"] == "easy" else "Medium" if bp["difficulty"] == "medium" else "Subtle",
        "is_phishing": True, "display_time": bp["display_time"],
        "scenario_id": "v18:" + bp["signature"], "scenario_meta": bp,
    }


def _v18_legitimate(bp, role, index):
    rng = _v18_rng(index, bp["role_type"], bp["difficulty"], bp["phase"] + "_legit")
    person = _v18_recipient(bp["role_type"], index, bp["phase"])
    is_ar = bp["language"] == "Arabic"
    greeting = _v18_greeting(bp, person, rng)
    official_sender = f"{bp['area']} <no-reply@hospital.org>"
    if is_ar:
        body = f"{greeting}\n\nهذا إشعار رسمي بخصوص {bp['topic']}. يمكن مراجعة التفاصيل من خلال القنوات الداخلية المعتمدة في المستشفى فقط.\n\nلا يلزم إرسال كلمة مرور أو رمز تحقق عبر البريد الإلكتروني.\n\nمع التحية،\n{bp['area']}"
        why = "هذه رسالة شرعية تستخدم نطاق المستشفى الرسمي ولا تطلب بيانات حساسة أو رابطًا خارجيًا."
        tip = "افتح الأنظمة الداخلية من خلال البوابة الرسمية بدل روابط البريد."
    else:
        body = f"{greeting}\n\nThis is an official notice regarding {bp['topic']}. Details can be reviewed through the hospital's approved internal channels only.\n\nNo passwords, PINs, or verification codes are required by email.\n\nRegards,\n{bp['area']}"
        why = "This is a legitimate message from the official hospital domain and it does not request sensitive information or an external link."
        tip = "Open internal systems from the official hospital portal rather than email links."
    return {
        "email_type": bp["topic"], "attack_type": "Legitimate", "from": official_sender,
        "to": person.get("email", "employee@hospital.org"), "subject": f"Official update: {bp['topic']}",
        "attachment": "", "body": body, "suspicious_text": "", "suspicious_link": "",
        "injected_errors": [], "indicators": [], "why_risky": why, "learning_tip": tip,
        "risk_level": "Safe", "is_phishing": False, "display_time": bp["display_time"],
        "scenario_id": "v18:legit:" + bp["signature"], "scenario_meta": bp,
    }


def _v18_api_prompt(seed, bp):
    is_ar = bp["language"] == "Arabic"
    language_rule = "Write every natural-language field in Arabic." if is_ar else "Write every natural-language field in English."
    contract = V19_DIFFICULTY_CONTRACT[bp["difficulty"]]
    lo, hi = contract["indicator_range"]
    indicator_desc = f"exactly {lo}" if lo == hi else f"between {lo} and {hi}"
    subject_hint = bp["subject_template"].format(area=bp["area"], topic=bp["topic"])
    return f"""
You are generating one email for a PhD phishing-awareness experiment in a Saudi hospital.
{language_rule}
Return valid JSON only. Preserve the exact scenario, role, difficulty, recipient, sender domain, attachment, and display_time from the seed.
Do not copy the seed wording. Write the subject and body yourself, from scratch, using the creative direction below — do not just reword the seed's fallback sentences.

CRITICAL LINK RULE — read carefully:
- Do NOT write out any URL or link text yourself, anywhere in the body. Do NOT invent, shorten, paraphrase, or repeat a link.
- Instead, put the exact placeholder token {{{{LINK}}}} (literally these characters, once) at the single point in the body where a link would naturally appear.
- The system will replace {{{{LINK}}}} with the real controlled link after you respond. If the email is not phishing (legitimate), omit {{{{LINK}}}} entirely.
- A body containing any raw "http" text, or more than one {{{{LINK}}}} token, is invalid.

Role: {bp['role_type']}
Difficulty: {bp['difficulty']}
Department/context: {bp['area']} — {bp['topic']}
Attack channel: {bp['attack']}
Required structure style: {bp['structure']}

Creative direction (use these as inspiration, put them in your own words — do not quote them verbatim):
- Subject line idea: "{subject_hint}" — you may adjust the wording, but keep it about {bp['topic']}.
- Opening idea: the email should open around this situation — {bp['opening_seed']}.
- Urgency/deadline idea: frame the time pressure around — {bp['urgency_seed']}.

Difficulty contract for "{bp['difficulty']}" (follow every point):
- Link appearance: {contract['link_style']}.
- Greeting: {contract['greeting']}.
- Language register: {contract['language']}.
- How credentials are requested: {contract['credential_request']}.
- Urgency style: {contract['urgency_style']}.
- Number of red-flag indicators: {indicator_desc}.
- Attachment: {contract['attachment']}.
- QR code: {contract['qr']}.
- Microsoft 365 / Outlook branding: {contract['ms365_outlook']}.
- Fake login page realism: {contract['login_page_realism']}.

Seed JSON (for protected fields only — do not copy its wording):
{json.dumps(seed, ensure_ascii=False)}
"""


def _v18_parse_provider(data):
    # FIXED: call_ai() already normalizes every provider's successful
    # response (groq, openai, anthropic, gemini) into the same
    # {"choices": [{"message": {"content": "..."}}]} shape before it ever
    # reaches here. The old provider-specific branches below (reading
    # data["content"] blocks for anthropic, data["candidates"] for gemini)
    # never matched that normalized shape, so parsing silently failed for
    # those two providers and every call quietly fell back to the seed
    # template instead of the AI-rewritten text. We now read the normalized
    # shape first for all providers, keeping the raw provider shapes only
    # as a defensive fallback.
    try:
        if not isinstance(data, dict) or "error" in data:
            return None
        raw = None
        if "choices" in data:
            raw = data["choices"][0]["message"]["content"]
        elif "content" in data:
            raw = "".join(x.get("text", "") for x in data.get("content", []) if isinstance(x, dict))
        elif "candidates" in data:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
        if not raw:
            return None
        return parse_json_response(str(raw).strip())
    except Exception:
        return None


_V18_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)


def _v18_strip_and_place_link(body, link, is_phishing):
    """FIXED (root cause of the duplicate/conflicting-link bug): the model
    was asked to 'preserve the seed link' but LLMs routinely ignore that and
    write out their own plausible-looking URL inline instead. _v18_enforce
    then compared the seed link against the body with an exact substring
    check; since the AI's invented link never matched byte-for-byte, the
    real link got appended a second time — leaving TWO different fake links
    in the same email (one AI-invented and unstyled, one correct and
    highlighted). This function guarantees exactly one link, always the
    correct controlled one:
      1) Remove every raw http(s)/www URL the model wrote on its own.
      2) Fill in the {{LINK}} placeholder if the model used it.
      3) Otherwise insert the single controlled link once, at a position
         that varies (see _v18_place_link_fallback) instead of always
         landing in the exact same spot right before the signature.
    """
    text = str(body or "")
    text = _V18_URL_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces left by a removed URL
    text = re.sub(r"\n{3,}", "\n\n", text).strip()  # collapse blank-line gaps left behind

    if not is_phishing or not link:
        return text.replace("{{LINK}}", "").replace("{LINK}", "").strip()

    if "{{LINK}}" in text:
        return text.replace("{{LINK}}", link)
    if "{LINK}" in text:
        return text.replace("{LINK}", link)

    return _v18_place_link_fallback(text, link)


_V18_LINK_ACTION_WORDS = (
    "verify", "confirm", "click", "provide", "update", "reactivate", "sign in",
    "log in", "review", "complete", "restore", "reset",
    "تحقق", "أكّد", "أكد", "قدّم", "حدّث", "أعد تفعيل", "سجّل الدخول", "راجع", "أكمل",
)


def _v18_place_link_fallback(text, link):
    """FIXED: previously the link, whenever the model skipped the
    {{LINK}} placeholder, always landed in the exact same spot (right
    before the signature) in every single email — a monotony that stood
    out just as much as an outright duplicate. It now rotates, once per
    session, between three natural positions so consecutive examples don't
    all look identical."""
    lines = text.split("\n")
    sig_markers = ("regards", "sincerely", "thank you", "best,", "support",
                   "مع التحية", "تحياتي", "مع الشكر", "شكرًا", "وتفضلوا", "مع التقدير")
    sig_idx = next((i for i, ln in enumerate(lines)
                     if ln.strip() and any(m in ln.strip().lower() for m in sig_markers)), None)
    action_idx = next((i for i, ln in enumerate(lines)
                        if ln.strip() and any(w in ln.lower() for w in _V18_LINK_ACTION_WORDS)), None)

    strategies = ["before_signature", "own_line_end"]
    if action_idx is not None:
        strategies.append("after_action_sentence")

    key = f"v18_recent_link_position_{st.session_state.get('v18_cycle_id', 0)}"
    strategy = _v18_no_repeat_choice(strategies, key)

    if strategy == "after_action_sentence" and action_idx is not None:
        lines[action_idx:action_idx + 1] = [lines[action_idx].rstrip(), "", link]
        return "\n".join(lines).strip()
    if strategy == "before_signature" and sig_idx is not None:
        lines[sig_idx:sig_idx] = [link, ""]
        return "\n".join(lines).strip()
    return text.rstrip() + "\n\n" + link


def _v18_enforce(result, seed, bp):
    if not isinstance(result, dict):
        return seed
    protected = ["from", "to", "attachment", "suspicious_link", "display_time", "is_phishing", "scenario_id", "scenario_meta", "risk_level"]
    for k in protected:
        result[k] = seed.get(k)
    result.setdefault("email_type", seed.get("email_type"))
    result.setdefault("attack_type", seed.get("attack_type"))
    result.setdefault("subject", seed.get("subject"))
    result.setdefault("body", seed.get("body"))
    result.setdefault("why_risky", seed.get("why_risky"))
    result.setdefault("learning_tip", seed.get("learning_tip"))
    result["indicators"] = _v18_indicator_objects(bp, seed.get("suspicious_link", ""), bool(seed.get("is_phishing")))
    if bp["difficulty"] == "easy":
        result["attachment"] = ""
        result["body"] = re.sub(r"\[QR[^\]]*\]", "", str(result.get("body", "")), flags=re.I)
        lines = str(result.get("body", "")).splitlines()
        if lines:
            # FIXED (root cause of every email showing "Dear Staff,"): this
            # used to compare against only 3 fixed strings and, since an
            # AI paraphrase essentially never matches one of them verbatim,
            # it silently overwrote line 1 with allowed[0] on almost every
            # single call — collapsing all greeting diversity to one value.
            # We now draw from the same expanded, session-deduped pool used
            # by the deterministic path (_v18_greeting), so a forced
            # rewrite still lands on a fresh, varied greeting instead of
            # always the same one.
            pool = V18_EASY_GREETINGS_AR if bp["language"] == "Arabic" else V18_EASY_GREETINGS_EN
            if not any(lines[0].strip().lower() == x.lower() for x in pool):
                key = f"v18_recent_greeting_easy_{bp['language']}_{st.session_state.get('v18_cycle_id', 0)}"
                lines[0] = _v18_no_repeat_choice(pool, key)
            result["body"] = "\n".join(lines)
    if bp["difficulty"] == "medium":
        result["body"] = re.sub(r"\[QR[^\]]*\]", "", str(result.get("body", "")), flags=re.I)
        if str(result.get("attachment", "")).lower().endswith((".xlsx", ".xls", ".docx")):
            result["attachment"] = ""
    # FIXED: exactly one link, always the correct controlled one — see
    # _v18_strip_and_place_link docstring above for why this replaced the
    # old naive "append if missing" substring check.
    link = seed.get("suspicious_link", "")
    result["body"] = _v18_strip_and_place_link(result.get("body", ""), link, bool(seed.get("is_phishing")))
    try:
        result = clean_result(result, bp["language"] == "Arabic")
    except Exception:
        result = seed
    return result


def _v18_generate(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    phase = "assess" if assessment else "learn"
    bp = _v18_pick_blueprint(role, index, language, difficulty, phase, is_phishing)
    seed = _v18_fallback_phishing(bp, role, index) if is_phishing else _v18_legitimate(bp, role, index)
    result = seed
    try:
        data = call_ai(_v18_api_prompt(seed, bp), max_tokens=2400)
        parsed = _v18_parse_provider(data)
        if parsed:
            result = _v18_enforce(parsed, seed, bp)
    except Exception as exc:
        try:
            _store_debug("v18_api_fallback", str(exc))
        except Exception:
            pass
        result = seed
    try:
        evaluate_and_log_auto_scores(result, bp["difficulty"], language, is_phishing=bool(is_phishing))
    except Exception:
        pass
    return result


def generate_email(role, index, language, difficulty="medium"):
    return _v18_generate(role, index, language, difficulty, True, assessment=False)


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    return _v18_generate(role, index, language, difficulty, bool(is_phishing), assessment=True)


def generate_other_email(index, language, difficulty):
    role = "أخرى" if language == "Arabic" else "Other"
    return generate_email(role, index, language, difficulty)


def generate_other_assess_email(index, is_phishing, language, difficulty):
    role = "أخرى" if language == "Arabic" else "Other"
    return generate_assess_email(role, index, is_phishing, language, difficulty)


def go_to_learning(role):
    st.session_state["role"] = role
    st.session_state["page"] = "learning"
    st.session_state["example_index"] = 0
    st.session_state["emails"] = {}
    st.session_state["v18_cycle_id"] = random.randint(100000, 999999)
    for k in list(st.session_state.keys()):
        if str(k).startswith("v18_") and k != "v18_cycle_id":
            del st.session_state[k]

# =============================================================
# END FINAL RESEARCH ENGINE v18
# =============================================================

# =============================================================
# RESEARCH ENGINE v20 — semantic diversity + grounded analysis
# Added without changing the UI, providers, admin panel, exports,
# scoring, bilingual flow, or existing data structures.
# =============================================================

V20_SCENARIO_DIMENSIONS = {
    "clinical": {
        "units": ["Emergency Department", "Outpatient Clinic", "ICU", "Cardiology", "Oncology", "Pediatrics", "Radiology", "Laboratory", "Pharmacy", "Infection Control", "Operating Theatre", "Blood Bank", "Dialysis Unit", "Patient Safety", "Medical Affairs", "Clinical Education"],
        "events": [
            ("schedule_change", "a revised clinic or duty schedule requiring acknowledgement"),
            ("lab_followup", "a patient laboratory result requiring a documented follow-up"),
            ("referral_review", "a referral or transfer request awaiting clinical review"),
            ("medication_review", "a medication reconciliation or restricted-drug approval"),
            ("infection_notice", "an infection-control exposure or isolation update"),
            ("patient_safety", "a patient-safety incident or near-miss follow-up"),
            ("imaging_review", "an imaging report or PACS exception requiring review"),
            ("procedure_list", "a procedure list or theatre booking amendment"),
            ("handover_issue", "a handover discrepancy requiring confirmation"),
            ("equipment_alert", "a clinical-device allocation or maintenance notice"),
            ("training_update", "a mandatory clinical competency or simulation session"),
            ("policy_update", "a clinical protocol or guideline acknowledgement"),
            ("blood_product", "a blood-product release or transfusion documentation request"),
            ("audit_case", "a case audit, mortality review, or tracer evidence request"),
            ("telemedicine", "a remote consultation schedule or platform notification"),
            ("credentialing", "a clinical privilege or professional credential review"),
        ],
        "actions": ["review the case", "confirm receipt", "acknowledge the update", "validate the record", "approve the pending item", "open the referenced file", "respond with the case reference", "complete the requested review"],
    },
    "admin": {
        "units": ["Human Resources", "Finance", "Procurement", "Patient Access", "Medical Records", "Insurance Office", "Revenue Cycle", "Quality Office", "Facilities", "Executive Office", "Training and Development", "Legal Affairs", "Patient Relations", "Compliance", "Supply Chain", "Scheduling Office"],
        "events": [
            ("payroll_exception", "a payroll or allowance exception requiring confirmation"),
            ("leave_update", "a leave balance, roster, or attendance correction"),
            ("invoice_review", "a supplier invoice or payment exception"),
            ("vendor_change", "a vendor profile or banking-detail change"),
            ("insurance_claim", "an insurance claim or coverage discrepancy"),
            ("appointment_capacity", "a clinic-capacity or appointment scheduling update"),
            ("records_request", "a medical-records retention or release request"),
            ("audit_evidence", "an audit evidence or accreditation submission"),
            ("contract_review", "a contract amendment or renewal document"),
            ("benefits_enrollment", "an employee benefits or allowance enrollment notice"),
            ("training_registration", "a mandatory course or professional-development registration"),
            ("complaint_case", "a patient complaint or service-recovery case"),
            ("procurement_shortage", "a critical supply shortage or allocation approval"),
            ("board_document", "a confidential committee or board document"),
            ("privacy_case", "a privacy incident or data-protection response"),
            ("facility_workorder", "a facility access or work-order update"),
        ],
        "actions": ["review the request", "confirm the record", "approve the item", "validate the change", "acknowledge receipt", "open the supporting document", "submit the requested information", "respond with the reference number"],
    },
    "it": {
        "units": ["IT Service Desk", "Identity Management", "Network Operations", "Cybersecurity", "Cloud Operations", "Database Services", "Enterprise Applications", "PACS Support", "Digital Health", "Endpoint Management", "Data Center", "Integration Team", "Information Security", "Business Continuity", "Telephony", "Change Advisory Board"],
        "events": [
            ("identity_review", "an account-access or identity recertification request"),
            ("mfa_change", "an MFA token enrollment or re-registration notice"),
            ("vpn_migration", "a VPN profile migration or remote-access update"),
            ("security_alert", "a suspicious sign-in or endpoint security alert"),
            ("certificate_expiry", "a certificate renewal or trust-chain exception"),
            ("backup_failure", "a backup verification or recovery test exception"),
            ("database_change", "a database maintenance or credential-rotation request"),
            ("firewall_exception", "a firewall or conditional-access exception"),
            ("software_license", "a software license or application entitlement review"),
            ("device_compliance", "a managed-device compliance or quarantine notice"),
            ("interface_issue", "an HL7 or system-interface monitoring exception"),
            ("pacs_routing", "a PACS or DICOM routing exception"),
            ("incident_handoff", "an incident-response evidence or forensic handoff"),
            ("dr_test", "a disaster-recovery or failover approval"),
            ("change_request", "an emergency production change request"),
            ("cloud_federation", "a cloud tenant or federation configuration review"),
        ],
        "actions": ["review the alert", "confirm the change", "validate the exception", "approve the request", "acknowledge the maintenance window", "open the technical record", "respond with the ticket number", "complete the access review"],
    },
}

_V20_OLD_PICK_BLUEPRINT = _v18_pick_blueprint
_V20_OLD_API_PROMPT = _v18_api_prompt


def _v18_pick_blueprint(role, index, language, difficulty, phase, is_phishing=True):
    """Create a role-locked, semantically unique scenario blueprint.
    Uniqueness is based on the event family, not merely wording."""
    bp = _V20_OLD_PICK_BLUEPRINT(role, index, language, difficulty, phase, is_phishing)
    rt = bp["role_type"]
    dims = V20_SCENARIO_DIMENSIONS[rt]
    rng = _v18_rng(index, rt, bp["difficulty"], phase + "_v20")
    cid = st.session_state.get("v18_cycle_id", 0)

    # Learning and assessment share this memory, so test items do not reuse
    # the same semantic family already shown during learning.
    family_key = f"v20_used_families_{rt}_{cid}"
    used_families = set(st.session_state.get(family_key, []))
    available_events = [e for e in dims["events"] if e[0] not in used_families]
    if not available_events:
        available_events = list(dims["events"])
        used_families = set()
    family, event_text = rng.choice(available_events)
    used_families.add(family)
    st.session_state[family_key] = list(used_families)

    unit = _v18_no_repeat_choice(dims["units"], f"v20_units_{rt}_{cid}", rng)
    action = _v18_no_repeat_choice(dims["actions"], f"v20_actions_{rt}_{cid}", rng)
    bp["area"] = unit
    bp["scenario_family"] = family
    bp["event_text"] = event_text
    bp["action_text"] = action
    bp["topic"] = f"{event_text}; requested action: {action}"
    bp["signature"] = f"v20|{rt}|{bp['difficulty']}|{family}|{unit}|{action}|{bp['attack']}|{bp['structure']}"
    return bp


def _v20_find_phrase(body, patterns):
    for pattern in patterns:
        m = re.search(pattern, body, re.I)
        if m:
            return m.group(0).strip(" .,:;،؛")
    return ""


def _v20_grounded_analysis(result, bp):
    """Build tutor indicators from evidence that is visible in the final email.
    Each indicator also carries a render target so badge numbers and tutor text
    always point to the same element."""
    if not result.get("is_phishing"):
        result["indicators"] = []
        result["suspicious_text"] = ""
        return result

    is_ar = bp["language"] == "Arabic"
    body = str(result.get("body", ""))
    subject = str(result.get("subject", ""))
    sender = str(result.get("from", ""))
    link = str(result.get("suspicious_link", "")).strip()
    candidates = []

    def add(key, title_en, title_ar, desc_en, desc_ar, evidence, target):
        if evidence:
            candidates.append({
                "key": key,
                "title": title_ar if is_ar else title_en,
                "description": desc_ar if is_ar else desc_en,
                "evidence": evidence,
                "target": target,
            })

    mdom = re.search(r"@([A-Za-z0-9.-]+)", sender)
    sender_domain = mdom.group(1) if mdom else ""
    if sender_domain and sender_domain.lower() != "hospital.org":
        add("domain", "Non-official sender domain", "نطاق مرسل غير رسمي",
            f"The sender uses {sender_domain}, not the hospital's official domain.",
            f"عنوان المرسل يستخدم النطاق {sender_domain} وليس نطاق المستشفى الرسمي.",
            sender_domain, "from")

    urgency = _v20_find_phrase(subject + "\n" + body, [
        r"(?:immediate(?:ly)?|urgent(?:ly)?|within (?:the )?(?:hour|\d+ hours?)|today|before your next shift|as soon as possible|failure to act)",
        r"(?:فوراً|فورًا|عاجل|بشكل عاجل|خلال ساعة|خلال \d+ ساعات?|اليوم|قبل المناوبة القادمة|عدم الاستجابة)",
    ])
    if urgency:
        target = "subject" if urgency.lower() in subject.lower() else "body"
        add("urgency", "Time pressure", "ضغط زمني",
            "The message uses urgency or a deadline to push the recipient into acting quickly.",
            "تستخدم الرسالة مهلة أو استعجالاً لدفع المستلم للتصرف بسرعة.",
            urgency, target)

    if link and link in body:
        add("link", "Unapproved external link", "رابط خارجي غير معتمد",
            f"The link points to a non-official destination: {link}",
            f"الرابط يقود إلى نطاق غير رسمي: {link}",
            link, "link")

    first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if re.match(r"(?:Dear (?:Staff|Team|Employee|Colleague|Healthcare Team)|Hello Staff|عزيزي الموظف|فريق العمل العزيز|الزملاء الأعزاء|عزيزي الزميل)", first, re.I):
        add("greeting", "Generic greeting", "تحية عامة",
            "The email does not address the recipient by name, which can indicate bulk targeting.",
            "لا تخاطب الرسالة المستلم باسمه، ما قد يدل على إرسال جماعي.",
            first.rstrip("،,"), "greeting")

    credential = _v20_find_phrase(body, [
        r"(?:enter|confirm|verify|provide|reset|submit)\s+(?:your\s+)?(?:password|PIN|OTP|login credentials|credentials|account details)",
        r"(?:أدخل|أكد|تحقق من|زوّدنا بـ|أرسل)\s+(?:كلمة المرور|الرقم السري|رمز التحقق|بيانات الدخول)",
    ])
    if credential:
        add("credential", "Credential request", "طلب بيانات دخول",
            "The email asks for sensitive sign-in information that should not be provided by email.",
            "تطلب الرسالة بيانات حساسة لا ينبغي إرسالها عبر البريد.",
            credential, "body")

    attachment = str(result.get("attachment", "")).strip()
    if attachment:
        add("attachment", "Unexpected attachment", "مرفق غير متوقع",
            f"The message asks the recipient to open {attachment} as part of a sensitive request.",
            f"تطلب الرسالة فتح المرفق {attachment} ضمن طلب حساس.",
            attachment, "attachment")

    desired = {"easy": 4, "medium": 3, "hard": 2}[bp["difficulty"]]
    # Prefer distinct visible UI targets. This prevents two tutor numbers from
    # pointing to the same URL or a badge appearing with the wrong explanation.
    priority = {
        "easy": ["domain", "urgency", "link", "greeting", "credential", "attachment"],
        "medium": ["domain", "link", "credential", "urgency", "attachment", "greeting"],
        "hard": ["domain", "link", "attachment", "credential", "urgency", "greeting"],
    }[bp["difficulty"]]
    by_key = {c["key"]: c for c in candidates}
    chosen = [by_key[k] for k in priority if k in by_key][:desired]

    result["indicators"] = []
    for i, item in enumerate(chosen, 1):
        result["indicators"].append({
            "number": i,
            "title": item["title"],
            "description": item["description"],
            "evidence": item["evidence"],
            "target": item["target"],
            "key": item["key"],
        })

    # Legacy renderer compatibility; the updated renderer uses indicator targets.
    body_item = next((x for x in result["indicators"] if x.get("target") == "body"), None)
    result["suspicious_text"] = body_item.get("evidence", "") if body_item else ""
    return result

def _v20_validate(result, bp):
    if not isinstance(result, dict):
        return False
    body = str(result.get("body", "")).strip()
    subject = str(result.get("subject", "")).strip()
    if not body or not subject or len(body.split()) < (45 if bp["language"] == "English" else 35):
        return False
    if subject.lower() in body.lower():
        return False
    if result.get("is_phishing"):
        link = str(result.get("suspicious_link", ""))
        if link and body.count(link) != 1:
            return False
        lo, hi = {"easy": (4, 5), "medium": (3, 4), "hard": (1, 2)}[bp["difficulty"]]
        if not (lo <= len(result.get("indicators", [])) <= hi):
            return False
    # Guard against cross-role drift using a minimum of role-specific context.
    role_terms = {
        "clinical": ["patient", "clinical", "clinic", "ward", "lab", "pharmacy", "radiology", "مريض", "سريري", "عيادة", "قسم", "مختبر", "صيدلية"],
        "admin": ["payroll", "invoice", "supplier", "insurance", "records", "procurement", "employee", "رواتب", "فاتورة", "مورد", "تأمين", "سجلات", "موظف"],
        "it": ["account", "network", "server", "security", "system", "access", "VPN", "شبكة", "خادم", "أمن", "نظام", "وصول"],
    }
    combined = f"{subject} {body} {bp.get('area','')} {bp.get('event_text','')}".lower()
    if not any(term.lower() in combined for term in role_terms[bp["role_type"]]):
        return False
    return True


def _v18_api_prompt(seed, bp):
    base = _V20_OLD_API_PROMPT(seed, bp)
    length_rule = {"easy": "65-120", "medium": "85-150", "hard": "100-180"}[bp["difficulty"]]
    return base + f"""

V20 QUALITY AND DIVERSITY REQUIREMENTS:
- This email belongs ONLY to the {bp['role_type']} role. Do not introduce unrelated IT, administrative, or clinical scenarios.
- Semantic scenario family: {bp.get('scenario_family')}. Event: {bp.get('event_text')}.
- Requested action: {bp.get('action_text')}.
- Target body length: {length_rule} words (Arabic may be moderately shorter).
- Use 2-4 coherent paragraphs plus a natural closing and a role-appropriate signature.
- Do not repeat the subject verbatim in the body.
- Do not use the generic account-update/password-reset story unless the selected event explicitly requires it.
- Make sender name, subject, opening, call-to-action, closing, and signature fit this exact workplace event.
- Keep exactly one natural location for {{{{LINK}}}}. Never put it after the signature.
- Return indicators only if they are directly supported by visible evidence in the final email. The application will verify them.
"""


_V20_OLD_ENFORCE = _v18_enforce


def _v18_enforce(result, seed, bp):
    result = _V20_OLD_ENFORCE(result, seed, bp)
    return _v20_grounded_analysis(result, bp)


def _v18_generate(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    """Generate, validate, and retry while preserving the existing return schema."""
    phase = "assess" if assessment else "learn"
    best = None
    for attempt in range(3):
        bp = _v18_pick_blueprint(role, index + attempt * 97, language, difficulty, phase, is_phishing)
        seed = _v18_fallback_phishing(bp, role, index) if is_phishing else _v18_legitimate(bp, role, index)
        seed = _v20_grounded_analysis(seed, bp)
        candidate = seed
        try:
            data = call_ai(_v18_api_prompt(seed, bp), max_tokens=2600)
            parsed = _v18_parse_provider(data)
            if parsed:
                candidate = _v18_enforce(parsed, seed, bp)
        except Exception as exc:
            try:
                _store_debug("v20_api_fallback", str(exc))
            except Exception:
                pass
        candidate = _v20_grounded_analysis(candidate, bp)
        if _v20_validate(candidate, bp):
            best = candidate
            break
        best = seed
    result = best
    try:
        evaluate_and_log_auto_scores(result, str(difficulty or "medium").lower(), language, is_phishing=bool(is_phishing))
    except Exception:
        pass
    return result


# Public entry points remain unchanged for the rest of the application.
def generate_email(role, index, language, difficulty="medium"):
    return _v18_generate(role, index, language, difficulty, True, assessment=False)


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    return _v18_generate(role, index, language, difficulty, bool(is_phishing), assessment=True)

# =============================================================
# END RESEARCH ENGINE v20
# =============================================================


# =============================================================
# RULE-GUIDED PHISHING SCENARIO ENGINE v30
# Knowledge base -> planner -> difficulty contract -> composer -> validator
# This block intentionally overrides the legacy generate_email entry points.
# =============================================================
import hashlib as _v30_hashlib
import time as _v30_time

_V30_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_history_v30.json")
_V30_RNG = random.SystemRandom()

V30_DIFFICULTY = {
    "easy": {
        "indicator_count": 5, "generic_greeting": True, "personal_greeting": False,
        "direct_credentials": True, "urgency": "strong", "domain_style": "obvious",
        "allowed_channels": ("link",), "attachments": False, "qr": False,
        "body_words": (70, 125),
    },
    "medium": {
        "indicator_count": 3, "generic_greeting": False, "personal_greeting": False,
        "direct_credentials": False, "urgency": "moderate", "domain_style": "similar",
        "allowed_channels": ("button", "link", "pdf"), "attachments": "sometimes", "qr": False,
        "body_words": (85, 145),
    },
    "hard": {
        "indicator_count": 2, "generic_greeting": False, "personal_greeting": True,
        "direct_credentials": False, "urgency": "logical", "domain_style": "near",
        "allowed_channels": ("button", "pdf", "xlsx", "sharepoint", "m365", "qr"),
        "attachments": "allowed", "qr": "allowed", "body_words": (95, 165),
    },
}

# Each family is a coherent workplace workflow. Sender identities and actions are
# tied to the family so Pharmacy can never sign an Infection Control message, etc.
V30_KNOWLEDGE = {
    "clinical": [
        {"id":"lab_critical","area":"Laboratory Services","event":"a critical laboratory result requiring documented follow-up","objects":["potassium result","blood culture result","coagulation result","troponin result","specimen rejection notice"],"actions":["review the result","acknowledge the finding","open the case record"],"senders":["Laboratory Results Desk","Clinical Laboratory Services"],"signatures":["Laboratory Results Team","Clinical Laboratory Services"]},
        {"id":"referral","area":"Referral Coordination","event":"a specialist referral awaiting clinical review","objects":["cardiology referral","oncology referral","neurology referral","urgent outpatient referral","inter-facility transfer request"],"actions":["review the referral","confirm clinical acceptance","open the referral record"],"senders":["Referral Coordination Unit","Patient Flow Office"],"signatures":["Referral Coordination Team","Patient Flow Office"]},
        {"id":"radiology","area":"Radiology","event":"a diagnostic imaging report awaiting review","objects":["CT report","MRI report","ultrasound report","critical imaging addendum","contrast safety checklist"],"actions":["review the report","acknowledge the addendum","open the imaging record"],"senders":["Radiology Reporting Office","Diagnostic Imaging Services"],"signatures":["Radiology Reporting Team","Diagnostic Imaging Services"]},
        {"id":"medication","area":"Pharmacy","event":"a medication-safety item requiring clinical review","objects":["high-alert medication review","medication reconciliation exception","formulary substitution notice","antimicrobial approval","dose clarification request"],"actions":["review the medication item","acknowledge the safety notice","open the medication record"],"senders":["Medication Safety Office","Clinical Pharmacy Services"],"signatures":["Medication Safety Team","Clinical Pharmacy Services"]},
        {"id":"infection","area":"Infection Prevention","event":"an infection-control notification requiring acknowledgement","objects":["isolation precaution update","exposure follow-up","outbreak advisory","hand-hygiene audit finding","screening protocol update"],"actions":["review the notice","acknowledge the update","open the exposure record"],"senders":["Infection Prevention and Control","Occupational Health Infection Desk"],"signatures":["Infection Prevention Team","Occupational Health"]},
        {"id":"patient_safety","area":"Patient Safety","event":"a patient-safety case awaiting documented review","objects":["near-miss report","medication variance","fall-risk incident","handover concern","clinical escalation record"],"actions":["review the case","acknowledge the incident","open the safety record"],"senders":["Patient Safety Office","Clinical Governance Unit"],"signatures":["Patient Safety Team","Clinical Governance"]},
        {"id":"schedule","area":"Clinical Operations","event":"a clinical schedule change requiring confirmation","objects":["operating theatre list","on-call rota","clinic coverage schedule","weekend duty roster","procedure-room allocation"],"actions":["review the revised schedule","confirm availability","open the duty roster"],"senders":["Clinical Operations Office","Medical Workforce Scheduling"],"signatures":["Clinical Operations","Medical Workforce Team"]},
        {"id":"blood_bank","area":"Blood Bank","event":"a transfusion workflow item requiring clinical action","objects":["crossmatch request","blood-product release","transfusion reaction follow-up","massive transfusion checklist","blood availability notice"],"actions":["review the transfusion item","acknowledge the request","open the blood-bank record"],"senders":["Transfusion Services","Blood Bank Coordination"],"signatures":["Transfusion Services Team","Blood Bank"]},
        {"id":"discharge","area":"Health Information Management","event":"a discharge-documentation item awaiting completion","objects":["discharge summary","clinical coding query","unsigned progress note","incomplete medication list","pending follow-up plan"],"actions":["review the documentation","complete the acknowledgement","open the patient record"],"senders":["Clinical Documentation Office","Health Information Management"],"signatures":["Clinical Documentation Team","Health Information Management"]},
        {"id":"education","area":"Clinical Education","event":"a mandatory clinical learning item requiring completion","objects":["annual competency assessment","resuscitation update","medication-safety module","infection-control module","patient-identification refresher"],"actions":["open the learning item","confirm completion","review the assigned module"],"senders":["Clinical Education Department","Professional Development Unit"],"signatures":["Clinical Education Team","Professional Development"]},
        {"id":"equipment","area":"Biomedical Engineering","event":"a medical-device notice requiring acknowledgement","objects":["infusion pump advisory","patient monitor update","ventilator safety notice","defibrillator inspection","glucose meter recall"],"actions":["review the device notice","acknowledge the advisory","open the equipment record"],"senders":["Biomedical Engineering","Medical Device Safety Desk"],"signatures":["Biomedical Engineering Team","Medical Device Safety"]},
        {"id":"telehealth","area":"Virtual Care","event":"a remote-consultation item awaiting clinician review","objects":["telemedicine appointment","remote monitoring alert","virtual clinic handover","video consultation note","home-care escalation"],"actions":["review the virtual-care item","open the consultation record","confirm the handover"],"senders":["Virtual Care Coordination","Telehealth Operations"],"signatures":["Virtual Care Team","Telehealth Operations"]},
    ],
    "admin": [
        {"id":"insurance","area":"Insurance Coordination","event":"an insurance case awaiting administrative review","objects":["coverage exception","claim rejection","pre-authorization request","policy eligibility update","reimbursement query"],"actions":["review the case","open the claim record","confirm the administrative response"],"senders":["Insurance Coordination Unit","Revenue Cycle Office"],"signatures":["Insurance Coordination","Revenue Cycle Team"]},
        {"id":"billing","area":"Patient Billing","event":"a billing item requiring reconciliation","objects":["invoice discrepancy","unposted payment","billing adjustment","patient account exception","coding-related charge query"],"actions":["review the billing item","open the account record","acknowledge the discrepancy"],"senders":["Patient Billing Office","Revenue Integrity Unit"],"signatures":["Patient Billing Team","Revenue Integrity"]},
        {"id":"procurement","area":"Procurement","event":"a supplier transaction awaiting review","objects":["purchase order amendment","supplier invoice","contract renewal","delivery discrepancy","quotation approval"],"actions":["review the transaction","open the procurement record","confirm receipt"],"senders":["Medical Procurement Office","Supply Chain Management"],"signatures":["Procurement Team","Supply Chain Management"]},
        {"id":"appointments","area":"Patient Access","event":"an appointment workflow item requiring action","objects":["clinic overbooking notice","appointment rescheduling batch","waiting-list release","referral booking exception","patient registration correction"],"actions":["review the booking item","open the scheduling record","acknowledge the change"],"senders":["Patient Access Services","Appointment Coordination"],"signatures":["Patient Access Team","Appointment Coordination"]},
        {"id":"hr","area":"Human Resources","event":"an employee administration item requiring review","objects":["leave balance correction","benefits enrollment","contract detail update","attendance exception","staff credential renewal"],"actions":["review the employee item","open the HR record","confirm the update"],"senders":["Human Resources Services","Employee Relations Office"],"signatures":["Human Resources","Employee Relations"]},
        {"id":"payroll","area":"Payroll","event":"a payroll exception requiring confirmation","objects":["salary payment exception","IBAN verification case","allowance adjustment","overtime reconciliation","end-of-service calculation"],"actions":["review the payroll item","open the payment record","confirm the correction"],"senders":["Payroll Services","Compensation and Benefits"],"signatures":["Payroll Team","Compensation and Benefits"]},
        {"id":"records","area":"Medical Records Administration","event":"an administrative patient-record item awaiting review","objects":["release-of-information request","record merge exception","demographic correction","archiving notice","document indexing query"],"actions":["review the records item","open the administrative record","acknowledge the request"],"senders":["Medical Records Administration","Health Information Services"],"signatures":["Medical Records Team","Health Information Services"]},
        {"id":"facilities","area":"Facilities Management","event":"a facilities service item requiring coordination","objects":["access badge review","maintenance closure","office relocation","parking permit update","building access exception"],"actions":["review the service notice","open the facilities request","confirm the arrangement"],"senders":["Facilities Management","Workplace Services"],"signatures":["Facilities Team","Workplace Services"]},
    ],
    "it": [
        {"id":"vpn","area":"Network Services","event":"a remote-access service item requiring technical review","objects":["VPN certificate update","remote access exception","gateway maintenance","privileged access review","secure tunnel migration"],"actions":["review the service item","open the network ticket","acknowledge the change"],"senders":["Network Operations Centre","Infrastructure Services"],"signatures":["Network Operations","Infrastructure Services"]},
        {"id":"identity","area":"Identity and Access Management","event":"an identity-governance item awaiting review","objects":["privileged role recertification","inactive account review","MFA registration exception","service-account ownership","access entitlement review"],"actions":["review the access item","open the IAM case","acknowledge the entitlement"],"senders":["Identity and Access Management","Cybersecurity Operations"],"signatures":["IAM Team","Cybersecurity Operations"]},
        {"id":"server","area":"Infrastructure Operations","event":"a server operation item requiring action","objects":["storage threshold alert","backup verification","patch maintenance window","cluster failover review","virtual machine ownership"],"actions":["review the infrastructure item","open the operations ticket","acknowledge the maintenance"],"senders":["Infrastructure Operations","Data Centre Services"],"signatures":["Infrastructure Operations","Data Centre Services"]},
        {"id":"security","area":"Cybersecurity","event":"a security case awaiting technical review","objects":["endpoint detection alert","firewall rule exception","vulnerability remediation","phishing investigation","certificate anomaly"],"actions":["review the security case","open the incident ticket","acknowledge the finding"],"senders":["Security Operations Centre","Cybersecurity Governance"],"signatures":["Security Operations","Cybersecurity Governance"]},
        {"id":"database","area":"Database Services","event":"a database service item requiring confirmation","objects":["backup integrity check","replication lag alert","schema change request","database account review","restore-test result"],"actions":["review the database item","open the service ticket","acknowledge the change"],"senders":["Database Administration","Enterprise Applications"],"signatures":["Database Services","Enterprise Applications"]},
        {"id":"application","area":"Clinical Applications","event":"an enterprise application item awaiting review","objects":["EMR interface error","application release note","license allocation","integration exception","production change request"],"actions":["review the application item","open the change record","acknowledge the release"],"senders":["Clinical Applications Support","Enterprise Systems"],"signatures":["Clinical Applications","Enterprise Systems"]},
        {"id":"cloud","area":"Cloud and Backup Services","event":"a cloud service item requiring technical review","objects":["backup retention exception","cloud access review","storage policy update","recovery test","subscription renewal"],"actions":["review the cloud item","open the service record","acknowledge the policy"],"senders":["Cloud Platform Operations","Backup and Recovery Services"],"signatures":["Cloud Operations","Backup and Recovery"]},
        {"id":"helpdesk","area":"IT Service Desk","event":"a support case requiring technician action","objects":["escalated service ticket","remote support request","device enrollment case","software deployment issue","asset handover"],"actions":["review the support case","open the ticket","acknowledge the assignment"],"senders":["IT Service Desk","End User Computing"],"signatures":["IT Service Desk","End User Computing"]},
    ],
}


# Extra workflow families ensure that 6 learning + 10 assessment items can use
# different semantic families in one full cycle, not merely different wording.
V30_KNOWLEDGE["clinical"].extend([
    {"id":"emergency","area":"Emergency Medicine","event":"an emergency-care workflow item requiring review","objects":["triage escalation","observation-unit handover","trauma pathway exception","emergency discharge follow-up","critical-care transfer note"],"actions":["review the emergency record","acknowledge the handover","open the escalation note"],"senders":["Emergency Operations Desk","Acute Care Coordination"],"signatures":["Emergency Operations","Acute Care Coordination"]},
    {"id":"surgery","area":"Surgical Services","event":"a perioperative item awaiting clinical review","objects":["surgical consent exception","theatre list amendment","pre-operative checklist","post-operative handover","implant traceability record"],"actions":["review the surgical item","acknowledge the amendment","open the perioperative record"],"senders":["Surgical Services Office","Perioperative Coordination"],"signatures":["Surgical Services","Perioperative Coordination"]},
    {"id":"maternity","area":"Maternity Services","event":"a maternal-care item requiring follow-up","objects":["antenatal risk review","labour ward handover","newborn screening result","postnatal follow-up","fetal monitoring addendum"],"actions":["review the maternity item","acknowledge the handover","open the maternal record"],"senders":["Maternity Coordination","Women’s Health Services"],"signatures":["Maternity Services","Women’s Health Services"]},
    {"id":"dialysis","area":"Renal Services","event":"a renal-care workflow item requiring review","objects":["dialysis schedule change","vascular access note","renal medication review","dialysis adequacy result","transplant clinic follow-up"],"actions":["review the renal item","acknowledge the update","open the dialysis record"],"senders":["Renal Services Coordination","Dialysis Unit Office"],"signatures":["Renal Services","Dialysis Unit"]},
    {"id":"respiratory","area":"Respiratory Care","event":"a respiratory-care item awaiting acknowledgement","objects":["ventilator setting review","oxygen therapy order","pulmonary function report","respiratory isolation note","home oxygen assessment"],"actions":["review the respiratory item","acknowledge the order","open the care record"],"senders":["Respiratory Care Services","Pulmonary Coordination"],"signatures":["Respiratory Care","Pulmonary Services"]},
    {"id":"rehabilitation","area":"Rehabilitation Services","event":"a rehabilitation plan requiring clinical review","objects":["physiotherapy plan","occupational therapy assessment","mobility risk update","speech therapy note","discharge rehabilitation plan"],"actions":["review the rehabilitation plan","acknowledge the assessment","open the therapy record"],"senders":["Rehabilitation Coordination","Allied Health Services"],"signatures":["Rehabilitation Services","Allied Health Services"]},
    {"id":"vaccination","area":"Occupational Health","event":"a workforce-health item requiring follow-up","objects":["vaccination record","fitness-to-work review","exposure assessment","screening result","immunity status update"],"actions":["review the occupational-health item","acknowledge the notice","open the staff-health record"],"senders":["Occupational Health Services","Employee Health Clinic"],"signatures":["Occupational Health","Employee Health Clinic"]},
    {"id":"pathology","area":"Anatomic Pathology","event":"a pathology workflow item awaiting review","objects":["histopathology addendum","specimen discrepancy","cytology report","tumour board pathology note","biopsy tracking exception"],"actions":["review the pathology item","acknowledge the addendum","open the specimen record"],"senders":["Anatomic Pathology Services","Specimen Coordination Desk"],"signatures":["Anatomic Pathology","Specimen Coordination"]},
    {"id":"nutrition","area":"Clinical Nutrition","event":"a nutrition-care item requiring review","objects":["enteral feeding plan","nutrition risk assessment","diet order exception","parenteral nutrition review","allergy-related meal update"],"actions":["review the nutrition item","acknowledge the plan","open the nutrition record"],"senders":["Clinical Nutrition Services","Dietetic Coordination"],"signatures":["Clinical Nutrition","Dietetic Services"]},
    {"id":"oncology","area":"Oncology Services","event":"an oncology-care item requiring clinical follow-up","objects":["chemotherapy protocol review","tumour board decision","infusion schedule change","treatment consent note","oncology follow-up plan"],"actions":["review the oncology item","acknowledge the plan","open the treatment record"],"senders":["Oncology Care Coordination","Cancer Centre Operations"],"signatures":["Oncology Services","Cancer Centre Operations"]},
])

V30_KNOWLEDGE["admin"].extend([
    {"id":"quality_admin","area":"Quality Administration","event":"a quality-governance item requiring administrative follow-up","objects":["accreditation evidence request","policy publication record","committee action log","audit document index","quality dashboard exception"],"actions":["review the quality item","open the evidence record","acknowledge the request"],"senders":["Quality Management Office","Accreditation Coordination"],"signatures":["Quality Management","Accreditation Coordination"]},
    {"id":"legal","area":"Legal Affairs","event":"a legal-administration item awaiting review","objects":["contract clarification","records preservation notice","consent-form revision","legal correspondence log","regulatory response draft"],"actions":["review the legal item","open the case record","acknowledge the notice"],"senders":["Legal Affairs Office","Compliance Administration"],"signatures":["Legal Affairs","Compliance Administration"]},
    {"id":"training_admin","area":"Learning and Development","event":"an employee-learning administration item requiring action","objects":["course enrollment","attendance correction","certificate record","training nomination","orientation schedule"],"actions":["review the training item","open the learning record","confirm the nomination"],"senders":["Learning and Development","Workforce Development"],"signatures":["Learning and Development","Workforce Development"]},
    {"id":"executive","area":"Executive Office","event":"an executive-administration item requiring coordination","objects":["committee briefing pack","leadership meeting action","executive correspondence","board document review","departmental status request"],"actions":["review the executive item","open the briefing record","acknowledge the action"],"senders":["Executive Office","Corporate Affairs"],"signatures":["Executive Office","Corporate Affairs"]},
    {"id":"communications","area":"Corporate Communications","event":"a communications item awaiting administrative review","objects":["public notice draft","staff announcement","media enquiry log","campaign approval","website content update"],"actions":["review the communication","open the approval record","acknowledge the draft"],"senders":["Corporate Communications","Public Affairs Office"],"signatures":["Corporate Communications","Public Affairs"]},
    {"id":"transport","area":"Transport Services","event":"a logistics item requiring coordination","objects":["patient transport schedule","fleet booking exception","courier route change","ambulance transfer log","vehicle access permit"],"actions":["review the logistics item","open the transport record","confirm the arrangement"],"senders":["Transport Services","Logistics Coordination"],"signatures":["Transport Services","Logistics Coordination"]},
    {"id":"housing","area":"Staff Services","event":"a staff-services item requiring administrative review","objects":["housing allocation","staff shuttle request","uniform issue","employee ID replacement","staff accommodation maintenance"],"actions":["review the staff-service item","open the request record","confirm the arrangement"],"senders":["Staff Services Office","Employee Experience"],"signatures":["Staff Services","Employee Experience"]},
    {"id":"inventory","area":"Warehouse Operations","event":"an inventory item requiring reconciliation","objects":["stock discrepancy","expiry review","goods receipt exception","inventory transfer","consignment count"],"actions":["review the inventory item","open the stock record","acknowledge the discrepancy"],"senders":["Warehouse Operations","Materials Management"],"signatures":["Warehouse Operations","Materials Management"]},
])

V30_KNOWLEDGE["it"].extend([
    {"id":"integration","area":"Integration Services","event":"an interface workflow item requiring technical review","objects":["HL7 queue exception","API certificate update","interface mapping change","message-routing alert","integration endpoint review"],"actions":["review the integration item","open the interface ticket","acknowledge the change"],"senders":["Integration Services","Interoperability Team"],"signatures":["Integration Services","Interoperability Team"]},
    {"id":"endpoint","area":"Endpoint Management","event":"a managed-device item awaiting review","objects":["device compliance exception","encryption status alert","software deployment","laptop ownership review","mobile device enrollment"],"actions":["review the endpoint item","open the device record","acknowledge the deployment"],"senders":["Endpoint Management","Digital Workplace Operations"],"signatures":["Endpoint Management","Digital Workplace"]},
    {"id":"telecom","area":"Unified Communications","event":"a communications-platform item requiring technical review","objects":["call routing change","contact centre queue","Teams voice migration","extension ownership review","video conference gateway"],"actions":["review the communications item","open the service request","acknowledge the change"],"senders":["Unified Communications","Telecommunications Services"],"signatures":["Unified Communications","Telecommunications Services"]},
    {"id":"pacs","area":"Imaging Informatics","event":"an imaging-system item requiring technical review","objects":["PACS archive alert","modality worklist exception","DICOM routing change","image retention policy","viewer access review"],"actions":["review the imaging-system item","open the PACS ticket","acknowledge the change"],"senders":["Imaging Informatics","PACS Support"],"signatures":["Imaging Informatics","PACS Support"]},
    {"id":"analytics","area":"Data and Analytics","event":"a data-platform item awaiting review","objects":["dashboard refresh failure","data-quality exception","report ownership review","analytics workspace access","scheduled extract alert"],"actions":["review the data item","open the analytics ticket","acknowledge the exception"],"senders":["Data and Analytics","Business Intelligence Services"],"signatures":["Data and Analytics","Business Intelligence"]},
    {"id":"change","area":"IT Change Management","event":"a production-change item requiring technical approval","objects":["emergency change request","maintenance window","rollback plan","change collision alert","post-implementation review"],"actions":["review the change item","open the change record","acknowledge the schedule"],"senders":["IT Change Management","Technology Governance"],"signatures":["IT Change Management","Technology Governance"]},
    {"id":"vendor_it","area":"Technology Vendor Management","event":"a vendor-support item requiring technical coordination","objects":["support entitlement renewal","vendor remote session","license true-up","maintenance agreement","technical escalation"],"actions":["review the vendor item","open the support record","acknowledge the request"],"senders":["Technology Vendor Management","IT Commercial Services"],"signatures":["Technology Vendor Management","IT Commercial Services"]},
    {"id":"continuity","area":"IT Service Continuity","event":"a resilience item requiring technical review","objects":["disaster recovery exercise","business continuity dependency","recovery time validation","failover test","critical service inventory"],"actions":["review the continuity item","open the resilience record","acknowledge the test"],"senders":["IT Service Continuity","Technology Resilience Office"],"signatures":["IT Service Continuity","Technology Resilience"]},
])

V30_STRUCTURES = ["context-action-deadline", "notice-background-action", "case-summary-request", "follow-up-reference-action", "service-impact-action", "policy-context-response", "handover-context-review", "exception-summary-resolution"]
V30_GENERIC_GREETINGS = {"English":["Dear Staff Member","Hello Clinical Team","Dear Healthcare Employee","Attention Clinical Staff","Dear Colleague"], "Arabic":["عزيزي الموظف","فريق العمل السريري","عزيزي الممارس الصحي","إلى الزملاء السريريين","عزيزي الزميل"]}
V30_DEPARTMENT_GREETINGS = {"English":["Dear {area} Team","Hello {area} Colleague","To the {area} Team"], "Arabic":["فريق {area} العزيز","الزميل في {area}","إلى فريق {area}"]}
V30_FIRST_NAMES = ["Sarah Almutairi","Ahmed Alotaibi","Noura Alshamri","Fahad Aldosari","Mona Alharbi","Khalid Alanazi","Lama Alqahtani","Faisal Alzahrani"]

V30_OBVIOUS_DOMAINS = ["secure-staff-check.net","hospital-access-alert.co","portal-confirm-now.net","clinical-update-login.org","staff-verify-center.com"]
V30_SIMILAR_DOMAINS = ["hospital-portal.org","hospital-services.net","moh-clinical.org","hospital-workflow.com","staff-hospital.org"]
V30_NEAR_DOMAINS = ["hospitaI.org","hospital-sso.org","hospital365.org","hospital-sharepoint.org","hospital-sa.org"]


def _v30_load_history():
    try:
        with open(_V30_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _v30_save_history(rows):
    try:
        with open(_V30_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(rows[-12000:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _v30_role_type(role):
    return ROLE_MAP.get(role, ROLE_MAP["Clinical"])[2]


def _v30_session_init(phase, index):
    key = f"v30_{phase}_used"
    if index == 0 or key not in st.session_state:
        st.session_state[key] = {"fingerprints": [], "families": [], "structures": [], "senders": [], "objects": [], "channels": []}
    return st.session_state[key]


def _v30_fingerprint(plan):
    raw = "|".join(str(plan.get(k,"")) for k in ["role_type","difficulty","family_id","object","action","structure","channel","sender","indicator_pattern"])
    return _v30_hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _v30_plan(role, index, language, difficulty, phase, is_phishing):
    role_type = _v30_role_type(role)
    if role_type == "other":
        role_type = _V30_RNG.choice(["clinical","admin","it"])
    difficulty = str(difficulty or "medium").lower()
    if difficulty not in V30_DIFFICULTY: difficulty = "medium"
    used = _v30_session_init(phase, index)
    history_fps = {r.get("fingerprint") for r in _v30_load_history()[-8000:] if isinstance(r, dict)}
    learning_families = set(st.session_state.get("v30_learn_used", {}).get("families", [])) if phase == "assess" else set()
    families = V30_KNOWLEDGE[role_type]
    candidates = [f for f in families if f["id"] not in used["families"] and f["id"] not in learning_families]
    if not candidates:
        candidates = [f for f in families if f["id"] not in learning_families] or families

    for _ in range(250):
        fam = _V30_RNG.choice(candidates)
        obj = _V30_RNG.choice(fam["objects"])
        action = _V30_RNG.choice(fam["actions"])
        sender = _V30_RNG.choice(fam["senders"])
        signature = fam["signatures"][fam["senders"].index(sender) % len(fam["signatures"])]
        structure_opts = [x for x in V30_STRUCTURES if x not in used["structures"]] or V30_STRUCTURES
        structure = _V30_RNG.choice(structure_opts)
        channel_opts = list(V30_DIFFICULTY[difficulty]["allowed_channels"])
        channel = _V30_RNG.choice([x for x in channel_opts if x not in used["channels"]] or channel_opts)
        plan = {"role_type":role_type,"difficulty":difficulty,"language":language,"phase":phase,
                "family_id":fam["id"],"area":fam["area"],"event":fam["event"],"object":obj,
                "action":action,"sender":sender,"signature":signature,"structure":structure,
                "channel":channel,"is_phishing":bool(is_phishing)}
        plan["indicator_pattern"] = {"easy":"domain+urgency+credential+link+greeting","medium":"domain+workflow+channel","hard":"identity+channel"}[difficulty]
        fp = _v30_fingerprint(plan)
        if fp not in used["fingerprints"] and fp not in history_fps and obj not in used["objects"] and sender not in used["senders"]:
            plan["fingerprint"] = fp
            break
    else:
        plan["fingerprint"] = _v30_fingerprint(plan) + str(_V30_RNG.randrange(1000,9999))

    for k, val in [("fingerprints",plan["fingerprint"]),("families",plan["family_id"]),("structures",plan["structure"]),("senders",plan["sender"]),("objects",plan["object"]),("channels",plan["channel"])]:
        used[k].append(val)
    hist = _v30_load_history(); hist.append({"timestamp":_v30_time.time(),"fingerprint":plan["fingerprint"],"role":role_type,"difficulty":difficulty,"family":plan["family_id"],"object":obj,"phase":phase}); _v30_save_history(hist)
    return plan


def _v30_recipient(role, index, language, phase):
    try: return get_recipient(role, index, language, phase="assess" if phase=="assess" else "learn")
    except Exception: return "employee@hospital.org"


def _v30_display_name(email):
    local = (email or "employee").split("@")[0].replace("."," ").replace("_"," ")
    parts = [p for p in local.split() if len(p)>1 and p.lower() not in {"dr","n","m","t","ph","s"}]
    return " ".join(p.capitalize() for p in parts[-2:]) or _V30_RNG.choice(V30_FIRST_NAMES)


def _v30_domain(difficulty):
    return _V30_RNG.choice({"easy":V30_OBVIOUS_DOMAINS,"medium":V30_SIMILAR_DOMAINS,"hard":V30_NEAR_DOMAINS}[difficulty])


def _v30_subject(plan, urgency_phrase):
    obj = plan["object"].title()
    if plan.get("language") == "Arabic":
        easy = [f"عاجل: إجراء مطلوب اليوم بشأن {obj}", f"مراجعة فورية مطلوبة: {obj}", f"إشعار أخير: متابعة {obj}", f"إجراء مطلوب اليوم — {obj}"]
        medium = [f"متابعة مطلوبة: {obj}", f"إشعار سير عمل — {obj}", f"فترة مراجعة {obj}", f"بانتظار التأكيد: {obj}"]
        hard = [f"{obj} — تحديث المرجع", f"متابعة بشأن {obj}", f"ملاحظة مراجعة: {obj}", f"بخصوص {obj}"]
    else:
        easy = [f"Urgent: {obj} Requires Action Today", f"Immediate Review Required: {obj}", f"Final Notice: {obj} Access Pending", f"Action Needed Today — {obj}"]
        medium = [f"Follow-up Required: {obj}", f"Workflow Notice — {obj}", f"Review Window for {obj}", f"Pending Confirmation: {obj}"]
        hard = [f"{obj} — Reference Update", f"Follow-up on {obj}", f"Review note: {obj}", f"Regarding {obj}"]
    return _V30_RNG.choice({"easy":easy,"medium":medium,"hard":hard}[plan["difficulty"]])


def _v30_link(plan, domain):
    short = re.sub(r"[^a-z0-9]+","-",plan["family_id"].lower()).strip("-")
    token = _V30_RNG.randrange(1000,9999)
    return f"http://{domain}/{short}/{token}"


def _v30_indicator(number, key, title, description, evidence, target):
    return {"number":number,"key":key,"title":title,"description":description,"evidence":evidence,"target":target}


def _v30_compose_phishing(plan, role, index):
    lang = plan["language"]; ar = lang == "Arabic"; diff = plan["difficulty"]
    recipient = _v30_recipient(role,index,lang,plan["phase"]); person = _v30_display_name(recipient)
    domain = _v30_domain(diff); link = _v30_link(plan,domain)
    mailbox = _V30_RNG.choice(["alerts","workflow","notifications","case-review","service-desk"])
    sender = f'{plan["sender"]} <{mailbox}@{domain}>'
    urgency = _V30_RNG.choice(["today","immediately","within the next hour","before your next shift"])
    subject = _v30_subject(plan, urgency)

    if diff == "easy":
        greeting = _V30_RNG.choice(V30_GENERIC_GREETINGS[lang])
        credential = _V30_RNG.choice(["enter your password","confirm your staff PIN","submit your login credentials","verify your account details"])
        if ar:
            cred_map={"enter your password":"أدخل كلمة المرور","confirm your staff PIN":"أكد الرقم السري للموظف","submit your login credentials":"أرسل بيانات الدخول","verify your account details":"تحقق من بيانات حسابك"}; credential=cred_map[credential]
            body = f"{greeting}،\n\nورد إلى {plan['area']} إشعار بخصوص {plan['object']}، ويتطلب الإجراء متابعة سريعة حتى لا يتوقف سير العمل. يرجى {credential} {('فورًا' if urgency=='immediately' else 'اليوم')} لإتاحة {plan['action']} وإغلاق التنبيه.\n\nاستخدم الرابط التالي لإكمال الإجراء: {link}\n\nعدم الاستجابة {('خلال الساعة القادمة' if 'hour' in urgency else 'قبل المناوبة القادمة')} قد يؤدي إلى تعليق الوصول مؤقتًا.\n\nشكرًا لتعاونك،\n{plan['signature']}"
            indicators=[
                _v30_indicator(1,"domain","نطاق مرسل غير رسمي",f"عنوان المرسل يستخدم النطاق {domain} وليس نطاق المستشفى الرسمي.",domain,"from"),
                _v30_indicator(2,"urgency","استعجال أو تهديد واضح","يستخدم العنوان والنص مهلة قصيرة لدفع المستلم إلى التصرف دون تحقق.",next(x for x in ["فورًا","اليوم","خلال الساعة القادمة","قبل المناوبة القادمة"] if x in subject+body),"subject" if any(x in subject for x in ["عاجل","فوري","اليوم"]) else "body"),
                _v30_indicator(3,"credential","طلب مباشر لبيانات الدخول","تطلب الرسالة كلمة مرور أو رقمًا سريًا أو بيانات حساب مباشرة.",credential,"body"),
                _v30_indicator(4,"link","رابط خارجي واضح",f"الرابط يقود إلى النطاق غير الرسمي {domain}.",link,"link"),
                _v30_indicator(5,"greeting","تحية عامة","لا تخاطب الرسالة المستلم باسمه، ما قد يدل على إرسال جماعي.",greeting,"greeting"),
            ]
            why="تجمع الرسالة بين نطاق غير رسمي وطلب مباشر لبيانات الدخول ورابط خارجي مع ضغط زمني واضح، وهي مؤشرات أساسية على التصيد."
            tip="لا تُدخل كلمة المرور أو الرقم السري عبر رابط وارد في بريد مفاجئ. افتح النظام الرسمي مباشرة أو تواصل مع القسم عبر قناة موثوقة."
        else:
            body = f"{greeting},\n\n{plan['area']} has received a notification concerning {plan['object']}. The item requires prompt follow-up to prevent disruption to the related clinical workflow. Please {credential} {urgency} so you can {plan['action']} and close the notification.\n\nUse the following link to complete the action: {link}\n\nFailure to respond {('within the next hour' if 'hour' in urgency else 'before your next shift')} may result in temporary access suspension.\n\nThank you for your cooperation,\n{plan['signature']}"
            urgency_ev = next(x for x in ["Urgent","Immediate","Today","today","immediately","within the next hour","before your next shift"] if x in subject+body)
            indicators=[
                _v30_indicator(1,"domain","Non-official sender domain",f"The sender uses {domain}, not the hospital's official domain.",domain,"from"),
                _v30_indicator(2,"urgency","Strong urgency or threat","The message uses an immediate deadline or access consequence to pressure the recipient.",urgency_ev,"subject" if urgency_ev in subject else "body"),
                _v30_indicator(3,"credential","Direct credential request","The email directly asks for a password, staff PIN, or login credentials.",credential,"body"),
                _v30_indicator(4,"link","Visible external link",f"The link points to the non-official domain {domain}.",link,"link"),
                _v30_indicator(5,"greeting","Generic greeting","The message does not address the recipient by name, which may indicate bulk targeting.",greeting,"greeting"),
            ]
            why="The email combines a non-official domain, a direct credential request, an external link, and strong time pressure—clear beginner-level phishing indicators."
            tip="Never enter a password or staff PIN through an unexpected email link. Open the official system directly or verify the request through a trusted channel."
        attachment=""
    elif diff == "medium":
        greeting = _V30_RNG.choice(V30_DEPARTMENT_GREETINGS[lang]).format(area=plan["area"])
        deadline = _V30_RNG.choice(["within 24 hours","by tomorrow afternoon","before the current review window closes"])
        if ar:
            dmap={"within 24 hours":"خلال 24 ساعة","by tomorrow afternoon":"قبل ظهر الغد","before the current review window closes":"قبل إغلاق فترة المراجعة الحالية"}; deadline=dmap[deadline]
            cta=f"سجّل الدخول إلى بوابة المتابعة لمراجعة {plan['object']}"; label="فتح سجل المتابعة"
            body=f"{greeting}،\n\nتجري {plan['area']} مراجعة دورية للطلبات المفتوحة، وقد ظهر {plan['object']} ضمن العناصر التي تحتاج إلى متابعة. يرجى {plan['action']} {deadline} حتى يبقى السجل ضمن مسار العمل الحالي.\n\n{cta}: [{label}]({link})\n\nإذا لم تكن مسؤولًا عن هذا العنصر، يرجى إعادة توجيه الإشعار إلى المنسق المعتمد.\n\nمع التحية،\n{plan['signature']}"
            indicators=[
                _v30_indicator(1,"domain","نطاق مشابه لكنه غير رسمي",f"النطاق {domain} يبدو قريبًا من جهة رسمية لكنه ليس hospital.org.",domain,"from"),
                _v30_indicator(2,"workflow","طلب تسجيل دخول خارج مسار العمل المعتاد","تطلب الرسالة فتح بوابة خارجية بدل استخدام النظام الداخلي المعروف.",cta,"body"),
                _v30_indicator(3,"link","رابط بوابة غير معتمدة",f"زر المتابعة يقود إلى {domain}، وهو نطاق غير رسمي.",link,"link"),
            ]; why="تبدو الرسالة مهنية ومرتبطة بالعمل، لكن نطاق البوابة غير رسمي وطريقة تسجيل الدخول لا تتبع المسار المعتاد."; tip="افتح النظام من الاختصار الرسمي أو المفضلة، ولا تعتمد على زر تسجيل دخول داخل بريد غير متوقع."
        else:
            cta=f"Sign in to the review portal to {plan['action']}"; label="Open review record"
            body=f"{greeting},\n\n{plan['area']} is completing a routine review of open workflow items. {plan['object'].title()} has been listed for follow-up. Please {plan['action']} {deadline} so the record remains in the current workflow queue.\n\n{cta}: [{label}]({link})\n\nIf this item is not assigned to you, forward the notice to the approved coordinator rather than entering case details by email.\n\nRegards,\n{plan['signature']}"
            indicators=[
                _v30_indicator(1,"domain","Look-alike sender domain",f"The domain {domain} resembles a workplace service but is not hospital.org.",domain,"from"),
                _v30_indicator(2,"workflow","Unusual external sign-in workflow","The email directs the recipient to sign in through an emailed portal instead of the known internal system.",cta,"body"),
                _v30_indicator(3,"link","Unapproved portal link",f"The review button resolves to the non-official domain {domain}.",link,"link"),
            ]; why="The message is professionally written and job-relevant, but the look-alike domain and external sign-in workflow are inconsistent with approved hospital practice."; tip="Navigate to the official system independently. Do not use an emailed sign-in button unless the destination has been verified."
        attachment=""
    else:
        greeting = (f"Dear {person}" if not ar else f"عزيزي {person}")
        # hard uses subtle channel/identity issue; no direct credentials or artificial deadline
        channel=plan["channel"]
        attachment=""
        if channel in ("pdf","xlsx"):
            ext = ".pdf" if channel=="pdf" else ".xlsx"; attachment=re.sub(r"[^a-z0-9]+","_",plan["family_id"])+f"_{_V30_RNG.randrange(10,99)}{ext}"
        if ar:
            body=f"{greeting}،\n\nأرسل لك متابعة بخصوص {plan['object']} التي نوقشت ضمن {plan['area']}. أضفت مرجع الحالة حتى تتمكن من {plan['action']} عند توفر الوقت، ثم تدوين الملاحظة في السجل المعتاد.\n\n"
            if attachment: body += f"ستجد التفاصيل في المرفق {attachment}. يرجى مطابقته مع رقم الحالة في النظام قبل اعتماده.\n\n"
            else: body += f"مرجع المتابعة متاح هنا: [عرض مرجع الحالة]({link})\n\n"
            body += f"لا توجد حاجة لإرسال أي معلومات حساسة عبر البريد.\n\nمع التقدير،\n{plan['signature']}"
            indicators=[_v30_indicator(1,"domain","اختلاف دقيق في هوية النطاق",f"النطاق {domain} قريب بصريًا من النطاق الرسمي لكنه مختلف.",domain,"from")]
            if attachment: indicators.append(_v30_indicator(2,"attachment","مرفق غير متوقع","يجب التحقق من المرفق عبر النظام الداخلي قبل فتحه، حتى لو بدا السياق واقعيًا.",attachment,"attachment"))
            else: indicators.append(_v30_indicator(2,"link","رابط خارجي خفي",f"نص الرابط يبدو مهنيًا لكن وجهته هي {domain}.",link,"link"))
            why="السياق واللغة طبيعيان جدًا، لكن هناك اختلافًا دقيقًا في هوية النطاق وقناة خارجية غير متوقعة. هذه مؤشرات متقدمة تتطلب التحقق من التفاصيل."; tip="في الرسائل الواقعية، ركّز على النطاق الفعلي ومسار العمل، وليس على جودة اللغة أو معرفة المرسل بالسياق."
        else:
            body=f"{greeting},\n\nI am following up on {plan['object']}, which was discussed through {plan['area']}. I included the case reference so you can {plan['action']} when convenient and record the outcome in the usual system.\n\n"
            if attachment: body += f"The supporting detail is in {attachment}. Please match it against the case number in the internal system before relying on it.\n\n"
            else: body += f"The reference is available here: [View case reference]({link})\n\n"
            body += f"No sensitive information needs to be sent by email.\n\nKind regards,\n{plan['signature']}"
            indicators=[_v30_indicator(1,"domain","Subtle domain discrepancy",f"The domain {domain} is visually close to the official domain but is not identical.",domain,"from")]
            if attachment: indicators.append(_v30_indicator(2,"attachment","Unexpected contextual attachment","The attachment should be verified in the internal system before it is opened, despite the realistic context.",attachment,"attachment"))
            else: indicators.append(_v30_indicator(2,"link","Hidden external destination",f"The professional-looking link text resolves to {domain}.",link,"link"))
            why="The message is natural and context-aware, but a subtle domain discrepancy and an unexpected external channel indicate a sophisticated phishing attempt."; tip="For realistic emails, inspect the exact domain and verify the workflow independently; polished language is not proof of legitimacy."

    return {"from":sender,"to":recipient,"subject":subject,"body":body,"attachment":attachment,
            "suspicious_link":link if not attachment else (link if diff!="hard" else ""),"suspicious_text":next((i["evidence"] for i in indicators if i["target"]=="body"),""),
            "indicators":indicators,"why_risky":why,"learning_tip":tip,"is_phishing":True,
            "email_type":"Phishing","attack_type":{"easy":"Credential harvesting","medium":"Look-alike portal","hard":"Contextual spear phishing"}[diff],
            "risk_level":diff,"scenario_id":plan["fingerprint"],"scenario_meta":plan,"display_time":_V30_RNG.choice(["Today, 8:15 AM","Today, 10:42 AM","Monday, 2:31 PM","Yesterday, 4:05 PM","Thursday, 9:18 AM"])}


def _v30_compose_legitimate(plan, role, index):
    lang=plan["language"]; ar=lang=="Arabic"; recipient=_v30_recipient(role,index,lang,plan["phase"]); person=_v30_display_name(recipient)
    sender=f'{plan["sender"]} <notifications@hospital.org>'; subject=f'{plan["object"].title()} — Information Notice'
    if ar:
        greeting=f"عزيزي {person}"; body=f"{greeting}،\n\nنود إحاطتك بأن {plan['area']} حدّثت سجل {plan['object']}. يمكنك {plan['action']} من خلال النظام الداخلي المعتاد عند توفر الوقت.\n\nلا تتطلب هذه الرسالة إرسال كلمة مرور أو رمز تحقق، ولا تحتوي على رابط تسجيل دخول خارجي. للاستفسار، استخدم دليل الهاتف الداخلي أو افتح تذكرة عبر البوابة الرسمية.\n\nمع التحية،\n{plan['signature']}"; tip="الرسالة تستخدم النطاق الرسمي وتوجّهك إلى القناة الداخلية المعروفة دون طلب بيانات حساسة."
    else:
        greeting=f"Dear {person}"; body=f"{greeting},\n\n{plan['area']} has updated the record for {plan['object']}. You may {plan['action']} through the usual internal system when convenient.\n\nThis message does not request a password or verification code and contains no external sign-in link. For questions, use the internal directory or open a ticket through the official portal.\n\nKind regards,\n{plan['signature']}"; tip="The message uses the official domain and directs you to a known internal channel without requesting sensitive information."
    return {"from":sender,"to":recipient,"subject":subject,"body":body,"attachment":"","suspicious_link":"","suspicious_text":"","indicators":[],"why_risky":"","learning_tip":tip,"is_phishing":False,"email_type":"Legitimate","attack_type":"None","risk_level":plan["difficulty"],"scenario_id":plan["fingerprint"],"scenario_meta":plan,"display_time":_V30_RNG.choice(["Today, 9:05 AM","Tuesday, 11:20 AM","Yesterday, 3:40 PM"])}


def _v30_validate(result, plan):
    if not isinstance(result,dict): return False
    body=str(result.get("body", "")); subject=str(result.get("subject", "")); sender=str(result.get("from", ""))
    if not all([body.strip(),subject.strip(),sender.strip()]): return False
    if subject.lower() in body.lower(): return False
    # exact difficulty indicator contract
    if result.get("is_phishing") and len(result.get("indicators",[])) != V30_DIFFICULTY[plan["difficulty"]]["indicator_count"]: return False
    # all evidence must visibly exist in the correct field
    fields={"from":sender,"subject":subject,"body":body,"greeting":body,"link":body,"attachment":str(result.get("attachment", ""))}
    for ind in result.get("indicators",[]):
        if str(ind.get("evidence", "")).lower() not in fields.get(ind.get("target"),"").lower(): return False
    # difficulty hard prohibitions
    if plan["difficulty"]=="hard" and re.search(r"password|PIN|OTP|immediately|account suspension",body,re.I): return False
    if plan["difficulty"]=="easy" and result.get("attachment"): return False
    # sender/signature coherence
    if plan["signature"].lower() not in body.lower(): return False
    # role-family lock
    if plan["family_id"] not in {f["id"] for f in V30_KNOWLEDGE[plan["role_type"]]}: return False
    return True


def _v30_generate(role,index,language,difficulty="medium",is_phishing=True,assessment=False):
    phase="assess" if assessment else "learn"
    plan=_v30_plan(role,index,language,difficulty,phase,is_phishing)
    result=_v30_compose_phishing(plan,role,index) if is_phishing else _v30_compose_legitimate(plan,role,index)
    if not _v30_validate(result,plan):
        # A second independently planned deterministic generation is safer than
        # displaying a rule-breaking model response.
        plan=_v30_plan(role,index+1009,language,difficulty,phase,is_phishing)
        result=_v30_compose_phishing(plan,role,index) if is_phishing else _v30_compose_legitimate(plan,role,index)
    try: evaluate_and_log_auto_scores(result,str(difficulty or "medium").lower(),language,is_phishing=bool(is_phishing))
    except Exception: pass
    return result


def generate_email(role,index,language,difficulty="medium"):
    return _v30_generate(role,index,language,difficulty,True,False)


def generate_assess_email(role,index,is_phishing,language,difficulty="medium"):
    return _v30_generate(role,index,language,difficulty,bool(is_phishing),True)

# =============================================================
# END RULE-GUIDED ENGINE v30
# =============================================================


# =============================================================
# RULE-GUIDED PHISHING SCENARIO ENGINE v31
# Diversity writer: varied rhetorical structures, natural legitimate
# messages, and strict evidence-linked tutor analysis.
# =============================================================

V31_PHISHING_STYLES = [
    "brief_alert", "case_followup", "manager_escalation", "service_notice",
    "deadline_reminder", "workflow_exception", "handover_request", "record_release"
]
V31_LEGIT_STYLES = [
    "informational_update", "meeting_followup", "record_correction", "schedule_note",
    "quality_review", "colleague_request", "policy_notice", "case_handover"
]
V31_BANNED_PHRASES = [
    "has received a notification concerning",
    "prevent disruption to the related clinical workflow",
    "close the notification",
    "use the following link to complete the action",
    "this message does not request a password",
    "contains no external sign-in link",
]


def _v31_pick(items, salt=0):
    # SystemRandom keeps sessions fresh; salt avoids repeated adjacent choices.
    if not items:
        return None
    return items[(_V30_RNG.randrange(len(items)) + int(salt or 0)) % len(items)]


def _v31_subject(plan, urgency, style):
    obj = plan["object"].title()
    area = plan["area"]
    easy = {
        "brief_alert": [f"Action Required Today: {obj}", f"Urgent Review: {obj}"],
        "case_followup": [f"Final Reminder — {obj}", f"{obj}: Response Needed Today"],
        "manager_escalation": [f"Escalated: {obj} Awaiting Your Response", f"Immediate Attention Requested — {obj}"],
        "service_notice": [f"Access Notice for {obj}", f"Service Alert: {obj}"],
        "deadline_reminder": [f"Deadline Today: {obj}", f"{obj} Will Expire {urgency.title()}"],
        "workflow_exception": [f"Workflow Exception: {obj}", f"Unresolved {obj} Requires Action"],
        "handover_request": [f"Shift Handover Item: {obj}", f"Before Your Next Shift — {obj}"],
        "record_release": [f"Release Pending: {obj}", f"Confirm Receipt of {obj}"],
    }
    medium = [f"{obj} Review Window", f"Follow-up Required: {obj}", f"{area} Update — {obj}"]
    hard = [f"{obj} Follow-up", f"Re: {obj}", f"{area} — {obj}"]
    if plan["difficulty"] == "easy":
        return _v31_pick(easy.get(style, easy["brief_alert"]))
    return _v31_pick(medium if plan["difficulty"] == "medium" else hard)


def _v31_easy_phishing(plan, role, index, style):
    lang = plan["language"]; ar = lang == "Arabic"
    recipient = _v30_recipient(role, index, lang, plan["phase"])
    domain = _v30_domain("easy")
    link = _v30_link(plan, domain)
    mailbox = _v31_pick(["alerts", "workflow", "review", "updates", "coordination"], index)
    sender = f'{plan["sender"]} <{mailbox}@{domain}>'
    urgency = _v31_pick(["today", "immediately", "within the next hour", "before your next shift"], index)
    subject = _v31_subject(plan, urgency, style)
    greeting = _v31_pick(V30_GENERIC_GREETINGS[lang], index)
    credential = _v31_pick([
        "enter your password", "confirm your staff PIN", "submit your login credentials",
        "verify your account details", "provide your verification code"
    ], index)
    obj = plan["object"]; action = plan["action"]; area = plan["area"]

    if ar:
        templates = {
            "brief_alert": f"{greeting}،\n\nتعذر إكمال متابعة {obj} في {area}. يلزم {credential} {urgency} لفتح السجل ومراجعة التفاصيل.\n\nرابط المتابعة: {link}\n\nسيؤدي عدم الإجراء في الوقت المحدد إلى تعليق الوصول مؤقتًا.\n\nمع التحية،\n{plan['signature']}",
            "case_followup": f"{greeting}،\n\nهذه متابعة أخيرة بشأن {obj}. ما زالت الخطوة المطلوبة غير مكتملة، ويجب {credential} {urgency} حتى تتمكن من {action}.\n\nافتح الحالة من هنا: {link}\n\nإذا لم يتم الرد، ستُعاد الحالة تلقائيًا وقد يتوقف الوصول.\n\n{plan['signature']}",
            "manager_escalation": f"{greeting}،\n\nتم تصعيد {obj} إلى {plan['sender']} لعدم وجود رد. أكمل التحقق عبر {credential} {urgency}.\n\nمتابعة الطلب: {link}\n\nالتأخير قد يؤدي إلى تعليق حسابك عن هذا المسار.\n\nشكرًا،\n{plan['signature']}",
            "service_notice": f"{greeting}،\n\nتم إيقاف الوصول إلى {obj} مؤقتًا أثناء تحديث الخدمة. لاستعادة الوصول، يرجى {credential} {urgency}.\n\nاستعادة الخدمة: {link}\n\nلن تتمكن من فتح السجل حتى تكتمل الخطوة.\n\n{plan['signature']}",
            "deadline_reminder": f"{greeting}،\n\nتنتهي مهلة {obj} {urgency}. استخدم {credential} لإبقاء الطلب فعالًا ثم {action}.\n\nإكمال الطلب: {link}\n\nسيتم إلغاء الوصول بعد انتهاء المهلة.\n\nمع التقدير،\n{plan['signature']}",
            "workflow_exception": f"{greeting}،\n\nرصد النظام استثناءً في {obj}. لتصحيح الحالة، يجب {credential} {urgency} ثم إعادة فتح العنصر.\n\nبوابة التصحيح: {link}\n\nقد يؤدي تجاهل الاستثناء إلى تعليق صلاحياتك.\n\n{plan['signature']}",
            "handover_request": f"{greeting}،\n\nيوجد عنصر تسليم متعلق بـ {obj} قبل مناوبتك القادمة. يرجى {credential} لعرضه وتأكيد الاستلام.\n\nعرض عنصر التسليم: {link}\n\nعدم التأكيد قد يمنع الوصول أثناء المناوبة.\n\n{plan['signature']}",
            "record_release": f"{greeting}،\n\nأصبح {obj} جاهزًا للإصدار، لكن يلزم تأكيد هويتك أولًا. يرجى {credential} {urgency}.\n\nفتح السجل: {link}\n\nسيتم سحب السجل إذا لم يُستكمل الإجراء.\n\n{plan['signature']}",
        }
        body = templates[style]
        urgency_ev = next((x for x in ["فورًا", "اليوم", "خلال الساعة القادمة", "قبل مناوبتك القادمة"] if x in body or x in subject), urgency)
        indicators = [
            _v30_indicator(1,"domain","نطاق مرسل غير رسمي",f"عنوان المرسل يستخدم النطاق {domain} وليس نطاق المستشفى الرسمي.",domain,"from"),
            _v30_indicator(2,"urgency","استعجال أو تهديد واضح","تستخدم الرسالة مهلة قصيرة أو عاقبة على الوصول لدفع المستلم إلى التصرف بسرعة.",urgency_ev,"subject" if urgency_ev in subject else "body"),
            _v30_indicator(3,"credential","طلب مباشر لبيانات الدخول","تطلب الرسالة كلمة مرور أو رقمًا سريًا أو رمز تحقق مباشرة.",credential,"body"),
            _v30_indicator(4,"link","رابط خارجي ظاهر",f"الرابط يقود إلى النطاق غير الرسمي {domain}.",link,"link"),
            _v30_indicator(5,"greeting","تحية عامة","لا تستخدم الرسالة اسم المستلم، ما قد يشير إلى استهداف جماعي.",greeting,"greeting"),
        ]
        why = "تجمع الرسالة بين نطاق غير رسمي وطلب مباشر لبيانات الدخول ورابط خارجي وضغط زمني؛ وهي مؤشرات واضحة على التصيد للمستوى المبتدئ."
        tip = "لا تدخل كلمة مرور أو رمز موظف من رابط وصل عبر البريد. افتح النظام الرسمي مباشرة وتحقق من الطلب عبر قناة موثوقة."
    else:
        templates = {
            "brief_alert": f"{greeting},\n\nA review item for {obj} could not be completed in {area}. Please {credential} {urgency} to open the record and review the details.\n\nOpen the review item: {link}\n\nIf the step is not completed, access may be suspended temporarily.\n\nRegards,\n{plan['signature']}",
            "case_followup": f"{greeting},\n\nThis is a final follow-up regarding {obj}. The requested step is still outstanding. Please {credential} {urgency} so you can {action}.\n\nView the case: {link}\n\nAn unanswered request will be returned to the queue and may restrict access.\n\n{plan['signature']}",
            "manager_escalation": f"{greeting},\n\n{obj.title()} has been escalated to {plan['sender']} because no response was recorded. Complete verification by {credential} {urgency}.\n\nReview the escalation: {link}\n\nDelays may suspend your access to this workflow.\n\nThank you,\n{plan['signature']}",
            "service_notice": f"{greeting},\n\nAccess to {obj} was paused during a service update. To restore it, please {credential} {urgency}.\n\nRestore access: {link}\n\nThe record will remain unavailable until the step is completed.\n\n{plan['signature']}",
            "deadline_reminder": f"{greeting},\n\nThe response window for {obj} ends {urgency}. Use the portal to {credential}, then {action}.\n\nComplete the request: {link}\n\nAccess will be withdrawn after the deadline.\n\nRegards,\n{plan['signature']}",
            "workflow_exception": f"{greeting},\n\nThe system flagged an exception against {obj}. To clear it, you must {credential} {urgency} and reopen the item.\n\nResolve the exception: {link}\n\nIgnoring the exception may suspend your permissions.\n\n{plan['signature']}",
            "handover_request": f"{greeting},\n\nA handover item linked to {obj} is waiting before your next shift. Please {credential} to view it and acknowledge receipt.\n\nOpen the handover item: {link}\n\nFailure to acknowledge it may block access during the shift.\n\n{plan['signature']}",
            "record_release": f"{greeting},\n\n{obj.title()} is ready for release, but identity confirmation is required first. Please {credential} {urgency}.\n\nOpen the record: {link}\n\nThe item will be withdrawn if the action is not completed.\n\n{plan['signature']}",
        }
        body = templates[style]
        urgency_ev = next((x for x in ["today", "immediately", "within the next hour", "before your next shift"] if x in body.lower() or x in subject.lower()), urgency)
        indicators = [
            _v30_indicator(1,"domain","Non-official sender domain",f"The sender uses {domain}, not the hospital's official domain.",domain,"from"),
            _v30_indicator(2,"urgency","Strong urgency or threat","The message uses a short deadline or an access consequence to pressure the recipient.",urgency_ev,"subject" if urgency_ev.lower() in subject.lower() else "body"),
            _v30_indicator(3,"credential","Direct credential request","The message directly asks for a password, staff PIN, login credentials, or verification code.",credential,"body"),
            _v30_indicator(4,"link","Visible external link",f"The link points to the non-official domain {domain}.",link,"link"),
            _v30_indicator(5,"greeting","Generic greeting","The message does not address the recipient by name, which may indicate bulk targeting.",greeting,"greeting"),
        ]
        why = "The message combines a non-official domain, a direct credential request, an external link, and strong time pressure—clear beginner-level phishing indicators."
        tip = "Never enter a password, staff PIN, or verification code through an unexpected email link. Open the official system directly or verify the request through a trusted channel."

    return {"from":sender,"to":recipient,"subject":subject,"body":body,"attachment":"",
            "suspicious_link":link,"suspicious_text":credential,"indicators":indicators,
            "why_risky":why,"learning_tip":tip,"is_phishing":True,"email_type":"Phishing",
            "attack_type":"Credential harvesting","risk_level":"easy","scenario_id":plan["fingerprint"],
            "scenario_meta":dict(plan, writer_style=style),"display_time":_v31_pick(["Today, 8:15 AM","Today, 10:42 AM","Monday, 2:31 PM","Yesterday, 4:05 PM","Thursday, 9:18 AM"], index)}


def _v31_compose_phishing(plan, role, index):
    # Easy is fully rewritten with eight distinct rhetorical structures.
    if plan["difficulty"] == "easy":
        used_key = f"v31_{plan['phase']}_phish_styles"
        used = st.session_state.setdefault(used_key, [])
        options = [s for s in V31_PHISHING_STYLES if s not in used] or list(V31_PHISHING_STYLES)
        style = _v31_pick(options, index)
        used.append(style)
        return _v31_easy_phishing(plan, role, index, style)
    # Medium/Hard keep the v30 difficulty mechanics, but their plan structure
    # already rotates and they are not the source of the current easy-template issue.
    result = _v30_compose_phishing(plan, role, index)
    result.setdefault("scenario_meta", {})["writer_style"] = plan.get("structure")
    return result


def _v31_legit_subject(plan, style):
    obj = plan["object"].title(); area = plan["area"]
    bank = {
        "informational_update": [f"{obj} — Information Update", f"Update: {obj}"],
        "meeting_followup": [f"Follow-up Before the {area} Meeting", f"{obj} for Thursday's Review"],
        "record_correction": [f"Correction Needed: {obj}", f"Please Check the {obj} Record"],
        "schedule_note": [f"Schedule Note — {obj}", f"Timing Update for {obj}"],
        "quality_review": [f"Quality Review: {obj}", f"{obj} Audit Follow-up"],
        "colleague_request": [f"Could You Review {obj}?", f"Quick Check on {obj}"],
        "policy_notice": [f"{area} Guidance Update", f"Policy Note: {obj}"],
        "case_handover": [f"Handover Note: {obj}", f"{obj} for the Incoming Team"],
    }
    return _v31_pick(bank[style])


def _v31_compose_legitimate(plan, role, index):
    lang = plan["language"]; ar = lang == "Arabic"
    recipient = _v30_recipient(role,index,lang,plan["phase"]); person = _v30_display_name(recipient)
    used_key = f"v31_{plan['phase']}_legit_styles"
    used = st.session_state.setdefault(used_key, [])
    options = [s for s in V31_LEGIT_STYLES if s not in used] or list(V31_LEGIT_STYLES)
    style = _v31_pick(options, index); used.append(style)
    sender_mailbox = _v31_pick(["notifications","coordination","quality","clinical.ops","records"], index)
    sender = f'{plan["sender"]} <{sender_mailbox}@hospital.org>'
    subject = _v31_legit_subject(plan, style) if not ar else f"تحديث: {plan['object']}"
    obj=plan["object"]; action=plan["action"]; area=plan["area"]; signature=plan["signature"]

    if ar:
        templates = {
            "informational_update": f"عزيزي {person}،\n\nتم تحديث {obj} في {area}. يمكنك {action} من خلال النظام الداخلي عند توفر الوقت.\n\nمع التحية،\n{signature}",
            "meeting_followup": f"عزيزي {person}،\n\nنجهز لمراجعة {obj} في اجتماع {area} القادم. يرجى إضافة ملاحظتك في مساحة العمل الداخلية قبل نهاية يوم الأربعاء.\n\nشكرًا،\n{signature}",
            "record_correction": f"عزيزي {person}،\n\nلاحظنا اختلافًا بسيطًا في سجل {obj}. هل يمكنك مراجعة الحقل الأخير وتصحيحه من خلال النظام المعتاد؟\n\nمع التقدير،\n{signature}",
            "schedule_note": f"عزيزي {person}،\n\nتم تعديل توقيت {obj}. سيظهر الموعد الجديد في الجدول الداخلي خلال هذا اليوم. يرجى الرد فقط إذا تعارض مع مناوبتك.\n\n{signature}",
            "quality_review": f"عزيزي {person}،\n\nأصبحت خلاصة مراجعة الجودة الخاصة بـ {obj} متاحة في مساحة {area}. توجد ملاحظتان تحتاجان تعليق القسم قبل الاجتماع القادم.\n\nمع التحية،\n{signature}",
            "colleague_request": f"مرحبًا {person}،\n\nهل تستطيع مراجعة {obj} عندما تسمح المناوبة؟ أضفت ملاحظة في السجل الداخلي توضح النقطة المطلوبة.\n\nشكرًا،\n{signature}",
            "policy_notice": f"عزيزي {person}،\n\nنُشرت النسخة المحدثة من إرشادات {obj} في مكتبة السياسات الداخلية. سيبدأ تطبيقها الأسبوع القادم، وسنناقش التغيير في اجتماع الفريق.\n\n{signature}",
            "case_handover": f"عزيزي {person}،\n\nأضفت ملخص تسليم يتعلق بـ {obj} للفريق القادم. يرجى مراجعته في السجل السريري وإضافة أي نقطة ناقصة قبل نهاية المناوبة.\n\nمع التحية،\n{signature}",
        }
        body=templates[style]; tip="الرسالة تستخدم نطاق المستشفى الرسمي وسياقًا مهنيًا طبيعيًا وتوجّه إلى قناة داخلية معروفة."
    else:
        templates = {
            "informational_update": f"Dear {person},\n\nThe {obj} record has been updated in {area}. You can {action} it through the usual internal workspace when convenient.\n\nKind regards,\n{signature}",
            "meeting_followup": f"Dear {person},\n\nWe are preparing the {obj} item for the next {area} meeting. Please add your comment in the internal workspace by Wednesday afternoon.\n\nThank you,\n{signature}",
            "record_correction": f"Dear {person},\n\nWe noticed a minor discrepancy in the {obj} record. Could you check the final field and correct it in the usual system?\n\nBest regards,\n{signature}",
            "schedule_note": f"Dear {person},\n\nThe timing for {obj} has changed. The revised slot will appear in the internal schedule later today. Please reply only if it conflicts with your shift.\n\nRegards,\n{signature}",
            "quality_review": f"Dear {person},\n\nThe quality-review summary for {obj} is now available in the {area} workspace. Two observations need the department's comments before the next meeting.\n\nKind regards,\n{signature}",
            "colleague_request": f"Hello {person},\n\nCould you review {obj} when the shift allows? I added a note in the internal record showing the point that needs confirmation.\n\nThanks,\n{signature}",
            "policy_notice": f"Dear {person},\n\nThe revised guidance for {obj} has been published in the internal policy library. It takes effect next week, and the change will be covered at the team briefing.\n\nRegards,\n{signature}",
            "case_handover": f"Dear {person},\n\nI added a handover summary for {obj} for the incoming team. Please review it in the clinical record and add anything missing before the end of the shift.\n\nKind regards,\n{signature}",
        }
        body=templates[style]; tip="The message uses the official hospital domain, a natural workplace context, and a known internal workflow without asking for sensitive information."

    return {"from":sender,"to":recipient,"subject":subject,"body":body,"attachment":"",
            "suspicious_link":"","suspicious_text":"","indicators":[],"why_risky":"","learning_tip":tip,
            "is_phishing":False,"email_type":"Legitimate","attack_type":"None","risk_level":plan["difficulty"],
            "scenario_id":plan["fingerprint"],"scenario_meta":dict(plan, writer_style=style),
            "display_time":_v31_pick(["Today, 9:05 AM","Tuesday, 11:20 AM","Yesterday, 3:40 PM","Thursday, 1:15 PM"],index)}


def _v31_validate(result, plan):
    if not isinstance(result, dict): return False
    body=str(result.get("body", "")); subject=str(result.get("subject", "")); sender=str(result.get("from", ""))
    if not all([body.strip(), subject.strip(), sender.strip()]): return False
    if subject.lower() in body.lower(): return False
    if any(p in body.lower() for p in V31_BANNED_PHRASES): return False
    if plan["role_type"] not in ("clinical","admin","it"): return False
    if plan["family_id"] not in {f["id"] for f in V30_KNOWLEDGE[plan["role_type"]]}: return False
    if result.get("is_phishing"):
        count=len(result.get("indicators",[]))
        ranges={"easy":(4,5),"medium":(3,4),"hard":(1,2)}
        lo,hi=ranges[plan["difficulty"]]
        if not (lo <= count <= hi): return False
        fields={"from":sender,"subject":subject,"body":body,"greeting":body,"link":body,"attachment":str(result.get("attachment", ""))}
        for ind in result.get("indicators",[]):
            ev=str(ind.get("evidence", "")); target=ind.get("target")
            if not ev or ev.lower() not in fields.get(target, "").lower(): return False
    else:
        if not sender.lower().endswith("@hospital.org>"): return False
        if re.search(r"password|staff pin|login credentials|verification code",body,re.I): return False
        if result.get("suspicious_link") or result.get("indicators"): return False
    if plan["signature"].lower() not in body.lower(): return False
    return True


def _v31_generate(role,index,language,difficulty="medium",is_phishing=True,assessment=False):
    phase="assess" if assessment else "learn"
    for attempt in range(4):
        plan=_v30_plan(role,index + attempt*1009,language,difficulty,phase,is_phishing)
        result=_v31_compose_phishing(plan,role,index) if is_phishing else _v31_compose_legitimate(plan,role,index)
        if _v31_validate(result,plan):
            try: evaluate_and_log_auto_scores(result,str(difficulty or "medium").lower(),language,is_phishing=bool(is_phishing))
            except Exception: pass
            return result
    # Last-resort compatible output, but normally unreachable.
    return result


def generate_email(role,index,language,difficulty="medium"):
    return _v31_generate(role,index,language,difficulty,True,False)


def generate_assess_email(role,index,is_phishing,language,difficulty="medium"):
    return _v31_generate(role,index,language,difficulty,bool(is_phishing),True)

# =============================================================
# END RULE-GUIDED ENGINE v31
# =============================================================


# =============================================================
# RULE-GUIDED SCENARIO PLANNER v32
# -------------------------------------------------------------
# Safe upgrade over v31:
#   * changes ONLY the pre-writing scenario plan and the email writer
#   * keeps the stable indicator schema, renderer fields, scoring, UI,
#     provider selection, and results pipeline unchanged
#   * plans coherent role-locked combinations before composing text
#   * stores semantic history so later sessions avoid recent combinations
# =============================================================

_V32_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_history_v32.json")
_V32_RNG = random.SystemRandom()

V32_PHISH_STRUCTURES = [
    "pending_release", "failed_review", "workflow_handover", "service_interruption",
    "escalation_notice", "case_return", "deadline_reminder", "record_exception",
    "coordination_request", "unresolved_item", "status_change", "final_followup",
]

V32_LEGIT_STRUCTURES = [
    "brief_update", "colleague_note", "meeting_preparation", "record_correction",
    "schedule_change", "quality_followup", "policy_notice", "handover_summary",
    "request_for_comment", "completion_notice", "clarification_request", "team_briefing",
]

V32_GENERIC_GREETINGS_EN = [
    "Dear Staff Member", "Dear Healthcare Employee", "Dear Colleague",
    "Hello Clinical Team", "Attention Clinical Staff", "Dear Clinical Team",
]
V32_GENERIC_GREETINGS_AR = [
    "عزيزي الموظف", "عزيزي الزميل", "فريق العمل السريري", "الزملاء الأعزاء",
    "إلى أعضاء الفريق السريري", "عزيزي عضو الفريق",
]

V32_CREDENTIAL_REQUESTS_EN = [
    "confirm your staff PIN", "provide your verification code",
    "verify your account details", "submit your login credentials",
    "enter your password", "approve the sign-in verification",
]
V32_CREDENTIAL_REQUESTS_AR = [
    "تأكيد الرقم السري للموظف", "إدخال رمز التحقق", "التحقق من بيانات الحساب",
    "إرسال بيانات الدخول", "إدخال كلمة المرور", "اعتماد طلب تسجيل الدخول",
]

V32_DEADLINES_EN = [
    "today", "immediately", "within the next hour", "before your next shift",
    "before the record is returned", "before 3:00 PM",
]
V32_DEADLINES_AR = [
    "اليوم", "فورًا", "خلال الساعة القادمة", "قبل مناوبتك القادمة",
    "قبل إعادة السجل", "قبل الساعة الثالثة مساءً",
]

V32_MAILBOXES = ["review", "workflow", "alerts", "coordination", "updates", "case-notice"]
V32_FAKE_DOMAINS = [
    "portal-confirm-now.net", "staff-verify-center.com", "secure-staff-check.net",
    "hospital-access-alert.co", "clinical-update-login.org", "care-review-portal.net",
]

V32_FAMILY_CONTEXT = {
    "clinical": {
        "lab_critical": ["the result was flagged during final validation", "the specimen record was returned for correction", "the result release is being held"],
        "referral": ["the referral is waiting in the acceptance queue", "the destination service has not recorded a response", "the transfer request is pending release"],
        "radiology": ["the reporting queue contains an unresolved addendum", "the imaging record was paused during reconciliation", "the report is awaiting clinician acknowledgement"],
        "medication": ["the medication workflow contains an unresolved exception", "the pharmacy review was returned for clarification", "the order remains in the safety queue"],
        "infection": ["the infection-control item was escalated after no response", "the exposure follow-up remains incomplete", "the precaution update has not been acknowledged"],
        "patient_safety": ["the case review is still open", "the incident record was returned for a missing acknowledgement", "the safety item is waiting for documented follow-up"],
        "schedule": ["the revised roster has not been confirmed", "the coverage change remains pending", "the schedule item was returned to the staffing queue"],
        "blood_bank": ["the transfusion item is awaiting release", "the blood-product request remains unmatched", "the crossmatch workflow is pending acknowledgement"],
        "discharge": ["the documentation item remains unsigned", "the record was returned by the coding queue", "the discharge workflow is waiting for completion"],
        "education": ["the assigned module remains incomplete", "the competency item was escalated", "the training record has not been acknowledged"],
        "equipment": ["the device advisory is awaiting acknowledgement", "the inspection record was returned", "the equipment notice remains open"],
        "telehealth": ["the virtual consultation record is waiting for review", "the remote-care item was returned", "the telehealth workflow remains open"],
        "emergency": ["the acute-care item is awaiting handover", "the observation record remains open", "the escalation note was returned"],
        "icu": ["the critical-care review is still pending", "the monitoring item has not been acknowledged", "the ICU workflow contains an unresolved exception"],
        "oncology": ["the treatment-plan item remains pending", "the oncology record is waiting for acknowledgement", "the infusion workflow was returned"],
        "maternity": ["the maternity record is awaiting review", "the monitoring note remains incomplete", "the obstetric workflow was returned"],
        "renal": ["the renal-care item is awaiting release", "the dialysis review remains incomplete", "the renal workflow was returned"],
        "pathology": ["the specimen record is awaiting review", "the pathology addendum remains unreleased", "the tracking item was returned"],
        "rehabilitation": ["the therapy handover remains open", "the rehabilitation note is awaiting acknowledgement", "the care-plan item was returned"],
        "occupational_health": ["the employee-health record is awaiting review", "the fitness item remains open", "the screening workflow was returned"],
    },
    "admin": {},
    "it": {},
}


def _v32_load_history():
    try:
        with open(_V32_HISTORY_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _v32_save_history(rows):
    try:
        with open(_V32_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(rows[-3000:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _v32_choice(items, used=None):
    items = list(items)
    if used:
        fresh = [x for x in items if x not in used]
        if fresh:
            items = fresh
    return _V32_RNG.choice(items)


def _v32_session_bucket(role_type, language, difficulty):
    cycle = st.session_state.get("v32_cycle_id")
    if cycle is None:
        cycle = _V32_RNG.randrange(10_000_000, 99_999_999)
        st.session_state["v32_cycle_id"] = cycle
    key = f"v32_plan_memory_{cycle}_{role_type}_{language}_{difficulty}"
    return st.session_state.setdefault(key, {
        "semantic": [], "objects": [], "styles": [], "senders": [],
        "openings": [], "deadlines": [], "credentials": [], "subjects": [],
    })


def _v32_plan(role, index, language, difficulty, phase, is_phishing):
    """Generate a coherent plan only. It never writes links or tutor analysis."""
    diff = str(difficulty or "medium").lower()
    if diff not in V30_DIFFICULTY:
        diff = "medium"
    role_type = _v30_role_type(role)
    memory = _v32_session_bucket(role_type, language, diff)
    history = _v32_load_history()
    recent = {r.get("semantic") for r in history[-1200:] if r.get("role_type") == role_type}

    families = list(V30_KNOWLEDGE[role_type])
    # Try many compatible combinations before accepting one.
    for _ in range(120):
        family = _V32_RNG.choice(families)
        obj = _v32_choice(family["objects"], set(memory["objects"]))
        action = _V32_RNG.choice(family["actions"])
        sender = _v32_choice(family["senders"], set(memory["senders"]))
        signature = family["signatures"][family["senders"].index(sender) % len(family["signatures"])]
        styles = V32_PHISH_STRUCTURES if is_phishing else V32_LEGIT_STRUCTURES
        style = _v32_choice(styles, set(memory["styles"]))
        contexts = V32_FAMILY_CONTEXT.get(role_type, {}).get(family["id"], [family["event"]])
        context = _V32_RNG.choice(contexts)
        semantic = "|".join([role_type, family["id"], obj, action, style, context, "P" if is_phishing else "L"])
        if semantic not in recent and semantic not in memory["semantic"]:
            break

    plan = {
        "role_type": role_type, "language": language, "difficulty": diff,
        "phase": phase, "is_phishing": bool(is_phishing),
        "family_id": family["id"], "area": family["area"], "event": family["event"],
        "object": obj, "action": action, "sender": sender, "signature": signature,
        "structure": style, "operational_context": context,
        "channel": _V32_RNG.choice(V30_DIFFICULTY[diff]["allowed_channels"]),
        "semantic": semantic,
    }
    plan["fingerprint"] = "v32:" + str(abs(hash(semantic)))

    memory["semantic"].append(semantic); memory["objects"].append(obj)
    memory["styles"].append(style); memory["senders"].append(sender)
    for k in memory:
        memory[k] = memory[k][-40:]
    history.append({
        "semantic": semantic, "role_type": role_type, "difficulty": diff,
        "phase": phase, "family_id": family["id"], "object": obj, "style": style,
    })
    _v32_save_history(history)
    return plan


def _v32_subject(plan, urgency, style):
    obj = plan["object"].title()
    bank = {
        "pending_release": [f"Release Pending: {obj}", f"{obj} Awaiting Release"],
        "failed_review": [f"Review Could Not Be Completed: {obj}", f"Unresolved Review — {obj}"],
        "workflow_handover": [f"Before Your Next Shift — {obj}", f"Handover Item: {obj}"],
        "service_interruption": [f"Access Notice: {obj}", f"Service Alert — {obj}"],
        "escalation_notice": [f"Escalated: {obj} Awaiting Your Response", f"Escalation Notice — {obj}"],
        "case_return": [f"Returned to Queue: {obj}", f"Case Returned — {obj}"],
        "deadline_reminder": [f"Final Reminder — {obj}", f"Action Required {urgency.title()}: {obj}"],
        "record_exception": [f"Record Exception: {obj}", f"Correction Required — {obj}"],
        "coordination_request": [f"Coordination Request: {obj}", f"Response Requested — {obj}"],
        "unresolved_item": [f"Unresolved Item: {obj}", f"Pending Item — {obj}"],
        "status_change": [f"Status Change: {obj}", f"Update Required — {obj}"],
        "final_followup": [f"Final Follow-up: {obj}", f"Outstanding Response — {obj}"],
    }
    return _V32_RNG.choice(bank[style])


def _v32_easy_phishing(plan, role, index):
    ar = plan["language"] == "Arabic"
    recipient = _v30_recipient(role, index, plan["language"], plan["phase"])
    domain = _v32_choice(V32_FAKE_DOMAINS)
    mailbox = _v32_choice(V32_MAILBOXES)
    sender = f'{plan["sender"]} <{mailbox}@{domain}>'
    greeting = _v32_choice(V32_GENERIC_GREETINGS_AR if ar else V32_GENERIC_GREETINGS_EN)
    credential = _v32_choice(V32_CREDENTIAL_REQUESTS_AR if ar else V32_CREDENTIAL_REQUESTS_EN)
    deadline = _v32_choice(V32_DEADLINES_AR if ar else V32_DEADLINES_EN)
    slug = re.sub(r"[^a-z0-9]+", "-", plan["family_id"].lower()).strip("-")
    link = f"http://{domain}/{slug}/{_V32_RNG.randrange(1000,9999)}"
    subject = _v32_subject(plan, deadline if not ar else "اليوم", plan["structure"]) if not ar else f"إجراء مطلوب: {plan['object']}"
    obj, area, action, context, style = plan["object"], plan["area"], plan["action"], plan["operational_context"], plan["structure"]

    if ar:
        variants = {
            "pending_release": f"{greeting}،\n\nأصبح {obj} جاهزًا للإطلاق، لكن يلزم التحقق من الهوية أولًا. يرجى {credential} {deadline} حتى تتمكن من {action}.\n\nفتح السجل: {link}\n\nسيُسحب العنصر إذا لم يكتمل الإجراء.\n\n{plan['signature']}",
            "failed_review": f"{greeting}،\n\nتعذرت مراجعة {obj} ضمن {area}. يرجى {credential} {deadline} لفتح السجل ومراجعة التفاصيل.\n\nفتح عنصر المراجعة: {link}\n\nقد يتم تعليق الوصول مؤقتًا إذا لم يكتمل الإجراء.\n\nمع التحية،\n{plan['signature']}",
            "workflow_handover": f"{greeting}،\n\nيوجد عنصر تسليم مرتبط بـ {obj} ينتظر المتابعة {deadline}. يرجى {credential} لعرضه وتأكيد الاستلام.\n\nفتح عنصر التسليم: {link}\n\nقد يؤدي عدم التأكيد إلى حجب الوصول أثناء المناوبة.\n\n{plan['signature']}",
            "service_interruption": f"{greeting}،\n\nتم إيقاف الوصول إلى {obj} أثناء تحديث الخدمة. لاستعادته، يرجى {credential} {deadline}.\n\nاستعادة الوصول: {link}\n\nسيظل السجل غير متاح حتى اكتمال الخطوة.\n\n{plan['signature']}",
            "escalation_notice": f"{greeting}،\n\nتم تصعيد {obj} إلى {area} لعدم تسجيل استجابة. أكمل التحقق عبر {credential} {deadline}.\n\nمراجعة التصعيد: {link}\n\nقد تؤدي التأخيرات إلى تعليق وصولك لهذا المسار.\n\nشكرًا،\n{plan['signature']}",
            "case_return": f"{greeting}،\n\nأُعيد {obj} إلى قائمة المتابعة لأن خطوة التحقق لم تكتمل. يرجى {credential} {deadline} ثم {action}.\n\nعرض الحالة: {link}\n\nستبقى الحالة معلقة حتى استكمال الطلب.\n\n{plan['signature']}",
            "deadline_reminder": f"{greeting}،\n\nهذه متابعة أخيرة بخصوص {obj}. ما زالت الخطوة المطلوبة معلقة. يرجى {credential} {deadline} حتى تتمكن من {action}.\n\nعرض الحالة: {link}\n\nقد يؤدي عدم الرد إلى تقييد الوصول.\n\n{plan['signature']}",
            "record_exception": f"{greeting}،\n\nرُصد استثناء في سجل {obj}. يلزم {credential} {deadline} لتصحيح الحالة ومتابعة {action}.\n\nفتح الاستثناء: {link}\n\nسيبقى السجل محجوبًا حتى إنهاء الخطوة.\n\n{plan['signature']}",
            "coordination_request": f"{greeting}،\n\nيحتاج فريق {area} إلى ردك على {obj}. يرجى {credential} {deadline} حتى تتمكن من {action}.\n\nفتح الطلب: {link}\n\nقد يُعاد الطلب إلى قائمة الانتظار إذا لم يصل الرد.\n\n{plan['signature']}",
            "unresolved_item": f"{greeting}،\n\nما زال {obj} غير مكتمل ضمن {area}. يرجى {credential} {deadline} لفتح العنصر وإنهاء {action}.\n\nفتح العنصر: {link}\n\nقد يُعلّق الوصول إلى المسار حتى اكتمال الإجراء.\n\n{plan['signature']}",
            "status_change": f"{greeting}،\n\nتغيرت حالة {obj} بعد تحديث النظام. لتأكيد الحالة الجديدة، يرجى {credential} {deadline}.\n\nمراجعة التغيير: {link}\n\nلن يتاح السجل قبل اعتماد الخطوة.\n\n{plan['signature']}",
            "final_followup": f"{greeting}،\n\nلم نتلق ردًا على {obj}. يرجى {credential} {deadline} ثم {action}.\n\nمتابعة الطلب: {link}\n\nسيُغلق الطلب ويُقيّد الوصول إذا لم تكتمل الخطوة.\n\n{plan['signature']}",
        }
    else:
        variants = {
            "pending_release": f"{greeting},\n\n{obj.title()} is ready for release, but identity confirmation is required first. Please {credential} {deadline} so you can {action}.\n\nOpen the record: {link}\n\nThe item will be withdrawn if the action is not completed.\n\n{plan['signature']}",
            "failed_review": f"{greeting},\n\nA review item for {obj} could not be completed in {area}. Please {credential} {deadline} to open the record and review the details.\n\nOpen the review item: {link}\n\nIf the step is not completed, access may be suspended temporarily.\n\nRegards,\n{plan['signature']}",
            "workflow_handover": f"{greeting},\n\nA handover item linked to {obj} is waiting {deadline}. Please {credential} to view it and acknowledge receipt.\n\nOpen the handover item: {link}\n\nFailure to acknowledge it may block access during the shift.\n\n{plan['signature']}",
            "service_interruption": f"{greeting},\n\nAccess to {obj} was paused during a service update. To restore it, please {credential} {deadline}.\n\nRestore access: {link}\n\nThe record will remain unavailable until the step is completed.\n\n{plan['signature']}",
            "escalation_notice": f"{greeting},\n\n{obj.title()} has been escalated to {area} because no response was recorded. Complete verification by {credential} {deadline}.\n\nReview the escalation: {link}\n\nDelays may suspend your access to this workflow.\n\nThank you,\n{plan['signature']}",
            "case_return": f"{greeting},\n\n{obj.title()} was returned to the queue because the verification step was not completed. Please {credential} {deadline}, then {action}.\n\nView the case: {link}\n\nThe case will remain pending until the request is completed.\n\n{plan['signature']}",
            "deadline_reminder": f"{greeting},\n\nThis is a final follow-up regarding {obj}. The requested step is still outstanding. Please {credential} {deadline} so you can {action}.\n\nView the case: {link}\n\nAn unanswered request may be returned to the queue and may restrict access.\n\n{plan['signature']}",
            "record_exception": f"{greeting},\n\nA record exception was detected for {obj}. You must {credential} {deadline} to correct the status and {action}.\n\nOpen the exception: {link}\n\nThe record will stay locked until the step is finished.\n\n{plan['signature']}",
            "coordination_request": f"{greeting},\n\n{area} needs your response on {obj}. Please {credential} {deadline} so you can {action}.\n\nOpen the request: {link}\n\nThe request may be returned to the queue if no response is received.\n\n{plan['signature']}",
            "unresolved_item": f"{greeting},\n\n{obj.title()} remains unresolved in {area}. Please {credential} {deadline} to open the item and complete the required review.\n\nOpen the item: {link}\n\nAccess to the workflow may be suspended until the action is completed.\n\n{plan['signature']}",
            "status_change": f"{greeting},\n\nThe status of {obj} changed after a system update. To confirm the new status, please {credential} {deadline}.\n\nReview the change: {link}\n\nThe record will not be available until the step is approved.\n\n{plan['signature']}",
            "final_followup": f"{greeting},\n\nNo response has been recorded for {obj}. Please {credential} {deadline}, then {action}.\n\nFollow up on the request: {link}\n\nThe request may be closed and access restricted if the step is not completed.\n\n{plan['signature']}",
        }
    body = variants[style]

    # Stable grounded indicator contract. Evidence is guaranteed to exist.
    first_line = next(x.strip() for x in body.splitlines() if x.strip()).rstrip(",،")
    indicators = [
        _v30_indicator(1, "domain", "Non-official sender domain" if not ar else "نطاق مرسل غير رسمي",
                       (f"The sender uses {domain}, not the hospital's official domain." if not ar else f"يستخدم المرسل النطاق {domain} وليس نطاق المستشفى الرسمي."), domain, "from"),
        _v30_indicator(2, "urgency", "Strong urgency or threat" if not ar else "استعجال أو تهديد واضح",
                       ("The message uses a short deadline or an access consequence to pressure the recipient." if not ar else "تستخدم الرسالة مهلة قصيرة أو نتيجة مرتبطة بالوصول للضغط على المستلم."), deadline, "body"),
        _v30_indicator(3, "credential", "Direct credential request" if not ar else "طلب مباشر لبيانات الدخول",
                       ("The message directly asks for a password, staff PIN, login credentials, or verification code." if not ar else "تطلب الرسالة مباشرة كلمة مرور أو رقمًا سريًا أو بيانات دخول أو رمز تحقق."), credential, "body"),
        _v30_indicator(4, "link", "Visible external link" if not ar else "رابط خارجي ظاهر",
                       (f"The link points to the non-official domain {domain}." if not ar else f"يقود الرابط إلى النطاق غير الرسمي {domain}."), link, "link"),
        _v30_indicator(5, "greeting", "Generic greeting" if not ar else "تحية عامة",
                       ("The message does not address the recipient by name, which may indicate bulk targeting." if not ar else "لا تخاطب الرسالة المستلم باسمه، وقد يدل ذلك على إرسال جماعي."), first_line, "greeting"),
    ]
    why = ("The message combines a non-official domain, a direct credential request, an external link, and strong time pressure—clear beginner-level phishing indicators."
           if not ar else "تجمع الرسالة بين نطاق غير رسمي وطلب مباشر لبيانات الدخول ورابط خارجي وضغط زمني واضح، وهي مؤشرات تصيد مناسبة للمستوى المبتدئ.")
    tip = ("Never enter a password, staff PIN, or verification code through an unexpected email link. Open the official system directly or verify the request through a trusted channel."
           if not ar else "لا تُدخل كلمة مرور أو رقمًا سريًا أو رمز تحقق عبر رابط بريد غير متوقع. افتح النظام الرسمي مباشرة أو تحقق من الطلب عبر قناة موثوقة.")
    return {
        "from": sender, "to": recipient, "subject": subject, "body": body,
        "attachment": "", "suspicious_link": link, "suspicious_text": credential,
        "indicators": indicators, "why_risky": why, "learning_tip": tip,
        "is_phishing": True, "email_type": "Phishing", "attack_type": "Credential harvesting",
        "risk_level": "easy", "scenario_id": plan["fingerprint"], "scenario_meta": plan,
        "display_time": _V32_RNG.choice(["Today, 8:15 AM", "Today, 10:42 AM", "Monday, 2:31 PM", "Yesterday, 4:05 PM", "Thursday, 9:18 AM"]),
    }


def _v32_legitimate(plan, role, index):
    ar = plan["language"] == "Arabic"
    recipient = _v30_recipient(role, index, plan["language"], plan["phase"])
    person = _v30_display_name(recipient)
    sender = f'{plan["sender"]} <{_v32_choice(["notifications", "coordination", "quality", "clinical.ops", "records"])}@hospital.org>'
    obj, area, action, style = plan["object"], plan["area"], plan["action"], plan["structure"]
    if ar:
        subjects = {
            "brief_update": f"تحديث موجز: {obj}", "colleague_note": f"ملاحظة بخصوص {obj}",
            "meeting_preparation": f"{obj} للاجتماع القادم", "record_correction": f"تصحيح مطلوب: {obj}",
            "schedule_change": f"تغيير في الموعد: {obj}", "quality_followup": f"متابعة جودة: {obj}",
            "policy_notice": f"إشعار إرشادي: {obj}", "handover_summary": f"ملخص تسليم: {obj}",
            "request_for_comment": f"طلب ملاحظة: {obj}", "completion_notice": f"اكتمل تحديث {obj}",
            "clarification_request": f"استفسار بسيط: {obj}", "team_briefing": f"إحاطة الفريق: {obj}",
        }
        bodies = {
            "brief_update": f"عزيزي {person}،\n\nتم تحديث {obj} في {area}. يمكنك {action} من خلال مساحة العمل الداخلية عند توفر الوقت.\n\nمع التحية،\n{plan['signature']}",
            "colleague_note": f"مرحبًا {person}،\n\nأضفت ملاحظة قصيرة في سجل {obj}. هل يمكنك مراجعتها خلال المناوبة وإخباري إذا كانت هناك نقطة تحتاج تعديلًا؟\n\nشكرًا،\n{plan['signature']}",
            "meeting_preparation": f"عزيزي {person}،\n\nنجهز بند {obj} لاجتماع {area} القادم. يرجى إضافة تعليقك في النظام الداخلي قبل ظهر الأربعاء.\n\nشكرًا،\n{plan['signature']}",
            "record_correction": f"عزيزي {person}،\n\nلاحظنا اختلافًا بسيطًا في سجل {obj}. هل يمكنك مراجعة الحقل الأخير وتصحيحه في النظام المعتاد؟\n\nمع التقدير،\n{plan['signature']}",
            "schedule_change": f"عزيزي {person}،\n\nتم تعديل توقيت {obj}. سيظهر الموعد الجديد في الجدول الداخلي اليوم؛ يرجى الرد فقط إذا تعارض مع مناوبتك.\n\n{plan['signature']}",
            "quality_followup": f"عزيزي {person}،\n\nأصبحت خلاصة مراجعة الجودة الخاصة بـ {obj} متاحة في مساحة {area}. توجد ملاحظتان تحتاجان تعليق القسم قبل الاجتماع القادم.\n\nمع التحية،\n{plan['signature']}",
            "policy_notice": f"عزيزي {person}،\n\nنُشرت النسخة المحدثة من إرشادات {obj} في مكتبة السياسات الداخلية. سيبدأ تطبيقها الأسبوع القادم وسنغطي التغيير في إحاطة الفريق.\n\n{plan['signature']}",
            "handover_summary": f"عزيزي {person}،\n\nأضفت ملخص تسليم يتعلق بـ {obj} للفريق القادم. يرجى مراجعته في السجل وإضافة أي نقطة ناقصة قبل نهاية المناوبة.\n\nمع التحية،\n{plan['signature']}",
            "request_for_comment": f"مرحبًا {person}،\n\nهل يمكنك إضافة ملاحظتك على {obj} في مساحة {area} الداخلية قبل نهاية اليوم؟ لا يلزم الرد بالبريد إذا لم توجد تغييرات.\n\nشكرًا،\n{plan['signature']}",
            "completion_notice": f"عزيزي {person}،\n\nاكتمل تحديث {obj} وأصبح السجل متاحًا في النظام الداخلي. لا يلزم اتخاذ إجراء ما لم تلاحظ اختلافًا في التفاصيل.\n\nمع التقدير،\n{plan['signature']}",
            "clarification_request": f"مرحبًا {person}،\n\nلدينا استفسار بسيط بخصوص {obj}. يرجى مراجعة الملاحظة في السجل الداخلي وإضافة التصحيح عند توفر الوقت.\n\nشكرًا،\n{plan['signature']}",
            "team_briefing": f"عزيزي {person}،\n\nسيُناقش {obj} في إحاطة فريق {area} القادمة. أضفنا الملخص إلى مساحة العمل الداخلية للقراءة المسبقة.\n\nمع التحية،\n{plan['signature']}",
        }
        subject, body = subjects[style], bodies[style]
        tip = "الرسالة تستخدم نطاق المستشفى الرسمي وسياقًا مهنيًا طبيعيًا وقناة داخلية معروفة دون طلب بيانات حساسة."
    else:
        subjects = {
            "brief_update": f"Update: {obj.title()}", "colleague_note": f"A Note About {obj.title()}",
            "meeting_preparation": f"{obj.title()} for the Next Meeting", "record_correction": f"Correction Needed: {obj.title()}",
            "schedule_change": f"Schedule Change — {obj.title()}", "quality_followup": f"Quality Follow-up: {obj.title()}",
            "policy_notice": f"Guidance Notice: {obj.title()}", "handover_summary": f"Handover Summary: {obj.title()}",
            "request_for_comment": f"Comment Requested: {obj.title()}", "completion_notice": f"{obj.title()} Update Completed",
            "clarification_request": f"Quick Clarification: {obj.title()}", "team_briefing": f"Team Briefing — {obj.title()}",
        }
        bodies = {
            "brief_update": f"Dear {person},\n\nThe {obj} record has been updated in {area}. You can {action} it through the usual internal workspace when convenient.\n\nKind regards,\n{plan['signature']}",
            "colleague_note": f"Hello {person},\n\nI added a short note to the {obj} record. Could you review it during your shift and let me know if anything needs changing?\n\nThanks,\n{plan['signature']}",
            "meeting_preparation": f"Dear {person},\n\nWe are preparing the {obj} item for the next {area} meeting. Please add your comment in the internal workspace by Wednesday noon.\n\nThank you,\n{plan['signature']}",
            "record_correction": f"Dear {person},\n\nWe noticed a minor discrepancy in the {obj} record. Could you check the final field and correct it in the usual system?\n\nBest regards,\n{plan['signature']}",
            "schedule_change": f"Dear {person},\n\nThe timing for {obj} has changed. The revised slot will appear in the internal schedule today; please reply only if it conflicts with your shift.\n\nRegards,\n{plan['signature']}",
            "quality_followup": f"Dear {person},\n\nThe quality-review summary for {obj} is now available in the {area} workspace. Two observations need the department's comments before the next meeting.\n\nKind regards,\n{plan['signature']}",
            "policy_notice": f"Dear {person},\n\nThe revised guidance for {obj} has been published in the internal policy library. It takes effect next week and will be covered at the team briefing.\n\nRegards,\n{plan['signature']}",
            "handover_summary": f"Dear {person},\n\nI added a handover summary for {obj} for the incoming team. Please review it in the record and add anything missing before the end of the shift.\n\nKind regards,\n{plan['signature']}",
            "request_for_comment": f"Hello {person},\n\nCould you add your comment to {obj} in the internal {area} workspace before the end of the day? There is no need to reply by email if nothing has changed.\n\nThanks,\n{plan['signature']}",
            "completion_notice": f"Dear {person},\n\nThe update to {obj} has been completed and the record is available in the internal system. No action is needed unless you notice a discrepancy.\n\nBest regards,\n{plan['signature']}",
            "clarification_request": f"Hello {person},\n\nWe have a quick clarification regarding {obj}. Please check the note in the internal record and add the correction when convenient.\n\nThanks,\n{plan['signature']}",
            "team_briefing": f"Dear {person},\n\n{obj.title()} will be discussed at the next {area} team briefing. The summary has been added to the internal workspace for advance reading.\n\nKind regards,\n{plan['signature']}",
        }
        subject, body = subjects[style], bodies[style]
        tip = "The message uses the official hospital domain, a natural workplace context, and a known internal workflow without asking for sensitive information."
    return {
        "from": sender, "to": recipient, "subject": subject, "body": body,
        "attachment": "", "suspicious_link": "", "suspicious_text": "", "indicators": [],
        "why_risky": "", "learning_tip": tip, "is_phishing": False,
        "email_type": "Legitimate", "attack_type": "None", "risk_level": plan["difficulty"],
        "scenario_id": plan["fingerprint"], "scenario_meta": plan,
        "display_time": _V32_RNG.choice(["Today, 9:05 AM", "Tuesday, 11:20 AM", "Yesterday, 3:40 PM", "Thursday, 1:15 PM"]),
    }


def _v32_validate(result, plan):
    # Preserve all v31 validation and add semantic/session safeguards.
    if not _v31_validate(result, plan):
        return False
    body = str(result.get("body", "")); subject = str(result.get("subject", ""))
    if result.get("is_phishing") and plan["difficulty"] == "easy":
        if len(result.get("indicators", [])) != 5:
            return False
        link = str(result.get("suspicious_link", ""))
        if not link or body.count(link) != 1:
            return False
    # Reject repeated opening/subject inside the active cycle.
    mem = _v32_session_bucket(plan["role_type"], plan["language"], plan["difficulty"])
    opening = next((ln.strip().lower() for ln in body.splitlines() if ln.strip()), "")
    subj_key = re.sub(r"\W+", " ", subject.lower()).strip()
    if opening and opening in mem["openings"][:-1]:
        return False
    if subj_key and subj_key in mem["subjects"]:
        return False
    mem["openings"].append(opening); mem["subjects"].append(subj_key)
    mem["openings"] = mem["openings"][-40:]; mem["subjects"] = mem["subjects"][-40:]
    return True


def _v32_generate(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    phase = "assess" if assessment else "learn"
    result = None
    for attempt in range(10):
        plan = _v32_plan(role, index + attempt * 104729, language, difficulty, phase, is_phishing)
        if is_phishing and plan["difficulty"] == "easy":
            result = _v32_easy_phishing(plan, role, index)
        elif is_phishing:
            # Keep v31's proven medium/hard writer and indicator mapping.
            result = _v31_compose_phishing(plan, role, index)
        else:
            result = _v32_legitimate(plan, role, index)
        if _v32_validate(result, plan):
            try:
                evaluate_and_log_auto_scores(result, plan["difficulty"], language, is_phishing=bool(is_phishing))
            except Exception:
                pass
            return result
    return result


def generate_email(role, index, language, difficulty="medium"):
    return _v32_generate(role, index, language, difficulty, True, False)


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    return _v32_generate(role, index, language, difficulty, bool(is_phishing), True)

# =============================================================
# END RULE-GUIDED SCENARIO PLANNER v32
# =============================================================


# =============================================================
# SCENARIO ENGINE v33 — COMBINATORIAL PLANNER (SAFE UPGRADE)
# -------------------------------------------------------------
# This layer changes only scenario planning and message composition.
# It deliberately reuses the stable indicator objects, evidence mapping,
# link field, validator, renderer, scoring, UI, languages, and provider code.
# =============================================================

_V33_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_history_v33.json")
_V33_RNG = random.SystemRandom()

V33_PHISH_ARCHETYPES = [
    "release_gate", "exception_resolution", "handover_action", "queue_escalation",
    "deadline_followup", "service_restore", "record_reconciliation", "approval_hold",
    "case_assignment", "status_confirmation", "missing_acknowledgement", "workflow_lock",
    "review_returned", "coverage_change", "safety_followup", "document_release",
]

V33_LEGIT_ARCHETYPES = [
    "brief_update", "colleague_note", "meeting_preparation", "record_correction",
    "schedule_change", "quality_followup", "policy_notice", "handover_summary",
    "request_for_comment", "completion_notice", "clarification_request", "team_briefing",
    "shift_note", "meeting_minutes", "service_update", "reference_notice",
]

V33_TONES = [
    "formal", "concise", "operational", "collegial", "system_notice",
    "departmental", "case_focused", "follow_up",
]

V33_OPENING_MODES = [
    "state_change", "returned_item", "missed_response", "new_exception",
    "release_ready", "queue_notice", "follow_up", "service_event",
]

V33_PRESSURE_STYLES_EN = [
    "today", "within the next hour", "before your next shift", "before 3:00 PM",
    "before the case is returned", "before the queue closes", "before the handover window ends",
    "immediately",
]
V33_PRESSURE_STYLES_AR = [
    "اليوم", "خلال الساعة القادمة", "قبل مناوبتك القادمة", "قبل الساعة الثالثة مساءً",
    "قبل إعادة الحالة", "قبل إغلاق قائمة الانتظار", "قبل انتهاء فترة التسليم", "فورًا",
]

V33_CREDENTIAL_ACTIONS_EN = [
    "confirm your staff PIN", "provide the verification code", "verify your account details",
    "submit your login credentials", "enter your password", "approve the sign-in verification",
]
V33_CREDENTIAL_ACTIONS_AR = [
    "تأكيد الرقم السري للموظف", "إدخال رمز التحقق", "التحقق من بيانات الحساب",
    "إرسال بيانات الدخول", "إدخال كلمة المرور", "اعتماد طلب تسجيل الدخول",
]

V33_CTA_LABELS_EN = [
    "Open the record", "Review the item", "View the case", "Continue to the request",
    "Open the exception", "Review the handover", "Open the review", "View the update",
]
V33_CTA_LABELS_AR = [
    "فتح السجل", "مراجعة العنصر", "عرض الحالة", "متابعة الطلب",
    "فتح الاستثناء", "مراجعة التسليم", "فتح المراجعة", "عرض التحديث",
]

V33_CONSEQUENCES_EN = [
    "The item may be returned to the queue if the step is not completed.",
    "Access to this workflow may be restricted until the request is completed.",
    "The record will remain unavailable until verification is completed.",
    "The request may close automatically if no response is recorded.",
    "The case may be reassigned if the action is not completed.",
    "The workflow may remain locked until the step is finished.",
]
V33_CONSEQUENCES_AR = [
    "قد يُعاد العنصر إلى قائمة الانتظار إذا لم تكتمل الخطوة.",
    "قد يُقيّد الوصول إلى هذا المسار حتى اكتمال الطلب.",
    "سيظل السجل غير متاح حتى اكتمال التحقق.",
    "قد يُغلق الطلب تلقائيًا إذا لم تُسجل استجابة.",
    "قد تُعاد إسناد الحالة إذا لم يكتمل الإجراء.",
    "قد يظل المسار مقفلاً حتى إنهاء الخطوة.",
]

V33_SUBJECT_PATTERNS_EN = {
    "release_gate": ["Release Pending: {obj}", "{obj} Awaiting Release", "Release Hold — {obj}"],
    "exception_resolution": ["Correction Required — {obj}", "Record Exception: {obj}", "Review Exception — {obj}"],
    "handover_action": ["Handover Action Needed: {obj}", "Before the Next Shift — {obj}", "Handover Item — {obj}"],
    "queue_escalation": ["Escalation Notice — {obj}", "{obj} Returned to the Queue", "Escalated Response Needed: {obj}"],
    "deadline_followup": ["Final Follow-up: {obj}", "Outstanding Response — {obj}", "Action Needed Today: {obj}"],
    "service_restore": ["Service Access Notice: {obj}", "Restore Access — {obj}", "Temporary Access Hold: {obj}"],
    "record_reconciliation": ["Reconciliation Needed: {obj}", "Record Review Required — {obj}", "Unresolved Record: {obj}"],
    "approval_hold": ["Approval Hold: {obj}", "Approval Pending — {obj}", "Action Needed Before Release: {obj}"],
    "case_assignment": ["New Case Assignment: {obj}", "Case Response Needed — {obj}", "Assigned Item: {obj}"],
    "status_confirmation": ["Status Confirmation Required: {obj}", "Status Change — {obj}", "Confirm Updated Status: {obj}"],
    "missing_acknowledgement": ["Acknowledgement Missing: {obj}", "Response Not Recorded — {obj}", "Outstanding Acknowledgement: {obj}"],
    "workflow_lock": ["Workflow Locked: {obj}", "Access Blocked — {obj}", "Workflow Access Pending: {obj}"],
    "review_returned": ["Review Returned: {obj}", "{obj} Sent Back for Review", "Returned Review Item — {obj}"],
    "coverage_change": ["Coverage Change Awaiting Confirmation", "Schedule Update — {obj}", "Coverage Review Needed: {obj}"],
    "safety_followup": ["Safety Follow-up: {obj}", "Clinical Safety Review — {obj}", "Safety Response Required: {obj}"],
    "document_release": ["Document Release Pending: {obj}", "Release Review — {obj}", "Document Access Hold: {obj}"],
}

V33_SUBJECT_PATTERNS_AR = {
    k: [
        "إجراء مطلوب: {obj}", "متابعة مطلوبة: {obj}", "تحديث بانتظار الرد: {obj}"
    ] for k in V33_PHISH_ARCHETYPES
}

V33_GENERIC_GREETINGS_EN = [
    "Dear Staff Member", "Dear Healthcare Employee", "Dear Colleague",
    "Hello Clinical Team", "Attention Clinical Staff", "Dear Clinical Team",
]
V33_GENERIC_GREETINGS_AR = [
    "عزيزي الموظف", "عزيزي الزميل", "فريق العمل السريري",
    "الزملاء الأعزاء", "إلى أعضاء الفريق السريري", "عزيزي عضو الفريق",
]

V33_FAKE_DOMAINS = [
    "portal-confirm-now.net", "staff-verify-center.com", "secure-staff-check.net",
    "hospital-access-alert.co", "clinical-update-login.org", "care-review-portal.net",
    "clinical-record-check.com", "staff-access-review.net", "hospital-case-verify.org",
    "care-workflow-alert.co",
]
V33_MAILBOXES = [
    "review", "workflow", "alerts", "coordination", "updates", "case-notice",
    "records", "followup", "service", "response",
]


def _v33_load_history():
    try:
        with open(_V33_HISTORY_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _v33_save_history(rows):
    try:
        with open(_V33_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(rows[-6000:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _v33_bucket(role_type, language, difficulty):
    cycle = st.session_state.get("v33_cycle_id")
    if cycle is None:
        cycle = _V33_RNG.randrange(10_000_000, 99_999_999)
        st.session_state["v33_cycle_id"] = cycle
    key = f"v33_memory_{cycle}_{role_type}_{language}_{difficulty}"
    return st.session_state.setdefault(key, {
        "semantic": [], "families": [], "objects": [], "actions": [], "senders": [],
        "archetypes": [], "tones": [], "openings": [], "subjects": [], "credentials": [],
        "deadlines": [], "domains": [], "legit_archetypes": [],
    })


def _v33_pick(items, used=None):
    values = list(items)
    if used:
        fresh = [x for x in values if x not in set(used)]
        if fresh:
            values = fresh
    return _V33_RNG.choice(values)


def _v33_plan(role, index, language, difficulty, phase, is_phishing):
    """Combinatorial scenario plan. It never writes HTML, links, indicators, or tutor text."""
    diff = str(difficulty or "medium").lower()
    if diff not in V30_DIFFICULTY:
        diff = "medium"
    role_type = _v30_role_type(role)
    mem = _v33_bucket(role_type, language, diff)
    history = _v33_load_history()
    recent = {
        row.get("semantic") for row in history[-2500:]
        if row.get("role_type") == role_type and row.get("difficulty") == diff
    }

    families = list(V30_KNOWLEDGE[role_type])
    for _ in range(240):
        family = _v33_pick(families, [])
        obj = _v33_pick(family["objects"], mem["objects"][-24:])
        action = _v33_pick(family["actions"], mem["actions"][-20:])
        sender = _v33_pick(family["senders"], mem["senders"][-16:])
        sender_idx = family["senders"].index(sender)
        signature = family["signatures"][sender_idx % len(family["signatures"])]
        archetypes = V33_PHISH_ARCHETYPES if is_phishing else V33_LEGIT_ARCHETYPES
        archetype = _v33_pick(archetypes, mem["archetypes"][-12:] if is_phishing else mem["legit_archetypes"][-12:])
        tone = _v33_pick(V33_TONES, mem["tones"][-8:])
        opening_mode = _v33_pick(V33_OPENING_MODES, mem["openings"][-8:])
        semantic = "|".join([
            role_type, family["id"], obj, action, sender, archetype, tone,
            opening_mode, "P" if is_phishing else "L",
        ])
        # Strong session diversity: avoid same family three times in a short window.
        family_recent_count = mem["families"][-8:].count(family["id"])
        if semantic not in recent and semantic not in mem["semantic"] and family_recent_count < 2:
            break

    plan = {
        "role_type": role_type, "language": language, "difficulty": diff,
        "phase": phase, "is_phishing": bool(is_phishing),
        "family_id": family["id"], "area": family["area"], "event": family["event"],
        "object": obj, "action": action, "sender": sender, "signature": signature,
        "structure": archetype, "tone": tone, "opening_mode": opening_mode,
        "channel": _V33_RNG.choice(V30_DIFFICULTY[diff]["allowed_channels"]),
        "semantic": semantic,
    }
    plan["fingerprint"] = "v33:" + _v30_hashlib.sha256(semantic.encode("utf-8")).hexdigest()[:20]

    mem["semantic"].append(semantic); mem["families"].append(family["id"])
    mem["objects"].append(obj); mem["actions"].append(action); mem["senders"].append(sender)
    mem["tones"].append(tone); mem["openings"].append(opening_mode)
    (mem["archetypes"] if is_phishing else mem["legit_archetypes"]).append(archetype)
    for key in mem:
        mem[key] = mem[key][-60:]

    history.append({
        "semantic": semantic, "role_type": role_type, "difficulty": diff,
        "phase": phase, "family_id": family["id"], "object": obj,
        "archetype": archetype, "tone": tone,
    })
    _v33_save_history(history)
    return plan


def _v33_subject(plan):
    ar = plan["language"] == "Arabic"
    bank = V33_SUBJECT_PATTERNS_AR if ar else V33_SUBJECT_PATTERNS_EN
    templates = bank[plan["structure"]]
    return _v33_pick(templates).format(obj=plan["object"].title() if not ar else plan["object"])


def _v33_context_sentence(plan, ar=False):
    obj, area, mode = plan["object"], plan["area"], plan["opening_mode"]
    if ar:
        bank = {
            "state_change": f"تغيرت حالة {obj} في {area} بعد تحديث السجل.",
            "returned_item": f"أُعيد {obj} إلى قائمة {area} للمراجعة.",
            "missed_response": f"لم تُسجل استجابة على {obj} ضمن {area}.",
            "new_exception": f"تم اكتشاف استثناء جديد في سجل {obj}.",
            "release_ready": f"أصبح {obj} جاهزًا للإطلاق في {area}.",
            "queue_notice": f"يوجد عنصر متعلق بـ {obj} بانتظارك في قائمة {area}.",
            "follow_up": f"هذه متابعة بخصوص {obj} ضمن {area}.",
            "service_event": f"تأثر الوصول إلى {obj} أثناء تحديث الخدمة.",
        }
    else:
        bank = {
            "state_change": f"The status of {obj} changed after an update in {area}.",
            "returned_item": f"The {obj} item was returned to the {area} queue for review.",
            "missed_response": f"No response has been recorded for {obj} in {area}.",
            "new_exception": f"A new exception was detected in the {obj} record.",
            "release_ready": f"The {obj} item is ready for release in {area}.",
            "queue_notice": f"An item related to {obj} is waiting in the {area} queue.",
            "follow_up": f"This is a follow-up regarding {obj} in {area}.",
            "service_event": f"Access to {obj} was affected during a service update.",
        }
    return bank[mode]


def _v33_easy_phishing(plan, role, index):
    """Stable easy writer: one link field, evidence-grounded indicators, richer scenario wording."""
    ar = plan["language"] == "Arabic"
    recipient = _v30_recipient(role, index, plan["language"], plan["phase"])
    mem = _v33_bucket(plan["role_type"], plan["language"], plan["difficulty"])

    domain = _v33_pick(V33_FAKE_DOMAINS, mem["domains"][-8:])
    mailbox = _v33_pick(V33_MAILBOXES)
    sender = f'{plan["sender"]} <{mailbox}@{domain}>'
    greeting = _v33_pick(V33_GENERIC_GREETINGS_AR if ar else V33_GENERIC_GREETINGS_EN)
    credential = _v33_pick(V33_CREDENTIAL_ACTIONS_AR if ar else V33_CREDENTIAL_ACTIONS_EN, mem["credentials"][-5:])
    deadline = _v33_pick(V33_PRESSURE_STYLES_AR if ar else V33_PRESSURE_STYLES_EN, mem["deadlines"][-5:])
    cta = _v33_pick(V33_CTA_LABELS_AR if ar else V33_CTA_LABELS_EN)
    consequence = _v33_pick(V33_CONSEQUENCES_AR if ar else V33_CONSEQUENCES_EN)
    subject = _v33_subject(plan)
    slug = re.sub(r"[^a-z0-9]+", "-", plan["family_id"].lower()).strip("-")
    link = f"http://{domain}/{slug}/{_V33_RNG.randrange(1000,9999)}"
    context = _v33_context_sentence(plan, ar)
    obj, action, archetype = plan["object"], plan["action"], plan["structure"]

    if ar:
        action_sentence = f"يرجى {credential} {deadline} حتى تتمكن من {action}."
        connectors = {
            "release_gate": f"{context} يلزم تأكيد الهوية قبل إكمال الإطلاق.",
            "exception_resolution": f"{context} يلزم إكمال خطوة التحقق لتصحيح الحالة.",
            "handover_action": f"{context} يتطلب عنصر التسليم تأكيدًا قبل المتابعة.",
            "queue_escalation": f"{context} تم تصعيد الطلب بعد عدم تسجيل استجابة.",
            "deadline_followup": f"{context} ما زالت الخطوة المطلوبة معلقة.",
            "service_restore": f"{context} يلزم التحقق لاستعادة الوصول.",
            "record_reconciliation": f"{context} يلزم توثيق المراجعة قبل إغلاق العنصر.",
            "approval_hold": f"{context} ما زال الاعتماد متوقفًا حتى اكتمال التحقق.",
            "case_assignment": f"{context} يجب تأكيد الوصول قبل فتح المهمة.",
            "status_confirmation": f"{context} يلزم تأكيد الحالة الجديدة.",
            "missing_acknowledgement": f"{context} لم يتم تسجيل الإقرار المطلوب.",
            "workflow_lock": f"{context} ما زال المسار مقفلاً.",
            "review_returned": f"{context} أُعيدت المراجعة بسبب خطوة غير مكتملة.",
            "coverage_change": f"{context} يحتاج تعديل التغطية إلى تأكيد.",
            "safety_followup": f"{context} يحتاج بند السلامة إلى استجابة موثقة.",
            "document_release": f"{context} يتوقف إصدار المستند على إكمال التحقق.",
        }
        body = f"{greeting}،\n\n{connectors[archetype]} {action_sentence}\n\n{cta}: {link}\n\n{consequence}\n\n{plan['signature']}"
    else:
        action_sentence = f"Please {credential} {deadline} so you can {action}."
        connectors = {
            "release_gate": f"{context} Identity confirmation is required before release can continue.",
            "exception_resolution": f"{context} The verification step must be completed before the status can be corrected.",
            "handover_action": f"{context} The handover item requires confirmation before it can be acknowledged.",
            "queue_escalation": f"{context} The request has been escalated because no response was recorded.",
            "deadline_followup": f"{context} The requested step is still outstanding.",
            "service_restore": f"{context} Verification is required to restore access.",
            "record_reconciliation": f"{context} The review must be documented before the item can close.",
            "approval_hold": f"{context} Approval remains on hold until verification is completed.",
            "case_assignment": f"{context} Access confirmation is required before the task can be opened.",
            "status_confirmation": f"{context} The updated status requires confirmation.",
            "missing_acknowledgement": f"{context} The required acknowledgement has not been recorded.",
            "workflow_lock": f"{context} The workflow remains locked.",
            "review_returned": f"{context} The review was returned because a required step is incomplete.",
            "coverage_change": f"{context} The coverage change requires confirmation.",
            "safety_followup": f"{context} The safety item requires a documented response.",
            "document_release": f"{context} Document release is waiting for verification.",
        }
        body = f"{greeting},\n\n{connectors[archetype]} {action_sentence}\n\n{cta}: {link}\n\n{consequence}\n\n{plan['signature']}"

    first_line = next(x.strip() for x in body.splitlines() if x.strip()).rstrip(",،")
    indicators = [
        _v30_indicator(1, "domain", "Non-official sender domain" if not ar else "نطاق مرسل غير رسمي",
                       (f"The sender uses {domain}, not the hospital's official domain." if not ar else f"يستخدم المرسل النطاق {domain} وليس نطاق المستشفى الرسمي."), domain, "from"),
        _v30_indicator(2, "urgency", "Strong urgency or threat" if not ar else "استعجال أو تهديد واضح",
                       ("The message uses a short deadline or access consequence to pressure the recipient." if not ar else "تستخدم الرسالة مهلة قصيرة أو نتيجة مرتبطة بالوصول للضغط على المستلم."), deadline, "body"),
        _v30_indicator(3, "credential", "Direct credential request" if not ar else "طلب مباشر لبيانات الدخول",
                       ("The message directly asks for a password, staff PIN, login credentials, or verification code." if not ar else "تطلب الرسالة مباشرة كلمة مرور أو رقمًا سريًا أو بيانات دخول أو رمز تحقق."), credential, "body"),
        _v30_indicator(4, "link", "Visible external link" if not ar else "رابط خارجي ظاهر",
                       (f"The link points to the non-official domain {domain}." if not ar else f"يقود الرابط إلى النطاق غير الرسمي {domain}."), link, "link"),
        _v30_indicator(5, "greeting", "Generic greeting" if not ar else "تحية عامة",
                       ("The message does not address the recipient by name, which may indicate bulk targeting." if not ar else "لا تخاطب الرسالة المستلم باسمه، وقد يدل ذلك على إرسال جماعي."), first_line, "greeting"),
    ]
    why = ("The message combines a non-official sender domain, a direct credential request, a visible external link, and strong pressure—clear beginner-level phishing indicators."
           if not ar else "تجمع الرسالة بين نطاق غير رسمي وطلب مباشر لبيانات الدخول ورابط خارجي وضغط واضح، وهي مؤشرات تصيد مناسبة للمستوى المبتدئ.")
    tip = ("Never enter a password, staff PIN, or verification code through an unexpected email link. Open the official system directly or verify the request through a trusted channel."
           if not ar else "لا تُدخل كلمة مرور أو رقمًا سريًا أو رمز تحقق عبر رابط بريد غير متوقع. افتح النظام الرسمي مباشرة أو تحقق من الطلب عبر قناة موثوقة.")

    mem["credentials"].append(credential); mem["deadlines"].append(deadline); mem["domains"].append(domain)
    mem["credentials"] = mem["credentials"][-20:]; mem["deadlines"] = mem["deadlines"][-20:]; mem["domains"] = mem["domains"][-20:]

    return {
        "from": sender, "to": recipient, "subject": subject, "body": body,
        "attachment": "", "suspicious_link": link, "suspicious_text": credential,
        "indicators": indicators, "why_risky": why, "learning_tip": tip,
        "is_phishing": True, "email_type": "Phishing", "attack_type": "Credential harvesting",
        "risk_level": "easy", "scenario_id": plan["fingerprint"], "scenario_meta": plan,
        "display_time": _V33_RNG.choice(["Today, 8:15 AM", "Today, 10:42 AM", "Monday, 2:31 PM", "Yesterday, 4:05 PM", "Thursday, 9:18 AM"]),
    }


def _v33_legitimate(plan, role, index):
    """Legitimate writer delegates to the proven v32 writer after mapping a diverse archetype."""
    # Map the four new legitimate archetypes to safe established structures.
    mapping = {
        "shift_note": "handover_summary",
        "meeting_minutes": "meeting_preparation",
        "service_update": "brief_update",
        "reference_notice": "policy_notice",
    }
    legacy_structure = mapping.get(plan["structure"], plan["structure"])
    legacy_plan = dict(plan, structure=legacy_structure)
    return _v32_legitimate(legacy_plan, role, index)


def _v33_validate(result, plan):
    if not _v32_validate(result, plan):
        return False
    body = str(result.get("body", ""))
    subject = str(result.get("subject", ""))
    if result.get("is_phishing") and plan["difficulty"] == "easy":
        link = str(result.get("suspicious_link", ""))
        if body.count(link) != 1:
            return False
        # Every tutor evidence item must be present in its actual source field.
        source_blob = {
            "from": str(result.get("from", "")), "body": body,
            "link": link, "greeting": body,
        }
        for item in result.get("indicators", []):
            evidence = str(item.get("evidence", "")).strip()
            location = str(item.get("location", "body"))
            if evidence and evidence.lower() not in source_blob.get(location, body).lower():
                return False
    # Avoid lexical near-duplicates inside one cycle.
    mem = _v33_bucket(plan["role_type"], plan["language"], plan["difficulty"])
    subj_key = re.sub(r"\W+", " ", subject.lower()).strip()
    if subj_key in mem["subjects"]:
        return False
    mem["subjects"].append(subj_key)
    mem["subjects"] = mem["subjects"][-60:]
    return True


def _v33_generate(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    phase = "assess" if assessment else "learn"
    result = None
    for attempt in range(16):
        plan = _v33_plan(role, index + attempt * 104729, language, difficulty, phase, is_phishing)
        if is_phishing and plan["difficulty"] == "easy":
            result = _v33_easy_phishing(plan, role, index)
        elif is_phishing:
            # Preserve the proven medium/hard writer and indicator mapping.
            legacy_plan = dict(plan, structure=_v33_pick(V32_PHISH_STRUCTURES))
            result = _v31_compose_phishing(legacy_plan, role, index)
        else:
            result = _v33_legitimate(plan, role, index)
        if _v33_validate(result, plan):
            try:
                evaluate_and_log_auto_scores(result, plan["difficulty"], language, is_phishing=bool(is_phishing))
            except Exception:
                pass
            return result
    return result


def generate_email(role, index, language, difficulty="medium"):
    return _v33_generate(role, index, language, difficulty, True, False)


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    return _v33_generate(role, index, language, difficulty, bool(is_phishing), True)

# =============================================================
# END SCENARIO ENGINE v33
# =============================================================

# ══════════════════════════════════════════════════════════════
# SIDEBAR — زر القفل السري في الأسفل
# ══════════════════════════════════════════════════════════════
# زر القفل السري — أسفل يسار الصفحة الرئيسية فقط، صغير جداً وشفاف
st.markdown("""
<style>
#MainMenu,header,footer{visibility:hidden;}
[data-testid="stSidebar"]{display:none !important;}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("page","home") == "home":
    if st.button("🔒", key="secret_admin_btn", help=""):
        st.session_state["page"] = "admin"
        st.rerun()

    # نستخدم MutationObserver لأن الزر قد يُعاد رسمه بعد تحميل الصفحة
    import streamlit.components.v1 as components
    components.html("""
    <script>
    function styleSecretLock() {
        const doc = window.parent.document;
        const btns = doc.querySelectorAll('button');
        for (const b of btns) {
            if (b.innerText.trim() === '🔒') {
                b.style.setProperty('background', 'transparent', 'important');
                b.style.setProperty('border', 'none', 'important');
                b.style.setProperty('box-shadow', 'none', 'important');
                b.style.setProperty('font-size', '0.6rem', 'important');
                b.style.setProperty('color', 'rgba(255,255,255,0.10)', 'important');
                b.style.setProperty('padding', '0px', 'important');
                b.style.setProperty('min-height', 'unset', 'important');
                b.style.setProperty('height', '16px', 'important');
                b.style.setProperty('width', '16px', 'important');
                b.style.setProperty('line-height', '1', 'important');
                const wrapper = b.closest('div[data-testid="stButton"]') || b.closest('.element-container');
                if (wrapper) {
                    wrapper.style.setProperty('position', 'fixed', 'important');
                    wrapper.style.setProperty('bottom', '4px', 'important');
                    wrapper.style.setProperty('left', '4px', 'important');
                    wrapper.style.setProperty('z-index', '99999', 'important');
                    wrapper.style.setProperty('width', '18px', 'important');
                    wrapper.style.setProperty('margin', '0', 'important');
                    wrapper.style.setProperty('padding', '0', 'important');
                    // أيضاً نضبط أي عنصر أب يحتوي على padding كبير
                    let parent = wrapper.parentElement;
                    if (parent) {
                        parent.style.setProperty('position', 'static', 'important');
                    }
                }
            }
        }
    }
    styleSecretLock();
    const observer = new MutationObserver(styleSecretLock);
    observer.observe(window.parent.document.body, {childList:true, subtree:true});
    setInterval(styleSecretLock, 500);
    </script>
    """, height=0, width=0)

# ══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ══════════════════════════════════════════════════════════════
pg = st.session_state.get("page", "home")
if pg == "admin":
    page_admin()
else:
    {
        "home":page_home,"login":page_login,"learning":page_learning,
        "complete":page_complete,"assessment":page_assessment,
        "results":page_results,"report":page_report,
        "phishing_caught":page_phishing_caught
    }.get(pg, page_home)()
