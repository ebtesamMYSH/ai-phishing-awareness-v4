# =============================================================
# AI Phishing Awareness Training Tool
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

def scenario_card_to_prompt(card, difficulty, is_ar=False):
    """Build a dynamic scenario instruction.

    The scenario card gives the healthcare idea. The content-shape engine gives
    style/structure/subject/signature variation. The API still writes the final
    email, but it is now constrained by both the 300-card content library and the
    difficulty framework.
    """
    attack_pool = _ATTACK_BY_DIFFICULTY.get(difficulty, _ATTACK_BY_DIFFICULTY["medium"])
    attack = random.choice(attack_pool)
    shape = build_content_shape(card.get("role_type", "clinical"), difficulty, is_ar)
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
    scenario_instruction = scenario_card_to_prompt(scenario_card, difficulty, is_ar)
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
    scenario_instruction = scenario_card_to_prompt(scenario_card, difficulty, is_ar)
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
        if _sl and _sl not in body_raw and not has_link_button:
            body_raw = _insert_before_signature(body_raw, _sl)
        elif _sl and _sl in body_raw:
            # Link already present somewhere — make sure it isn't sitting
            # detached after the signature (model's own placement).
            body_raw = _reposition_trailing_lone_link(body_raw, _sl)

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
    # SAFETY NET: some providers still print the raw URL as plain
    # text (e.g. in the signature) EVEN THOUGH it is also being
    # rendered as a real QR image or a real button below. If we
    # Remove duplicate URL — strip any standalone line that is exactly the suspicious_link
    # This applies ALWAYS (not just when has_qr or has_link_button) to prevent double display
    if suspicious_link:
        bare_link_pattern = re.escape(suspicious_link)
        bare_no_scheme    = re.escape(re.sub(r'^https?://', '', suspicious_link))
        # Remove if it appears as a standalone line (with or without http prefix)
        body_raw = re.sub(rf'^[ \t]*{bare_link_pattern}[ \t]*$\n?', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(rf'^[ \t]*{bare_no_scheme}[ \t]*$\n?', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    # For easy level: URL should appear once as plain text in body — add only if missing
    if _difficulty == "easy" and suspicious_link and not has_qr and not has_link_button:
        if suspicious_link not in body_raw:
            link_bare = re.sub(r'^https?://', '', suspicious_link)
            if link_bare not in body_raw:
                body_raw = body_raw.rstrip() + f'\n\n{suspicious_link}'

    # For medium/hard: if no button and no QR and link missing — add as fallback only for medium
    if _difficulty == "medium" and suspicious_link and not has_qr and not has_link_button:
        link_bare = re.sub(r'^https?://', '', suspicious_link)
        if suspicious_link not in body_raw and link_bare not in body_raw:
            body_raw = body_raw.rstrip() + f'\n\n{suspicious_link}'

    has_attachment  = bool((email.get("attachment") or "").strip())

    # Remove duplicate attachment filename that sometimes appears as plain text at end
    _att_name = (email.get("attachment") or "").strip()
    if _att_name:
        att_escaped = re.escape(_att_name)
        body_raw = re.sub(rf'^\s*Attachment\s*:\s*{att_escaped}\s*$', '', body_raw, flags=re.MULTILINE|re.IGNORECASE)
        body_raw = re.sub(rf'^\s*{att_escaped}\s*$', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    body_html   = html_lib.escape(body_raw)
    badge_count = [4 if has_attachment else 3]

    def make_badge(n, color="#DC2626"):
        return (f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                f'width:20px;height:20px;border-radius:50%;background:{color};color:white;'
                f'font-size:.7rem;font-weight:800;margin:0 3px;vertical-align:middle;">{n}</span>')

    def next_badge():
        b = badge_count[0]; badge_count[0] += 1; return b

    if show_badges:
        if suspicious_text:
            safe_s = html_lib.escape(suspicious_text)
            if safe_s in body_html:
                b = next_badge()
                body_html = body_html.replace(safe_s,
                    f'<span style="border:2px solid rgba(239,68,68,.6);border-radius:8px;'
                    f'padding:.2rem .5rem;background:rgba(239,68,68,.08);color:#FCA5A5;'
                    f'box-decoration-break:clone;-webkit-box-decoration-break:clone;">'
                    f'{make_badge(b)}{safe_s}</span>', 1)

        if suspicious_link:
            safe_l = html_lib.escape(suspicious_link)
            if has_qr or has_link_button:
                pass  # rendered as a real QR image / real button below instead
            elif safe_l in body_html:
                b = next_badge()
                body_html = body_html.replace(safe_l,
                    f'<span style="border:2px solid rgba(239,68,68,.6);border-radius:6px;'
                    f'padding:.2rem .5rem;background:rgba(239,68,68,.08);color:#60A5FA;'
                    f'text-decoration:underline;box-decoration-break:clone;'
                    f'-webkit-box-decoration-break:clone;">{make_badge(b)}{safe_l}</span>', 1)
            else:
                b = next_badge()
                body_html += (f'<br><br><span style="border:2px solid rgba(239,68,68,.6);'
                              f'border-radius:6px;padding:.2rem .5rem;background:rgba(239,68,68,.08);'
                              f'color:#60A5FA;text-decoration:underline;box-decoration-break:clone;'
                              f'-webkit-box-decoration-break:clone;">'
                              f'{make_badge(b)}{html_lib.escape(suspicious_link)}</span>')

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

    b_from = make_badge(1) if show_badges else ""
    b_subj = make_badge(2) if show_badges else ""
    b_att  = make_badge(3) if show_badges else ""

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

    # Dynamic random email time (realistic variation)
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
.stButton>button:hover,.stButton>button:focus{{background:linear-gradient(90deg,#0B4FA8,#0284C7);color:white;border-color:#1EA7FF !important;}}
.start-btn>button{{min-height:56px !important;background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.65) !important;font-size:1.05rem !important;font-weight:900 !important;border-radius:14px !important;}}
.start-btn>button:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border-color:#1EA7FF !important;}}
div[data-baseweb="select"] *{{color:#EAF4FF!important;}}
div[data-baseweb="select"] > div{{background:rgba(15,23,42,.82)!important;border:1px solid rgba(37,99,235,.65)!important;border-radius:12px!important;}}
div[data-baseweb="popover"] *{{color:#EAF4FF!important;}}
.stSelectbox>div>div,.stTextInput>div>div>input{{background-color:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;border-radius:12px !important;min-height:48px;direction:{dir_attr};text-align:{text_align};}}
div[data-baseweb="select"] span{{color:white !important;}}
div[data-baseweb="popover"] ul li{{text-align:{text_align} !important;direction:{dir_attr} !important;}}
.footer-bar{{margin-top:2rem;padding:1.5rem 0;border-top:1px solid rgba(37,99,235,.35);display:flex;justify-content:space-between;align-items:center;color:#7DD3FC;font-size:.95rem;direction:{dir_attr};}}
.footer-side{{display:flex;align-items:center;gap:.8rem;}}
.diff-btn>button{{width:100% !important;min-height:52px !important;border-radius:14px !important;font-weight:800 !important;font-size:.95rem !important;transition:.2s ease !important;background:rgba(2,6,23,.55) !important;border:2px solid rgba(37,99,235,.35) !important;color:#94A3B8 !important;}}
.diff-btn>button:hover{{background:rgba(11,79,168,.25) !important;border-color:#1EA7FF !important;color:#FFFFFF !important;}}
.diff-btn-sel>button{{background:linear-gradient(135deg,#0B4FA8,#0284C7) !important;border:2px solid #1EA7FF !important;color:#FFFFFF !important;box-shadow:0 0 18px rgba(30,167,255,.3) !important;opacity:1 !important;cursor:default !important;pointer-events:none !important;}}
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
        cur_lang  = (st.session_state.get("language") or "")
        en_cls = "lang-btn-sel" if cur_lang == "English" else "lang-btn"
        ar_cls = "lang-btn-sel" if cur_lang == "Arabic"  else "lang-btn"
        st.markdown(f"""<style>
.lang-btn button {{background:rgba(15,23,42,.78) !important;border:1px solid rgba(37,99,235,.55) !important;color:#EAF4FF !important;}}
.lang-btn-sel button {{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border:2px solid #1EA7FF !important;color:white !important;box-shadow:0 0 14px rgba(30,167,255,.35) !important;}}
.lang-btn-sel button:hover,.lang-btn-sel button:focus,.lang-btn-sel button:active {{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border:2px solid #1EA7FF !important;color:white !important;}}
</style>""", unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="{en_cls}">', unsafe_allow_html=True)
            st.button("English", key="english", on_click=set_language, args=("English",), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="{ar_cls}">', unsafe_allow_html=True)
            st.button("العربية", key="arabic",  on_click=set_language, args=("Arabic",), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
                is_sel  = current_diff == dk
                css_cls = "diff-btn diff-btn-sel" if is_sel else "diff-btn"
                st.markdown(f'<div class="{css_cls}">', unsafe_allow_html=True)
                if st.button(lbl, key=f"diff_{dk}", use_container_width=True):
                    st.session_state["difficulty"] = dk
                    st.session_state["diff_explicitly_chosen"] = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

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
                    pcss = "diff-btn diff-btn-sel" if is_psel else "diff-btn"
                    st.markdown(f'<div class="{pcss}">', unsafe_allow_html=True)
                    if st.button(plbl, key=f"prov_{pk}", use_container_width=True):
                        set_active_provider(pk)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:.72rem;color:#64748B;margin-top:.3rem;direction:{dir_attr};">Active: <b style="color:#F59E0B;">{provider_options.get(cur_provider,"")}</b></div>', unsafe_allow_html=True)

        st.markdown('<div class="start-btn" style="margin-top:.8rem;">',unsafe_allow_html=True)
        if st.button(t("Start Personalised Training","ابدأ التدريب المخصص"),key="start_training", use_container_width=True):
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
        for ind in indicators:
            row_dir = 'rtl' if is_arabic else 'ltr'
            pad     = 'padding-right:2rem;' if is_arabic else 'padding-left:2rem;'
            ta2     = 'right' if is_arabic else 'left'
            indicators_html += f"""
<div style="margin-bottom:1rem;">
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.3rem;direction:{row_dir};">
    <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;background:#DC2626;color:white;font-size:.75rem;font-weight:800;flex-shrink:0;">{(ind.get('number') or '')}</span>
    <span style="font-weight:700;color:#E2E8F0;font-size:.95rem;">{(ind.get('title') or '')}</span>
  </div>
  <div style="color:#94A3B8;font-size:.9rem;line-height:1.65;{pad};direction:{row_dir};text-align:{ta2};">{(ind.get('description') or '')}</div>
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
.stTextInput div[data-baseweb="base-input"]{{
    background:transparent!important;
    background-color:transparent!important;
    border:none!important;
    box-shadow:none!important;
}}
.stTextInput div[data-baseweb="input"]{{
    background:rgba(15,23,42,.70)!important;
    border:1.5px solid rgba(37,99,235,.65)!important;
    border-radius:12px!important;
    box-shadow:0 0 0 1px rgba(30,167,255,.10)!important;
}}
.stTextInput div[data-baseweb="input"]:focus-within{{
    border-color:#1EA7FF!important;
    box-shadow:0 0 0 2px rgba(30,167,255,.18)!important;
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

def _insert_before_signature(body, marker):
    """Place a link/QR/button where users expect it: immediately after the
    sentence that refers to accessing/clicking/scanning, otherwise before
    the closing signature. This prevents links appearing after Best regards."""
    body = (body or "").rstrip()
    marker = (marker or "").strip()
    if not body:
        return marker
    if marker and marker in body:
        return _reposition_trailing_lone_link(body, marker)

    lines = body.split("\n")
    cue_re = re.compile(r"(link below|following link|access it|open it|view it|click|scan|الرابط|اضغط|افتح|امسح|الوصول|للمراجعة)", re.I)
    for i in range(len(lines) - 1, -1, -1):
        if cue_re.search(lines[i]):
            new_lines = lines[:i+1] + [marker] + lines[i+1:]
            return "\n".join(new_lines).strip()

    for i in range(len(lines) - 1, -1, -1):
        if _SALUTATION_LINE_RE.match(lines[i].strip()):
            new_lines = lines[:i] + [marker, ""] + lines[i:]
            return "\n".join(new_lines).strip()

    paragraphs = re.split(r'\n\s*\n', body)
    if len(paragraphs) >= 2:
        paragraphs.insert(len(paragraphs) - 1, marker)
        return "\n\n".join(p.strip() for p in paragraphs if p.strip())
    return body + "\n\n" + marker

_SALUTATION_LINE_RE = re.compile(
    r'^(regards|best regards|kind regards|warm regards|thank you|sincerely|respectfully|'
    r'شكرا|شكراً|مع تحياتي|تحياتي|أطيب التحيات|وتفضلوا بقبول)',
    re.I,
)

def _reposition_trailing_lone_link(body, link):
    """Some providers write the referring sentence ('click the link
    below...') correctly in-flow, but then place the actual bare URL as
    its own line AFTER the closing signature instead of right after
    that sentence — sometimes separated by a blank line (paragraph
    break), sometimes by a single newline (signature lines joined to
    the link with no blank line at all). This is independent of our own
    code (which only appends a link if one is completely missing) —
    this catches the model's OWN misplacement in BOTH formatting cases
    by working line-by-line rather than only on blank-line paragraphs."""
    link = (link or "").strip()
    raw_body = body or ""
    if not link or not raw_body.strip():
        return raw_body

    lines = raw_body.rstrip().split("\n")
    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return raw_body
    last_line = lines[idx].strip()
    if not (last_line == link or (link in last_line and len(last_line) <= len(link) + 6)):
        return raw_body  # last line isn't a bare/near-bare link — nothing to fix

    before = lines[:idx]
    before_text = "\n".join(before).rstrip()
    if not before_text:
        return raw_body

    # Case 1: there's a blank-line paragraph break before the link —
    # insert the link as its own paragraph just before the last one
    # (assumed to be the signature).
    paragraphs = re.split(r'\n\s*\n', before_text)
    if len(paragraphs) >= 2:
        paragraphs.insert(len(paragraphs) - 1, last_line)
        return "\n\n".join(p.strip() for p in paragraphs if p.strip())

    # Case 2: everything is joined with single newlines (no blank-line
    # break at all) — find the closing salutation line ("Regards,"/
    # "Best regards,"/etc.) and insert the link right before it.
    for i in range(len(before) - 1, -1, -1):
        if _SALUTATION_LINE_RE.match(before[i].strip()):
            new_lines = before[:i] + [last_line, ""] + before[i:]
            return "\n".join(new_lines).rstrip()

    # Fallback: no salutation line found — insert two lines before the
    # very end rather than leaving the link fully detached.
    insert_at = max(0, len(before) - 2)
    new_lines = before[:insert_at] + [last_line, ""] + before[insert_at:]
    return "\n".join(new_lines).rstrip()

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
# ROOT-CONTROLLED GENERATION ENGINE — deterministic guardrail layer
# -------------------------------------------------------------
# Purpose: stop role drift, difficulty drift, QR misuse, commercial
# themes, and links appearing after the signature. The AI providers
# can still be used elsewhere, but emails shown to trainees are now
# built from a strict scenario blueprint that matches the selected
# role + difficulty before rendering.
# =============================================================

_SIGNATURE_RE_ROOT = re.compile(r"\n\s*(Best regards|Regards|Sincerely|Thank you|Thank You|مع التحية|تحياتي|مع الشكر|شكرًا|وتفضلوا)\b", re.I)

_ROOT_RECIPIENTS = {
    "clinical": [
        ("Dr. Yousef Alghamdi", "dr.yousef.alghamdi@hospital.org"),
        ("Dr. Ahmed Alotaibi", "dr.ahmed.alotaibi@hospital.org"),
        ("Nurse Reem Alzahrani", "n.reem.alzahrani@hospital.org"),
        ("Pharmacist Ziad Alharbi", "ph.ziad.alharbi@hospital.org"),
        ("Dr. Omar Alharthy", "dr.omar.alharthy@hospital.org"),
        ("Nurse Maha Alsubaie", "n.maha.alsubaie@hospital.org"),
    ],
    "admin": [
        ("Reem Alzahrani", "m.reem.alzahrani@hospital.org"),
        ("Abdullah Alqahtani", "m.abdullah.alqahtani@hospital.org"),
        ("Hind Alrashidi", "m.hind.alrashidi@hospital.org"),
        ("Sultan Alghamdi", "m.sultan.alghamdi@hospital.org"),
        ("Dalal Alzahrani", "m.dalal.alzahrani@hospital.org"),
        ("Omar Albaqami", "m.omar.albaqami@hospital.org"),
    ],
    "it": [
        ("Mohammed Alshahri", "t.mohammed.alshahri@hospital.org"),
        ("Rania Almalki", "t.rania.almalki@hospital.org"),
        ("Yusuf Aljuhani", "t.yusuf.aljuhani@hospital.org"),
        ("Lama Alumari", "t.lama.alumari@hospital.org"),
        ("Bandar Althubaiti", "t.bandar.althubaiti@hospital.org"),
        ("Nadia Alsalmi", "t.nadia.alsalmi@hospital.org"),
    ],
    "other": [
        ("Sara Alqahtani", "s.sara.alqahtani@hospital.org"),
        ("Khalid Alharbi", "s.khalid.alharbi@hospital.org"),
        ("Faisal Alzahrani", "s.faisal.alzahrani@hospital.org"),
        ("Nora Alotaibi", "s.nora.alotaibi@hospital.org"),
        ("Ahmed Alshamri", "s.ahmed.alshamri@hospital.org"),
        ("Hessa Aldosari", "s.hessa.aldosari@hospital.org"),
    ],
}

_ROOT_BLUEPRINTS = {
    "clinical": {
        "easy": [
            ("Patient Records Team", "patient-alert-review.com", "Urgent Patient Access Review Required", "patient access", "http://patient-alert-review.com/update", "verify your password immediatly", ["immediatly", "requiered"]),
            ("Clinical Reports Team", "fakehospitaldata.com", "Urgent Patient Data Action Required", "latest patient report", "http://fakehospitaldata.com/login", "provide your password immediatly", ["immediatly", "acess"]),
            ("Blood Bank Alert Team", "bloodbank-alerts-fake.com", "Blood Bank System Access Warning", "blood bank records", "http://bloodbank-alerts-fake.com/verify", "confirm your login details now", ["recieve", "urgant"]),
        ],
        "medium": [
            ("Clinical Systems Support", "hospital-clinical.net", "EMR Medication Review Notice", "medication review queue", "http://hospital-clinical.net/review", "confirm your clinical-system access within 48 hours", ["recieve"]),
            ("Lab Coordination Center", "hospital-lab.net", "Pending Lab Result Review", "pending lab result queue", "http://hospital-lab.net/result-check", "update your access before tomorrow's shift", ["attachement"]),
            ("Radiology Workflow Desk", "hospital-radiology.net", "PACS Image Review Request", "radiology image queue", "http://hospital-radiology.net/pacs-review", "confirm access within 48 hours", ["acess"]),
        ],
        "hard": [
            ("Dr. Mohammed Al-Ghamdi, Clinical Informatics Consultant", "hospital-clinical.org.sa", "Routine EMR Protocol Confirmation", "EMR protocol confirmation", "https://hospital-clinical.org.sa/secure-review", "confirm the new handover protocol", []),
            ("Dr. Noura Al-Qahtani, Medication Safety Lead", "hospital-pharmacy.org.sa", "Medication Reconciliation Workflow Update", "medication reconciliation workflow", "https://hospital-pharmacy.org.sa/workflow", "acknowledge the updated medication reconciliation procedure", []),
            ("Dr. Fahad Al-Harbi, Radiology Governance Lead", "hospital-radiology.org.sa", "Routine PACS Access Validation", "PACS access validation", "https://hospital-radiology.org.sa/validation", "complete the routine PACS validation", []),
        ],
    },
    "admin": {
        "easy": [
            ("Billing Team", "fake-billing-alert.com", "Urgent Billing Portal Password Check", "billing portal", "http://fake-billing-alert.com/update", "send your password immediatly", ["immediatly", "requiered"]),
            ("Insurance Office", "insurance-hospital-fake.com", "Insurance File Access Warning", "insurance files", "http://insurance-hospital-fake.com/login", "verify your login NOW", ["recieve", "acess"]),
            ("Records Office", "patient-records-fake.com", "Patient Records Account Closure", "patient records office", "http://patient-records-fake.com/verify", "provide your password today", ["urgant", "acess"]),
        ],
        "medium": [
            ("Revenue Cycle Support", "hospital-billing.net", "Billing Queue Update Needed", "billing queue", "http://hospital-billing.net/queue", "confirm your billing portal access within 48 hours", ["recieve"]),
            ("Procurement Support Center", "hospital-procurement.net", "Supplier Contract Review", "supplier contract review", "http://hospital-procurement.net/review", "update your procurement access before the review window closes", ["seperate"]),
            ("Insurance Claims Desk", "hospital-claims.net", "Claims Batch Confirmation", "insurance claims batch", "http://hospital-claims.net/confirm", "confirm claim-batch access within 48 hours", ["acess"]),
        ],
        "hard": [
            ("Ms. Reem Al-Mutairi, Revenue Cycle Manager", "hospital-billing.org.sa", "Routine Claims Reconciliation Confirmation", "claims reconciliation", "https://hospital-billing.org.sa/reconcile", "confirm the reconciliation worksheet", []),
            ("Mr. Abdullah Al-Dossari, Procurement Director", "hospital-procurement.org.sa", "Vendor Contract Renewal Validation", "vendor contract renewal", "https://hospital-procurement.org.sa/contracts", "validate the supplier renewal packet", []),
            ("Ms. Huda Al-Salem, Patient Access Manager", "hospital-access.org.sa", "Appointment Template Governance Review", "appointment template review", "https://hospital-access.org.sa/templates", "confirm the clinic template changes", []),
        ],
    },
    "it": {
        "easy": [
            ("Security Team", "fakealertsys.com", "Immediate MFA Verification Required", "MFA portal", "http://secure-update.fakealertsys.com", "provide your password immediatly", ["immediatly", "requiered"]),
            ("Server Admin", "server-alert-fake.com", "VPN Account Closure Today", "VPN account", "http://server-alert-fake.com/vpn", "send your login details NOW", ["acess", "urgant"]),
            ("Helpdesk Team", "helpdesk-fake-hospital.com", "Firewall Console Access Warning", "firewall console", "http://helpdesk-fake-hospital.com/firewall", "verify your password today", ["recieve", "immediatly"]),
        ],
        "medium": [
            ("IT Support Center", "hospital-it.net", "VPN Profile Update Required", "VPN profile", "http://hospital-it.net/vpn-update", "confirm your VPN profile within 48 hours", ["acess"]),
            ("Infrastructure Support", "hospital-server.net", "Server Backup Verification", "server backup console", "http://hospital-server.net/backup", "update your backup-console access before tonight", ["recieve"]),
            ("Certificate Services", "hospital-cert.net", "SSL Certificate Renewal Check", "SSL certificate console", "http://hospital-cert.net/renew", "confirm certificate access within 48 hours", ["requiered"]),
        ],
        "hard": [
            ("Eng. Yasser Al-Qahtani, Infrastructure Operations Lead", "hospital-it.org.sa", "Routine Privileged Access Review", "privileged access review", "https://hospital-it.org.sa/access-review", "validate the privileged-access review", []),
            ("Eng. Sara Al-Harbi, Cybersecurity Governance Lead", "hospital-security.org.sa", "Endpoint Exception Register Confirmation", "endpoint exception register", "https://hospital-security.org.sa/exceptions", "confirm endpoint exception records", []),
            ("Eng. Mohammed Al-Salem, Network Services Manager", "hospital-network.org.sa", "Network Change Ticket Confirmation", "network change ticket", "https://hospital-network.org.sa/change", "acknowledge the network change ticket", []),
        ],
    },
}

_ROOT_ATTACK_LABELS = {
    "easy": ("Credential Harvesting", "سرقة بيانات الدخول"),
    "medium": ("Look-alike Domain Phishing", "تصيد بنطاق مشابه"),
    "hard": ("Advanced QR and Attachment Phishing", "تصيد متقدم برمز QR ومرفق"),
}

_ROOT_ATTACHMENTS = {
    "clinical": ["EMR_Protocol_Update_2026.pdf", "Medication_Reconciliation_Workflow.pdf", "PACS_Access_Validation.pdf"],
    "admin": ["Claims_Reconciliation_Worksheet.pdf", "Vendor_Renewal_Packet.pdf", "Clinic_Template_Governance.pdf"],
    "it": ["Privileged_Access_Review.pdf", "Endpoint_Exception_Register.pdf", "Network_Change_Ticket.pdf"],
    "other": ["Department_Procedure_Review.pdf", "Internal_Service_Update.pdf", "Staff_Acknowledgement_Form.pdf"],
}

_ROOT_LEGIT_BLUEPRINTS = {
    "clinical": ("Clinical Education Office <education@hospital.org>", "Updated Clinical Training Schedule", "Dear {name},\n\nThe clinical education schedule for next week has been published on the internal hospital portal. Please review your assigned session when convenient. No login details are required by email.\n\nRegards,\nClinical Education Office", ""),
    "admin": ("Patient Access Office <access.office@hospital.org>", "Appointment Template Update", "Dear {name},\n\nThe appointment template for next week has been updated in the internal scheduling system. Please review it through the official hospital portal during working hours.\n\nRegards,\nPatient Access Office", ""),
    "it": ("IT Change Management <change.management@hospital.org>", "Approved Maintenance Window", "Dear {name},\n\nThe approved maintenance window is scheduled for Friday evening. Details are available in the internal change calendar. This notice does not require password or MFA verification.\n\nRegards,\nIT Change Management", ""),
    "other": ("Staff Services <staff.services@hospital.org>", "Internal Staff Notice", "Dear {name},\n\nA staff notice has been posted on the official hospital intranet. Please review it through normal internal access when convenient.\n\nRegards,\nStaff Services", ""),
}

def _root_role_type(role):
    try:
        return ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))[2]
    except Exception:
        return "clinical"

def _root_normalize_difficulty(difficulty):
    d = (difficulty or "medium").lower()
    if d in ("beginner", "easy", "سهل"):
        return "easy"
    if d in ("advanced", "hard", "صعب"):
        return "hard"
    return "medium"

def _root_recipient(role_type, index):
    pool = _ROOT_RECIPIENTS.get(role_type) or _ROOT_RECIPIENTS["other"]
    return pool[index % len(pool)]

def _root_apply_errors(body, errors):
    # The errors are already embedded by replacing selected clean words.
    return body

def _root_link_before_signature(body, link):
    if not link:
        return body.strip()
    body = re.sub(r"\n\s*https?://\S+\s*$", "", body.strip())
    if link in body:
        return body.strip()
    m = _SIGNATURE_RE_ROOT.search("\n" + body)
    if m:
        pos = m.start() - 1
        return body[:pos].rstrip() + f"\n\nReview link: {link}\n" + body[pos:].lstrip()
    return body.rstrip() + f"\n\nReview link: {link}"

def _root_build_body_en(role_type, diff, name, task, action, link, errors, idx):
    first = name.split()[1] if name.startswith(("Dr.", "Nurse", "Pharmacist")) and len(name.split()) > 1 else name.split()[0]
    if diff == "easy":
        body = (f"Dear Staff,\n\nYour {task} account is at risk. You must {action} or your access will be suspended TODAY. "
                f"This action is requiered for all hospital staff.\n\nUpdate here: {link}\n\nThank You,\nSupport Team")
        if "immediatly" in errors and "immediately" not in body:
            body = body.replace("TODAY", "TODAY immediatly")
        if "acess" in errors:
            body = body.replace("access", "acess", 1)
        if "recieve" in errors:
            body = body.replace("review", "recieve", 1)
        if "urgant" in errors:
            body = body.replace("Urgent", "Urgant")
        return body
    if diff == "medium":
        body = (f"Dear {first},\n\nThe {task} requires a profile update for your department. Please {action}. "
                f"This helps maintain current hospital workflow records.\n\nOpen the secure review page: {link}\n\nRegards,\nDepartment Support Center")
        if errors:
            body = body.replace("access", errors[0], 1) if "access" in body else body.replace("requires", errors[0], 1)
        return body
    # hard
    attach = (_ROOT_ATTACHMENTS.get(role_type) or _ROOT_ATTACHMENTS["other"])[idx % 3]
    body = (f"Dear {name},\n\nAs part of the routine quarterly review for {task}, please review the attached file "
            f"({attach}) and confirm the item assigned to your unit at your earliest convenience.\n\n"
            f"Use the internal review button or scan this code from a hospital-managed device only: [QR: {task} review]\n\n"
            f"Reference: CHG-{20260 + idx}\n\nSincerely,\nClinical Governance Office")
    return body

def _root_build_body_ar(role_type, diff, name, task, action, link, errors, idx):
    if diff == "easy":
        return (f"عزيزي الموظف،\n\nحساب {task} معرض للإيقاف. يجب عليك {action} فورًا وإلا سيتم تعليق الوصول اليوم. "
                f"هذا الإجراء مطلووب لجميع موظفي المستشفى.\n\nرابط التحديث: {link}\n\nمع الشكر،\nفريق الدعم")
    if diff == "medium":
        return (f"عزيزي/عزيزتي {name.split()[0]}،\n\nيتطلب {task} تحديث ملف الوصول الخاص بالقسم. يرجى {action} خلال 48 ساعة. "
                f"هذا يساعد على استمرار سير العمل في المستشفى بشكل منظم.\n\nرابط المراجعة: {link}\n\nمع التحية،\nمركز دعم القسم")
    attach = (_ROOT_ATTACHMENTS.get(role_type) or _ROOT_ATTACHMENTS["other"])[idx % 3]
    return (f"عزيزي/عزيزتي {name}،\n\nضمن المراجعة الدورية المتعلقة بـ {task}، يرجى الاطلاع على المرفق الرسمي "
            f"({attach}) وتأكيد البند المخصص لوحدتكم عند التفرغ.\n\n"
            f"يرجى استخدام زر المراجعة الداخلي أو مسح الرمز من جهاز تابع للمستشفى فقط: [QR: مراجعة {task}]\n\n"
            f"المرجع: CHG-{20260 + idx}\n\nمع التحية،\nمكتب الحوكمة السريرية")

def _root_indicators(diff, link, is_ar=False):
    if is_ar:
        if diff == "easy":
            return [
                {"number": 1, "title": "تحية عامة", "description": "الرسالة تخاطب الموظفين بشكل عام دون اسم محدد."},
                {"number": 2, "title": "نطاق واضح التزوير", "description": f"الرابط يستخدم نطاقًا غير رسمي: {link}."},
                {"number": 3, "title": "طلب بيانات دخول مباشر", "description": "الرسالة تطلب كلمة المرور أو بيانات الدخول بشكل مباشر."},
                {"number": 4, "title": "إلحاح مبالغ فيه", "description": "التهديد بالإيقاف اليوم يضغط على المستخدم للتصرف بسرعة."},
            ]
        if diff == "medium":
            return [
                {"number": 1, "title": "نطاق مشابه", "description": "النطاق قريب من اسم المستشفى لكنه ليس نطاقًا رسميًا."},
                {"number": 2, "title": "طلب غير مباشر", "description": "الرسالة تطلب تحديث الوصول بدل طلب كلمة المرور صراحة."},
                {"number": 3, "title": "إلحاح متوسط", "description": "مهلة 48 ساعة أقل وضوحًا من التهديد الفوري."},
            ]
        return [
            {"number": 1, "title": "نطاق شبه رسمي", "description": "النطاق يبدو قريبًا من الرسمي ويحتاج تدقيقًا."},
            {"number": 2, "title": "رمز QR", "description": "رمز QR قد يخفي وجهة الرابط الحقيقية."},
            {"number": 3, "title": "مرفق رسمي مسمى", "description": "المرفق يعطي الرسالة مظهرًا مهنيًا مقنعًا."},
        ]
    if diff == "easy":
        return [
            {"number": 1, "title": "Generic greeting", "description": "The message uses a broad greeting instead of a named recipient."},
            {"number": 2, "title": "Obviously fake domain", "description": f"The URL uses a non-official domain: {link}."},
            {"number": 3, "title": "Direct credential request", "description": "The email directly asks for password or login details."},
            {"number": 4, "title": "Extreme urgency", "description": "Threatening same-day suspension pressures the user to act quickly."},
        ]
    if diff == "medium":
        return [
            {"number": 1, "title": "Look-alike domain", "description": "The domain looks related to the hospital but is not official."},
            {"number": 2, "title": "Indirect access request", "description": "The email asks for an update or confirmation rather than openly asking for a password."},
            {"number": 3, "title": "Moderate urgency", "description": "The 48-hour window creates pressure without an obvious threat."},
        ]
    return [
        {"number": 1, "title": "Near-official domain", "description": "The domain is polished and close to a legitimate hospital domain."},
        {"number": 2, "title": "QR code", "description": "The QR code can hide the true destination from the user."},
        {"number": 3, "title": "Named official attachment", "description": "The attachment makes the request appear part of a routine workflow."},
    ]

def _root_make_email(role, index, language, difficulty, is_phishing=True):
    role_type = _root_role_type(role)
    if role_type == "other":
        role_type = ["clinical", "admin", "it"][index % 3]
    diff = _root_normalize_difficulty(difficulty)
    is_ar = (language == "Arabic")
    name, to_email = _root_recipient(role_type, index)
    if not is_phishing:
        frm, subject, body_tpl, att = _ROOT_LEGIT_BLUEPRINTS.get(role_type, _ROOT_LEGIT_BLUEPRINTS["other"])
        body = body_tpl.format(name=name)
        return {"email_type": "Legitimate Email" if not is_ar else "رسالة شرعية", "attack_type": "None", "risk_level": "Safe", "from": frm, "to": to_email, "subject": subject, "attachment": att, "body": body, "suspicious_text": "", "suspicious_link": "", "is_phishing": False, "indicators": [], "why_risky": "This is a legitimate internal notice and does not request credentials." if not is_ar else "هذه رسالة داخلية شرعية ولا تطلب بيانات الدخول.", "learning_tip": "Use official portals for routine notices." if not is_ar else "استخدمي البوابات الرسمية للإشعارات الروتينية."}
    bp = (_ROOT_BLUEPRINTS.get(role_type) or _ROOT_BLUEPRINTS["clinical"])[diff][index % 3]
    sender_name, domain, subject, task, link, action, errors = bp
    frm = f"{sender_name} <updates@{domain}>"
    if is_ar:
        body = _root_build_body_ar(role_type, diff, name, task, action, link, errors, index)
        subject = {"easy": "إجراء عاجل مطلوب لحساب المستشفى", "medium": "تأكيد تحديث القسم خلال 48 ساعة", "hard": "إجراء روتيني لمراجعة الوصول"}[diff]
    else:
        body = _root_build_body_en(role_type, diff, name, task, action, link, errors, index)
    body = _root_link_before_signature(body, link if diff in ("easy", "medium") else "")
    attachment = "" if diff in ("easy", "medium") else (_ROOT_ATTACHMENTS.get(role_type) or _ROOT_ATTACHMENTS["other"])[index % 3]
    suspicious_text = action if not is_ar else ("طلب تأكيد/تحديث الوصول" if diff != "hard" else "رمز QR مع مرفق رسمي")
    attack_en, attack_ar = _ROOT_ATTACK_LABELS[diff]
    result = {
        "email_type": attack_ar if is_ar else attack_en,
        "attack_type": attack_ar if is_ar else attack_en,
        "risk_level": "High" if diff != "hard" else "Critical",
        "from": frm,
        "to": to_email,
        "subject": subject,
        "attachment": attachment,
        "body": body,
        "suspicious_text": suspicious_text,
        "suspicious_link": link if diff in ("easy", "medium") else "",
        "injected_errors": errors if not is_ar else (["مطلووب"] if diff == "easy" else ([] if diff == "hard" else ["تحديث"])),
        "is_phishing": True,
        "indicators": _root_indicators(diff, link, is_ar),
        "why_risky": ("This email is risky because it matches the selected healthcare role but uses phishing indicators calibrated to the chosen difficulty level." if not is_ar else "هذه الرسالة خطرة لأنها مرتبطة بالدور الصحي المختار لكنها تستخدم مؤشرات تصيد مناسبة لمستوى الصعوبة المحدد."),
        "learning_tip": ("Verify requests through official hospital systems, not links or QR codes inside email." if not is_ar else "تحققي من الطلبات عبر أنظمة المستشفى الرسمية وليس من روابط أو رموز QR داخل البريد."),
    }
    return result

def _root_guardrail_passes(result, role, difficulty, is_phishing=True):
    # Final deterministic sanity checks. If any check fails, regenerate from blueprint.
    if not isinstance(result, dict):
        return False
    role_type = _root_role_type(role)
    if role_type != "other":
        txt = " ".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link", "email_type"])
        if not _ROLE_KEYWORDS.get(role_type, re.compile("." )).search(txt):
            return False
        if _ROLE_FORBIDDEN.get(role_type, re.compile(r"a^" )).search(txt):
            return False
    diff = _root_normalize_difficulty(difficulty)
    body = str(result.get("body", ""))
    if _COMMERCIAL_THEME_RE.search(" ".join(str(result.get(k, "")) for k in ["from", "subject", "body"])):
        return False
    if diff in ("easy", "medium") and re.search(r"\[\s*QR", body, re.I):
        return False
    if diff == "easy" and str(result.get("attachment", "")).strip():
        return False
    if diff == "hard" and (not str(result.get("attachment", "")).strip() or not re.search(r"\[\s*QR", body, re.I)):
        return False
    link = str(result.get("suspicious_link", "")).strip()
    if link and _SIGNATURE_RE_ROOT.search(body):
        sig_pos = _SIGNATURE_RE_ROOT.search(body).start()
        if body.find(link) > sig_pos:
            return False
    return True

# Preserve AI generators for admin comparison/debug, but do not trust them as final trainee content.
_AI_GENERATE_EMAIL_BEFORE_ROOT = generate_email
_AI_GENERATE_ASSESS_EMAIL_BEFORE_ROOT = generate_assess_email
_AI_GENERATE_OTHER_EMAIL_BEFORE_ROOT = generate_other_email
_AI_GENERATE_OTHER_ASSESS_EMAIL_BEFORE_ROOT = generate_other_assess_email

def generate_email(role, index, language, difficulty="medium"):
    result = _root_make_email(role, index, language, difficulty, True)
    evaluate_and_log_auto_scores(result, _root_normalize_difficulty(difficulty), language, is_phishing=True)
    return result

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    result = _root_make_email(role, index, language, difficulty, bool(is_phishing))
    evaluate_and_log_auto_scores(result, _root_normalize_difficulty(difficulty), language, is_phishing=bool(is_phishing))
    return result

def generate_other_email(index, language, difficulty):
    # Other intentionally rotates through clinical/admin/IT to produce mixed but controlled hospital examples.
    pseudo_role = ["Clinical", "Admin / Management", "IT / Informatics"][index % 3]
    return generate_email(pseudo_role, index, language, difficulty)

def generate_other_assess_email(index, is_phishing, language, difficulty):
    pseudo_role = ["Clinical", "Admin / Management", "IT / Informatics"][index % 3]
    return generate_assess_email(pseudo_role, index, is_phishing, language, difficulty)

# =============================================================
# END ROOT-CONTROLLED GENERATION ENGINE
# =============================================================




# =============================================================
# ENHANCED HYBRID SCENARIO ENGINE v2
# -------------------------------------------------------------
# Deterministic scenario planner + strict difficulty guardrails.
# AI providers remain available for the research/admin comparison,
# but trainee-facing emails are now assembled from controlled
# healthcare scenarios so role, difficulty, language, and diversity
# cannot drift.
# =============================================================

import hashlib as _hashlib_v2

_ENHANCED_ROLE_SUBROLES = {
    "clinical": [
        {"key":"doctor","title_en":"Dr.","title_ar":"د.","people":[("Dr. Yousef Alghamdi","dr.yousef.alghamdi@hospital.org"),("Dr. Ahmed Alotaibi","dr.ahmed.alotaibi@hospital.org"),("Dr. Ziad Alharbi","dr.ziad.alharbi@hospital.org")]},
        {"key":"nurse","title_en":"Nurse","title_ar":"الممرضة", "people":[("Nurse Reem Alzahrani","n.reem.alzahrani@hospital.org"),("Nurse Maha Alsubaie","n.maha.alsubaie@hospital.org"),("Nurse Noura Alshamri","n.noura.alshamri@hospital.org")]},
        {"key":"pharmacist","title_en":"Pharm.","title_ar":"الصيدلي", "people":[("Pharm. Khalid Alqahtani","ph.khalid.alqahtani@hospital.org"),("Pharm. Ziad Alharbi","ph.ziad.alharbi@hospital.org"),("Pharm. Sara Almutairi","ph.sara.almutairi@hospital.org")]},
        {"key":"lab","title_en":"Lab Specialist","title_ar":"أخصائي المختبر", "people":[("Lab Specialist Omar Alharthy","lab.omar.alharthy@hospital.org"),("Lab Specialist Hessa Aldosari","lab.hessa.aldosari@hospital.org")]},
        {"key":"radiology","title_en":"Radiographer","title_ar":"أخصائي الأشعة", "people":[("Radiographer Maha Alsubaie","rad.maha.alsubaie@hospital.org"),("Radiographer Faisal Alzahrani","rad.faisal.alzahrani@hospital.org")]},
    ],
    "admin": [
        {"key":"secretary","title_en":"Medical Secretary","title_ar":"السكرتير الطبي", "people":[("Medical Secretary Reem Alsabiei","sec.reem.alsabiei@hospital.org"),("Medical Secretary Hind Alrashidi","sec.hind.alrashidi@hospital.org")]},
        {"key":"reception","title_en":"Reception Officer","title_ar":"موظف الاستقبال", "people":[("Reception Officer Abdullah Alqahtani","rec.abdullah.alqahtani@hospital.org"),("Reception Officer Dalal Alzahrani","rec.dalal.alzahrani@hospital.org")]},
        {"key":"insurance","title_en":"Insurance Coordinator","title_ar":"منسق التأمين", "people":[("Insurance Coordinator Sultan Alghamdi","ins.sultan.alghamdi@hospital.org"),("Insurance Coordinator Nora Alotaibi","ins.nora.alotaibi@hospital.org")]},
        {"key":"hr","title_en":"HR Officer","title_ar":"موظف الموارد البشرية", "people":[("HR Officer Omar Albaqami","hr.omar.albaqami@hospital.org"),("HR Officer Hessa Aldosari","hr.hessa.aldosari@hospital.org")]},
        {"key":"billing","title_en":"Billing Specialist","title_ar":"أخصائي الفوترة", "people":[("Billing Specialist Khalid Alharbi","bill.khalid.alharbi@hospital.org"),("Billing Specialist Sara Alqahtani","bill.sara.alqahtani@hospital.org")]},
    ],
    "it": [
        {"key":"helpdesk","title_en":"IT Support Engineer","title_ar":"مهندس الدعم الفني", "people":[("IT Support Engineer Mohammed Alshahri","it.mohammed.alshahri@hospital.org"),("IT Support Engineer Rania Almalki","it.rania.almalki@hospital.org")]},
        {"key":"cyber","title_en":"Cybersecurity Analyst","title_ar":"محلل الأمن السيبراني", "people":[("Cybersecurity Analyst Yusuf Aljuhani","cy.yusuf.aljuhani@hospital.org"),("Cybersecurity Analyst Lama Alumari","cy.lama.alumari@hospital.org")]},
        {"key":"network","title_en":"Network Engineer","title_ar":"مهندس الشبكات", "people":[("Network Engineer Bandar Althubaiti","net.bandar.althubaiti@hospital.org"),("Network Engineer Nadia Alsalmi","net.nadia.alsalmi@hospital.org")]},
        {"key":"systems","title_en":"Systems Administrator","title_ar":"مسؤول الأنظمة", "people":[("Systems Administrator Faisal Alotaibi","sys.faisal.alotaibi@hospital.org"),("Systems Administrator Muna Alharbi","sys.muna.alharbi@hospital.org")]},
    ],
}

_ENHANCED_SCENARIOS = {
    "clinical": [
        {"sub":"doctor","topic":"EMR medication reconciliation","sender":"Medical Affairs Office","dept":"Clinical Governance","task":"review the EMR medication reconciliation note","path":"emr-med-review","legit":"Routine EMR medication reconciliation reminder"},
        {"sub":"doctor","topic":"ICU consultation roster","sender":"ICU Coordination Desk","dept":"ICU Services","task":"confirm the ICU consultation roster entry","path":"icu-roster","legit":"ICU consultation roster update"},
        {"sub":"doctor","topic":"surgery list update","sender":"Operating Theatre Office","dept":"Surgery Scheduling","task":"review the next-day operating list entry","path":"ot-list","legit":"Operating theatre schedule notice"},
        {"sub":"nurse","topic":"shift handover note","sender":"Nursing Affairs","dept":"Nursing Services","task":"acknowledge the shift handover note","path":"handover-note","legit":"Nursing shift handover reminder"},
        {"sub":"nurse","topic":"infection-control checklist","sender":"Infection Control Unit","dept":"Infection Control","task":"confirm the infection-control checklist","path":"infection-check","legit":"Infection-control checklist update"},
        {"sub":"nurse","topic":"medication cart audit","sender":"Medication Safety Unit","dept":"Nursing Quality","task":"review the medication cart audit item","path":"cart-audit","legit":"Medication cart audit schedule"},
        {"sub":"pharmacist","topic":"controlled-drug inventory","sender":"Pharmacy Governance","dept":"Pharmacy Services","task":"confirm the controlled-drug inventory item","path":"drug-inventory","legit":"Controlled-drug inventory reminder"},
        {"sub":"pharmacist","topic":"drug recall notice","sender":"Medication Safety Office","dept":"Pharmacy Safety","task":"review the drug recall confirmation","path":"drug-recall","legit":"Drug recall acknowledgement"},
        {"sub":"lab","topic":"critical lab result queue","sender":"Laboratory Coordination Center","dept":"Laboratory Services","task":"review the pending critical lab result queue","path":"lab-result-check","legit":"Critical lab queue reminder"},
        {"sub":"radiology","topic":"PACS image review","sender":"Radiology Workflow Desk","dept":"Radiology Services","task":"confirm the PACS image review request","path":"pacs-review","legit":"PACS image review assignment"},
    ],
    "admin": [
        {"sub":"secretary","topic":"clinic schedule change","sender":"Outpatient Scheduling Office","dept":"Clinic Scheduling","task":"confirm the clinic schedule change","path":"clinic-schedule","legit":"Clinic schedule update"},
        {"sub":"reception","topic":"appointment queue review","sender":"Patient Access Center","dept":"Patient Access","task":"review the appointment queue update","path":"appointment-queue","legit":"Appointment queue notice"},
        {"sub":"insurance","topic":"insurance pre-approval batch","sender":"Insurance Coordination Unit","dept":"Insurance Office","task":"confirm the insurance pre-approval batch","path":"insurance-batch","legit":"Insurance approval batch reminder"},
        {"sub":"billing","topic":"patient billing correction","sender":"Revenue Cycle Office","dept":"Billing Services","task":"review the patient billing correction list","path":"billing-correction","legit":"Billing correction workflow"},
        {"sub":"hr","topic":"mandatory policy acknowledgement","sender":"Human Resources Center","dept":"Human Resources","task":"acknowledge the updated hospital policy","path":"policy-ack","legit":"HR policy acknowledgement"},
        {"sub":"secretary","topic":"consultant meeting minutes","sender":"Director Office","dept":"Executive Office","task":"review consultant meeting minutes","path":"meeting-minutes","legit":"Meeting minutes distribution"},
        {"sub":"reception","topic":"visitor badge process","sender":"Security Administration","dept":"Facility Security","task":"confirm the visitor badge process update","path":"visitor-badge","legit":"Visitor badge process update"},
        {"sub":"insurance","topic":"payer portal access","sender":"Claims Support Center","dept":"Claims Management","task":"update payer portal access details","path":"payer-portal","legit":"Claims portal scheduled maintenance"},
    ],
    "it": [
        {"sub":"helpdesk","topic":"service desk ticket queue","sender":"IT Service Management","dept":"IT Helpdesk","task":"review the overdue service desk ticket queue","path":"ticket-queue","legit":"Service desk queue reminder"},
        {"sub":"cyber","topic":"security incident triage","sender":"Security Operations Center","dept":"Cybersecurity","task":"confirm the security incident triage item","path":"soc-triage","legit":"SOC triage summary"},
        {"sub":"network","topic":"VPN certificate renewal","sender":"Network Operations","dept":"Infrastructure","task":"review the VPN certificate renewal record","path":"vpn-cert","legit":"VPN certificate maintenance"},
        {"sub":"systems","topic":"backup job exception","sender":"Systems Monitoring","dept":"Systems Administration","task":"confirm the backup job exception","path":"backup-exception","legit":"Backup exception report"},
        {"sub":"helpdesk","topic":"Microsoft 365 mailbox alert","sender":"Messaging Support Center","dept":"Collaboration Systems","task":"update the mailbox access record","path":"mailbox-alert","legit":"Mailbox storage notification"},
        {"sub":"network","topic":"firewall change request","sender":"Change Advisory Board","dept":"Network Security","task":"review the firewall change request","path":"firewall-change","legit":"Firewall change approval"},
        {"sub":"systems","topic":"server patch window","sender":"Server Operations","dept":"Data Center","task":"confirm the server patch window","path":"server-patch","legit":"Server patch window notice"},
    ],
}

_DIFFICULTY_RULES_V2 = {
    "easy": {"greeting":"generic","domain":"fake","qr":False,"attachment":False,"urgency":"extreme","length":"short","request":"direct"},
    "medium": {"greeting":"first","domain":"lookalike","qr":False,"attachment":"optional_pdf","urgency":"48h","length":"medium","request":"indirect"},
    "hard": {"greeting":"full_title","domain":"near_official","qr":True,"attachment":True,"urgency":"routine","length":"long","request":"verification"},
}

_ATTACK_VARIANTS_V2 = {
    "easy": ["Credential Harvesting", "Fake Account Warning", "Password Verification", "Urgent Access Suspension"],
    "medium": ["Cloud Document Scam", "Look-alike Portal", "Department Workflow Update", "Attachment Review Scam"],
    "hard": ["QR Phishing", "Authority Impersonation", "Official Attachment Scam", "Routine Procedure Impersonation"],
}

_SIGNATURES_V2 = {
    "clinical": ["Medical Affairs Office", "Clinical Governance Office", "Nursing Affairs", "Medication Safety Unit", "Laboratory Services", "Radiology Administration"],
    "admin": ["Patient Access Center", "Human Resources Center", "Revenue Cycle Office", "Insurance Coordination Unit", "Director Office"],
    "it": ["IT Service Management", "Security Operations Center", "Network Operations", "Systems Monitoring", "Change Advisory Board"],
}

_AR_SIGNATURES_V2 = {
    "clinical": ["مكتب الشؤون الطبية", "مكتب الحوكمة السريرية", "إدارة التمريض", "وحدة سلامة الدواء", "خدمات المختبر", "إدارة الأشعة"],
    "admin": ["مركز وصول المرضى", "مركز الموارد البشرية", "مكتب دورة الإيرادات", "وحدة تنسيق التأمين", "مكتب المدير"],
    "it": ["إدارة خدمات تقنية المعلومات", "مركز عمليات الأمن السيبراني", "عمليات الشبكات", "مراقبة الأنظمة", "لجنة التغيير"],
}

def _enhanced_role_type(role):
    return _root_role_type(role) if '_root_role_type' in globals() else (ROLE_MAP.get(role, ROLE_MAP.get('Clinical'))[2])

def _enhanced_diff(difficulty):
    return _root_normalize_difficulty(difficulty) if '_root_normalize_difficulty' in globals() else str(difficulty).lower()

def _enhanced_pick(role_type, index, assessment=False):
    if role_type == "other":
        role_type = ["clinical", "admin", "it"][(index + (1 if assessment else 0)) % 3]
    scenarios = _ENHANCED_SCENARIOS[role_type]
    sc = scenarios[index % len(scenarios)]
    subs = {s["key"]: s for s in _ENHANCED_ROLE_SUBROLES[role_type]}
    sub = subs.get(sc["sub"], _ENHANCED_ROLE_SUBROLES[role_type][0])
    person = sub["people"][(index // max(1, len(scenarios))) % len(sub["people"])]
    return role_type, sc, sub, person

def _first_name_v2(full):
    clean = re.sub(r"^(Dr\.|Nurse|Pharm\.|Lab Specialist|Radiographer|Medical Secretary|Reception Officer|Insurance Coordinator|HR Officer|Billing Specialist|IT Support Engineer|Cybersecurity Analyst|Network Engineer|Systems Administrator)\s+", "", full).strip()
    return clean.split()[0] if clean else full.split()[0]

def _domain_v2(diff, role_type, sc, is_legit=False):
    slug = sc["path"].replace("-", "")[:14]
    if is_legit:
        return "hospital.org"
    if diff == "easy":
        return f"fake-{slug}.com"
    if diff == "medium":
        return f"hospital-{slug}.net"
    return f"hospital-{slug}.org.sa"

def _link_v2(diff, role_type, sc):
    d = _domain_v2(diff, role_type, sc, False)
    if diff == "hard":
        return ""
    return f"http://{d}/{sc['path']}"

def _subject_v2(diff, sc, sub, is_ar=False, legit=False):
    if legit:
        return sc["legit"] if not is_ar else "إشعار داخلي روتيني"
    if is_ar:
        if diff == "easy": return "إجراء عاجل مطلوب لحساب المستشفى"
        if diff == "medium": return f"مراجعة {sc['dept']} خلال 48 ساعة"
        return f"إجراء روتيني: {sc['dept']}"
    if diff == "easy":
        return random.choice([f"URGENT: {sc['topic'].title()} Required", f"Immediate Action Required: {sc['topic'].title()}", f"Account Warning: {sc['topic'].title()}"])
    if diff == "medium":
        return random.choice([f"{sc['topic'].title()} Review", f"{sc['dept']} Update Required", f"Pending {sc['topic'].title()}"])
    return random.choice([f"Routine {sc['dept']} Review", f"{sc['topic'].title()} Confirmation", f"Quarterly {sc['dept']} Workflow Check"])

def _attachment_v2(diff, sc, role_type):
    if diff == "easy":
        return ""
    if diff == "medium":
        return f"{sc['path'].replace('-', '_')}_summary.pdf"
    return f"Official_{sc['path'].replace('-', '_')}_Protocol_2026.pdf"

def _build_body_v2(role_type, sc, sub, person, diff, is_ar=False, legit=False, index=0):
    name, email = person
    first = _first_name_v2(name)
    link = _link_v2(diff, role_type, sc)
    sig = (_AR_SIGNATURES_V2 if is_ar else _SIGNATURES_V2)[role_type][index % len(_SIGNATURES_V2[role_type])]
    attach = _attachment_v2(diff, sc, role_type)
    title_name = name
    if legit:
        if is_ar:
            return f"عزيزي/عزيزتي {first}،\n\nهذا إشعار داخلي بخصوص {sc['topic']} من قسم {sc['dept']}. لا يتطلب هذا الإشعار إدخال كلمة مرور أو فتح رابط خارجي.\n\nمع التحية،\n{sig}"
        return f"Dear {first},\n\nThis is a routine internal notice about {sc['topic']} from {sc['dept']}. No password, external link, or urgent account action is required.\n\nRegards,\n{sig}"
    if is_ar:
        if diff == "easy":
            return f"عزيزي الموظف،\n\nحسابك في المستشفى سيُغلق اليوم. يجب إدخال بيانات الدخول فورًا لمراجعة {sc['topic']}. توجد أخطاء في السجل مطلووب إصلاحها الآن.\n\nرابط مباشر: {link}\n\nمع الشكر،\nفريق الدعم"
        if diff == "medium":
            return f"عزيزي/عزيزتي {first}،\n\nيرجى تحديث بيانات الوصول المرتبطة بـ {sc['topic']} لقسم {sc['dept']} خلال 48 ساعة. هذا الإجراء يساعد على استمرار سير العمل دون تأخير.\n\nرابط المراجعة: {link}\n\nمع التحية،\n{sig}"
        return f"عزيزي/عزيزتي {title_name}،\n\nضمن الإجراء الروتيني لقسم {sc['dept']}، يرجى مراجعة المرفق الرسمي ({attach}) المتعلق بـ {sc['topic']} وتأكيد البند المخصص لوحدتكم عند التفرغ.\n\nزر المراجعة الداخلي متاح من بوابة المستشفى. يمكن أيضًا مسح الرمز من جهاز تابع للمستشفى فقط: [QR: {sc['path']}]\n\nالمرجع: HSP-{202600+index}\n\nمع التحية،\n{sig}"
    if diff == "easy":
        return f"Dear Staff,\n\nYour hospital account will close TODAY. You must enter your login password now to complete {sc['topic']}. This is requiered immediatly for all hospital users.\n\nUpdate here: {link}\n\nThank You,\nSupport Team"
    if diff == "medium":
        return f"Dear {first},\n\nThe {sc['topic']} item for {sc['dept']} requires a profile update within 48 hours. Please confirm the request through the review page below so the department workflow is not delayed.\n\nOpen review page: {link}\n\nRegards,\n{sig}"
    return f"Dear {title_name},\n\nAs part of the routine quarterly workflow review for {sc['dept']}, please review the attached document ({attach}) related to {sc['topic']} and confirm the item assigned to your unit at your earliest convenience.\n\nUse the internal review button from a hospital-managed device. For mobile verification, scan this code only inside the hospital network: [QR: {sc['path']}]\n\nReference: HSP-{202600+index}\n\nSincerely,\n{sig}"

def _indicators_v2(diff, sc, link, is_ar=False):
    attack = _ATTACK_VARIANTS_V2[diff][0]
    if is_ar:
        if diff == "easy":
            return [
                {"number":1,"title":"تحية عامة","description":"الرسالة لا تستخدم اسم الموظف أو لقبه الصحي."},
                {"number":2,"title":"نطاق واضح التزوير","description":f"الرابط ظاهر ويستخدم نطاقًا مزيفًا: {link}."},
                {"number":3,"title":"طلب مباشر لبيانات الدخول","description":"الرسالة تطلب إدخال كلمة المرور بشكل مباشر."},
                {"number":4,"title":"إلحاح مبالغ فيه","description":"التهديد بإغلاق الحساب اليوم مؤشر واضح للتصيد."},
            ]
        if diff == "medium":
            return [
                {"number":1,"title":"تحية باسم أول","description":"الرسالة تبدو أكثر تخصيصًا لكنها ليست كاملة."},
                {"number":2,"title":"نطاق مشابه للرسمي","description":"النطاق يشبه نطاق المستشفى لكنه ليس رسميًا."},
                {"number":3,"title":"طلب غير مباشر","description":"تطلب تحديث/تأكيد الوصول بدل طلب كلمة المرور صراحة."},
                {"number":4,"title":"مهلة 48 ساعة","description":"الإلحاح متوسط ومصمم ليبدو مقبولًا."},
            ]
        return [
            {"number":1,"title":"اسم كامل ولقب وظيفي","description":"الرسالة تستخدم هوية مهنية كاملة لزيادة المصداقية."},
            {"number":2,"title":"نطاق شبه رسمي","description":"النطاق قريب من الرسمي ويحتاج تدقيقًا."},
            {"number":3,"title":"QR إلزامي","description":"رمز QR قد يخفي الوجهة الحقيقية."},
            {"number":4,"title":"مرفق رسمي مسمى","description":"المرفق الرسمي يزيد صعوبة اكتشاف التصيد."},
        ]
    if diff == "easy":
        return [
            {"number":1,"title":"Generic greeting","description":"The message uses a broad greeting instead of the recipient's healthcare title."},
            {"number":2,"title":"Obviously fake domain","description":f"The visible link uses a clearly fake domain: {link}."},
            {"number":3,"title":"Direct credential request","description":"It directly asks for login/password details."},
            {"number":4,"title":"Extreme urgency","description":"Same-day closure pressure is a classic phishing sign."},
        ]
    if diff == "medium":
        return [
            {"number":1,"title":"First-name greeting","description":"The message is partly personalized but not fully official."},
            {"number":2,"title":"Look-alike domain","description":"The domain resembles a hospital domain but is not official."},
            {"number":3,"title":"Indirect request","description":"It asks for an update/confirmation rather than openly asking for a password."},
            {"number":4,"title":"Moderate urgency","description":"The 48-hour window creates pressure while staying believable."},
        ]
    return [
        {"number":1,"title":"Full title and name","description":"The recipient identity is role-specific and convincing."},
        {"number":2,"title":"Near-official domain","description":"The domain appears polished and close to official."},
        {"number":3,"title":"Mandatory QR code","description":"The QR code hides the destination and increases difficulty."},
        {"number":4,"title":"Named official attachment","description":"The formal attachment matches an advanced phishing pattern."},
    ]

def _make_email_v2(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    role_type0 = _enhanced_role_type(role)
    role_type, sc, sub, person = _enhanced_pick(role_type0, index, assessment)
    diff = _enhanced_diff(difficulty)
    is_ar = (language == "Arabic")
    name, email = person
    legit = not bool(is_phishing)
    domain = _domain_v2(diff, role_type, sc, legit)
    sender = sc["sender"] if not is_ar else (_AR_SIGNATURES_V2[role_type][index % len(_AR_SIGNATURES_V2[role_type])])
    frm = f"{sender} <updates@{domain}>"
    subject = _subject_v2(diff, sc, sub, is_ar, legit)
    body = _build_body_v2(role_type, sc, sub, person, diff, is_ar, legit, index)
    link = _link_v2(diff, role_type, sc)
    attachment = "" if legit else _attachment_v2(diff, sc, role_type)
    if diff == "easy": attachment = ""
    if legit: attachment = "" if index % 2 else f"{sc['path'].replace('-', '_')}_notice.pdf"
    attack = "Legitimate Email" if legit else _ATTACK_VARIANTS_V2[diff][index % len(_ATTACK_VARIANTS_V2[diff])]
    attack_ar = "رسالة شرعية" if legit else {"easy":"تصيد واضح لبيانات الدخول","medium":"تصيد متوسط عبر نطاق مشابه","hard":"تصيد متقدم عبر QR ومرفق رسمي"}[diff]
    return {
        "email_type": attack_ar if is_ar else attack,
        "attack_type": attack_ar if is_ar else attack,
        "risk_level": "Safe" if legit else ("Critical" if diff == "hard" else "High"),
        "from": frm,
        "to": email,
        "subject": subject,
        "attachment": attachment,
        "body": body,
        "suspicious_text": "" if legit else ({"easy":"enter your login password now", "medium":"profile update within 48 hours", "hard":"QR code and official attachment"}[diff] if not is_ar else {"easy":"إدخال بيانات الدخول فورًا", "medium":"تحديث بيانات الوصول خلال 48 ساعة", "hard":"رمز QR ومرفق رسمي"}[diff]),
        "suspicious_link": "" if (legit or diff == "hard") else link,
        "is_phishing": not legit,
        "scenario_id": f"{role_type}:{sub['key']}:{sc['path']}:{diff}:{index}",
        "subrole": sub["key"],
        "indicators": [] if legit else _indicators_v2(diff, sc, link, is_ar),
        "why_risky": ("This is a legitimate operational message with no credential request or external pressure." if legit and not is_ar else "هذه رسالة تشغيلية شرعية ولا تطلب بيانات دخول أو ضغط خارجي." if legit else "This email is risky because it matches the selected healthcare role, but the phishing indicators match the chosen difficulty level." if not is_ar else "هذه الرسالة خطرة لأنها مرتبطة بالدور الصحي المختار لكنها تستخدم مؤشرات تصيد مناسبة لمستوى الصعوبة."),
        "learning_tip": ("Verify unusual requests through official hospital systems." if not is_ar else "تحققي من الطلبات غير المعتادة عبر أنظمة المستشفى الرسمية."),
    }

def _validate_v2(result, role, difficulty, is_phishing=True):
    if not isinstance(result, dict): return False
    diff = _enhanced_diff(difficulty)
    body = str(result.get("body", ""))
    link = str(result.get("suspicious_link", ""))
    attachment = str(result.get("attachment", ""))
    txt = " ".join(str(result.get(k,"")) for k in ["from","to","subject","body","attachment","email_type"])
    if bool(is_phishing):
        if diff in ("easy","medium") and re.search(r"\[\s*QR", body, re.I): return False
        if diff == "easy" and attachment.strip(): return False
        if diff == "hard" and (not attachment.strip() or not re.search(r"\[\s*QR", body, re.I)): return False
        if diff != "hard" and not link.startswith("http://"): return False
        if re.search(r"(prize|reward|anniversary|lottery|celebration)", txt, re.I): return False
    rt = _enhanced_role_type(role)
    if rt != "other":
        # Prevent obvious role drift only; subrole variety is allowed inside each role.
        forbidden = {
            "clinical": r"\b(VPN|firewall|server|payroll|invoice|visitor badge|recruitment)\b",
            "admin": r"\b(PACS|ICU|medication cart|controlled-drug|firewall|VPN|server patch)\b",
            "it": r"\b(ICU|medication|patient billing|insurance pre-approval|clinic schedule)\b",
        }.get(rt)
        if forbidden and re.search(forbidden, txt, re.I): return False
    return True

# Override trainee-facing generation with the enhanced scenario engine.
def generate_email(role, index, language, difficulty="medium"):
    result = _make_email_v2(role, index, language, difficulty, True, assessment=False)
    if not _validate_v2(result, role, difficulty, True):
        result = _make_email_v2(role, index+7, language, difficulty, True, assessment=False)
    try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=True)
    except Exception: pass
    return result

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    result = _make_email_v2(role, index, language, difficulty, bool(is_phishing), assessment=True)
    if not _validate_v2(result, role, difficulty, bool(is_phishing)):
        result = _make_email_v2(role, index+11, language, difficulty, bool(is_phishing), assessment=True)
    try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
    except Exception: pass
    return result

def generate_other_email(index, language, difficulty):
    pseudo_role = ["Clinical", "Admin / Management", "IT / Informatics"][index % 3]
    return generate_email(pseudo_role, index, language, difficulty)

def generate_other_assess_email(index, is_phishing, language, difficulty):
    pseudo_role = ["Clinical", "Admin / Management", "IT / Informatics"][(index+1) % 3]
    return generate_assess_email(pseudo_role, index, is_phishing, language, difficulty)

# Make OpenAI the clean default whenever no provider has been explicitly saved.
try:
    if "ai_provider" not in st.session_state or st.session_state.get("ai_provider") not in _VALID_PROVIDERS:
        st.session_state["ai_provider"] = "openai"
except Exception:
    pass

# =============================================================
# END ENHANCED HYBRID SCENARIO ENGINE v2
# =============================================================



# =============================================================
# SCENARIO ENGINE v3 — Framework Contract + Diversity Planner
# -------------------------------------------------------------
# This layer fixes the v2 repetition problem. The AI/provider layer
# stays available for research comparison, but every visible email is
# planned through: role -> hidden subrole -> scenario card -> channel
# -> difficulty contract -> validation. This prevents role drift,
# QR/attachment misuse, and repeated body templates.
# =============================================================

_GENERIC_GREETINGS_V3 = {
    "English": ["Dear Hospital Staff", "Dear Healthcare Employee", "Dear Staff Member", "Attention Hospital User", "Dear Team Member", "Dear Employee"],
    "Arabic": ["عزيزي/عزيزتي موظف المستشفى", "عزيزي/عزيزتي عضو الفريق الصحي", "تنبيه لموظفي المستشفى", "عزيزي/عزيزتي الموظف"]
}

_SCENARIO_CARDS_V3 = {
    "clinical": [
        {"sub":"doctor","topic":"OPD clinic list","dept":"Outpatient Clinics","sender":"Clinic Scheduling Office","task":"review tomorrow's outpatient clinic list","path":"opd-list","channels":["link","button"],"attach":"OPD_List_Summary.pdf"},
        {"sub":"doctor","topic":"surgery list revision","dept":"Operating Theatre","sender":"Operating Theatre Office","task":"confirm the revised operating list","path":"ot-schedule","channels":["pdf","button"],"attach":"Theatre_List_Update.pdf"},
        {"sub":"doctor","topic":"MDT case discussion","dept":"Medical Affairs","sender":"MDT Coordination Desk","task":"review a pending MDT case note","path":"mdt-case","channels":["link","calendar"],"attach":"MDT_Case_Summary.pdf"},
        {"sub":"doctor","topic":"clinical privilege renewal","dept":"Clinical Governance","sender":"Credentialing Office","task":"confirm clinical privilege renewal details","path":"privilege-renewal","channels":["button","pdf"],"attach":"Privilege_Renewal_Form.pdf"},
        {"sub":"nurse","topic":"shift handover note","dept":"Nursing Affairs","sender":"Nursing Affairs","task":"acknowledge the shift handover note","path":"handover-note","channels":["pdf","link"],"attach":"Handover_Note.pdf"},
        {"sub":"nurse","topic":"infection-control checklist","dept":"Infection Control","sender":"Infection Control Unit","task":"complete the infection-control checklist","path":"infection-check","channels":["button","link"],"attach":"Infection_Checklist.pdf"},
        {"sub":"nurse","topic":"medication cart audit","dept":"Nursing Quality","sender":"Medication Safety Unit","task":"review the medication cart audit record","path":"cart-audit","channels":["pdf","button"],"attach":"Medication_Cart_Audit.pdf"},
        {"sub":"nurse","topic":"CPR competency renewal","dept":"Nursing Education","sender":"Clinical Education Center","task":"confirm CPR competency renewal status","path":"cpr-renewal","channels":["calendar","link"],"attach":"CPR_Renewal_Notice.pdf"},
        {"sub":"pharmacist","topic":"drug recall notice","dept":"Pharmacy Safety","sender":"Medication Safety Office","task":"review a recalled medication notice","path":"drug-recall","channels":["pdf","button"],"attach":"Drug_Recall_Notice.pdf"},
        {"sub":"pharmacist","topic":"controlled-drug inventory","dept":"Pharmacy Services","sender":"Pharmacy Governance","task":"confirm controlled-drug inventory records","path":"controlled-inventory","channels":["button","link"],"attach":"Controlled_Drug_Count.pdf"},
        {"sub":"pharmacist","topic":"antibiotic stewardship review","dept":"Clinical Pharmacy","sender":"Antimicrobial Stewardship Unit","task":"review the antibiotic stewardship queue","path":"antibiotic-review","channels":["pdf","link"],"attach":"Stewardship_Queue.pdf"},
        {"sub":"lab","topic":"critical lab result queue","dept":"Laboratory Services","sender":"Laboratory Coordination Center","task":"review pending critical lab result entries","path":"critical-lab","channels":["link","pdf"],"attach":"Critical_Result_List.pdf"},
        {"sub":"lab","topic":"specimen rejection report","dept":"Laboratory Quality","sender":"Lab Quality Office","task":"review specimen rejection records","path":"specimen-report","channels":["pdf","button"],"attach":"Specimen_Rejection_Report.pdf"},
        {"sub":"radiology","topic":"PACS image review","dept":"Radiology Services","sender":"Radiology Workflow Desk","task":"confirm a PACS image review assignment","path":"pacs-review","channels":["link","button"],"attach":"PACS_Assignment.pdf"},
        {"sub":"radiology","topic":"contrast safety checklist","dept":"Radiology Quality","sender":"Radiology Safety Office","task":"review the contrast safety checklist","path":"contrast-check","channels":["pdf","calendar"],"attach":"Contrast_Safety_Checklist.pdf"},
    ],
    "admin": [
        {"sub":"secretary","topic":"consultant clinic schedule","dept":"Clinic Administration","sender":"Outpatient Scheduling Office","task":"confirm consultant clinic changes","path":"consultant-schedule","channels":["link","calendar"],"attach":"Clinic_Schedule.pdf"},
        {"sub":"reception","topic":"appointment queue update","dept":"Patient Access","sender":"Patient Access Center","task":"review appointment queue changes","path":"appointment-queue","channels":["button","link"],"attach":"Appointment_Queue.pdf"},
        {"sub":"insurance","topic":"insurance pre-approval batch","dept":"Insurance Office","sender":"Insurance Coordination Unit","task":"confirm insurance pre-approval records","path":"insurance-batch","channels":["pdf","link"],"attach":"Preapproval_Batch.pdf"},
        {"sub":"billing","topic":"patient billing correction","dept":"Billing Services","sender":"Revenue Cycle Office","task":"review patient billing corrections","path":"billing-correction","channels":["pdf","button"],"attach":"Billing_Correction_List.pdf"},
        {"sub":"hr","topic":"annual leave balance","dept":"Human Resources","sender":"Human Resources Center","task":"review annual leave balance records","path":"leave-balance","channels":["link","button"],"attach":"Leave_Balance_Report.pdf"},
        {"sub":"hr","topic":"mandatory staff training","dept":"Training Department","sender":"Staff Training Center","task":"confirm mandatory training completion","path":"training-status","channels":["calendar","link"],"attach":"Training_Status.pdf"},
        {"sub":"secretary","topic":"director meeting minutes","dept":"Executive Office","sender":"Director Office","task":"review director meeting minutes","path":"meeting-minutes","channels":["pdf","button"],"attach":"Meeting_Minutes.pdf"},
        {"sub":"reception","topic":"visitor badge process","dept":"Facility Security","sender":"Security Administration","task":"review visitor badge process changes","path":"visitor-badge","channels":["link","pdf"],"attach":"Visitor_Badge_Process.pdf"},
        {"sub":"insurance","topic":"payer portal access","dept":"Claims Management","sender":"Claims Support Center","task":"confirm payer portal access records","path":"payer-access","channels":["button","link"],"attach":"Payer_Access.pdf"},
        {"sub":"billing","topic":"monthly revenue report","dept":"Finance Office","sender":"Finance Coordination Desk","task":"review the monthly revenue report","path":"revenue-report","channels":["pdf","button"],"attach":"Revenue_Report.pdf"},
    ],
    "it": [
        {"sub":"helpdesk","topic":"service desk ticket queue","dept":"IT Helpdesk","sender":"IT Service Management","task":"review overdue service desk tickets","path":"ticket-queue","channels":["link","button"],"attach":"Ticket_Queue.pdf"},
        {"sub":"cyber","topic":"security incident triage","dept":"Cybersecurity","sender":"Security Operations Center","task":"confirm a security incident triage item","path":"soc-triage","channels":["button","pdf"],"attach":"SOC_Triage_Report.pdf"},
        {"sub":"network","topic":"VPN certificate renewal","dept":"Infrastructure","sender":"Network Operations","task":"review VPN certificate renewal records","path":"vpn-cert","channels":["link","pdf"],"attach":"VPN_Certificate_Record.pdf"},
        {"sub":"systems","topic":"backup job exception","dept":"Systems Administration","sender":"Systems Monitoring","task":"confirm backup job exceptions","path":"backup-exception","channels":["pdf","button"],"attach":"Backup_Exception_Report.pdf"},
        {"sub":"helpdesk","topic":"Microsoft 365 mailbox alert","dept":"Collaboration Systems","sender":"Messaging Support Center","task":"review mailbox storage exceptions","path":"mailbox-alert","channels":["link","button"],"attach":"Mailbox_Notice.pdf"},
        {"sub":"network","topic":"firewall change request","dept":"Network Security","sender":"Change Advisory Board","task":"review firewall change requests","path":"firewall-change","channels":["pdf","calendar"],"attach":"Firewall_Change_Request.pdf"},
        {"sub":"systems","topic":"server patch window","dept":"Data Center","sender":"Server Operations","task":"confirm server patch window records","path":"server-patch","channels":["calendar","link"],"attach":"Patch_Window.pdf"},
        {"sub":"cyber","topic":"phishing simulation report","dept":"Cybersecurity Awareness","sender":"Security Awareness Office","task":"review phishing simulation results","path":"sim-report","channels":["pdf","button"],"attach":"Simulation_Report.pdf"},
        {"sub":"systems","topic":"HIS integration status","dept":"Clinical Systems","sender":"HIS Support Office","task":"review HIS integration status","path":"his-status","channels":["button","link"],"attach":"HIS_Status_Report.pdf"},
        {"sub":"network","topic":"Wi-Fi maintenance window","dept":"Network Operations","sender":"Wireless Network Team","task":"confirm Wi-Fi maintenance timing","path":"wifi-maintenance","channels":["calendar","link"],"attach":"WiFi_Maintenance.pdf"},
    ],
}

_SIGNATURE_BY_SCENARIO_V3 = {
    "clinical": ["Medical Affairs Office", "Nursing Affairs", "Pharmacy Governance", "Laboratory Services", "Radiology Workflow Desk", "Clinical Education Center"],
    "admin": ["Human Resources Center", "Patient Access Center", "Revenue Cycle Office", "Director Office", "Insurance Coordination Unit"],
    "it": ["IT Service Management", "Security Operations Center", "Network Operations", "Systems Monitoring", "Change Advisory Board"],
}

def _pick_v3(role, index, assessment=False):
    role_type0 = _enhanced_role_type(role)
    role_type = ["clinical", "admin", "it"][(index + (1 if assessment else 0)) % 3] if role_type0 == "other" else role_type0
    cards = _SCENARIO_CARDS_V3[role_type]
    # stride avoids adjacent repeated subroles/topics between learning and assessment
    sc = cards[((index * 5) + (3 if assessment else 0)) % len(cards)]
    subs = [s for s in _ENHANCED_ROLE_SUBROLES[role_type] if s["key"] == sc["sub"]]
    sub = subs[0] if subs else _ENHANCED_ROLE_SUBROLES[role_type][0]
    person = sub["people"][(index // 2) % len(sub["people"])]
    return role_type, sc, sub, person

def _medium_display_name_v3(person_name):
    # Intermediate should still feel like a healthcare workplace: title + first name only.
    patterns = [
        (r"^Dr\.\s+(\w+).*$", r"Dr. \1"),
        (r"^Nurse\s+(\w+).*$", r"Nurse \1"),
        (r"^Pharm\.\s+(\w+).*$", r"Pharm. \1"),
        (r"^Lab Specialist\s+(\w+).*$", r"Lab Specialist \1"),
        (r"^Radiographer\s+(\w+).*$", r"Radiographer \1"),
        (r"^IT Support Engineer\s+(\w+).*$", r"Eng. \1"),
        (r"^Network Engineer\s+(\w+).*$", r"Eng. \1"),
        (r"^Systems Administrator\s+(\w+).*$", r"Systems Admin \1"),
        (r"^Cybersecurity Analyst\s+(\w+).*$", r"Analyst \1"),
        (r"^(HR Officer|Medical Secretary|Reception Officer|Insurance Coordinator|Billing Specialist)\s+(\w+).*$", r"\1 \2"),
    ]
    for pat, repl in patterns:
        if re.search(pat, person_name):
            return re.sub(pat, repl, person_name)
    return person_name.split()[0]

def _channel_v3(diff, sc, index, legit=False):
    if diff == "easy":
        return "link"
    if diff == "hard":
        return "qr_official"
    channels = sc.get("channels") or ["link", "pdf", "button"]
    return channels[index % len(channels)]

def _domain_v3(diff, sc, legit=False):
    if legit:
        return "hospital.org"
    slug = re.sub(r"[^a-z0-9]", "", sc["path"].lower())[:16]
    if diff == "easy":
        return random.choice([f"hospital-alert-{slug}.com", f"secure-{slug}-login.info", f"fakehospital-{slug}.net"])
    if diff == "medium":
        return random.choice([f"hospital-{slug}.net", f"hospital-{slug}-support.org", f"{slug}-hospital-review.net"])
    return random.choice([f"hospital-{slug}.org.sa", f"hosp-{slug}.org.sa", f"hospital-services-{slug}.org.sa"])

def _link_v3(diff, sc, channel, legit=False):
    if legit:
        return ""
    if diff == "hard":
        return ""
    return f"http://{_domain_v3(diff, sc, False)}/{sc['path']}"

def _subject_v3(diff, sc, channel, lang, legit=False):
    is_ar = lang == "Arabic"
    if legit:
        return ("إشعار داخلي: " + sc["dept"]) if is_ar else random.choice([
            f"Internal Notice: {sc['topic'].title()}", f"Scheduled Update: {sc['dept']}", f"Routine {sc['dept']} Notice"
        ])
    if is_ar:
        if diff == "easy": return random.choice(["تحذير عاجل لحساب المستشفى", "إجراء فوري مطلوب", "سيتم إغلاق حسابك اليوم"])
        if diff == "medium": return random.choice([f"مراجعة {sc['dept']} خلال 48 ساعة", f"تحديث مطلوب: {sc['topic']}", f"إشعار متابعة من {sc['dept']}"])
        return random.choice([f"إجراء روتيني: {sc['topic']}", f"مراجعة رسمية لقسم {sc['dept']}", f"تأكيد مستند داخلي: {sc['dept']}"])
    if diff == "easy":
        return random.choice(["URGENT: Hospital Account Action", "Immediate Password Verification Required", "Account Closure Warning Today"])
    if diff == "medium":
        return random.choice([f"{sc['topic'].title()} Review", f"{sc['dept']} Follow-Up Required", f"Pending {sc['topic'].title()} Confirmation"])
    return random.choice([f"Routine {sc['topic'].title()} Confirmation", f"Official {sc['dept']} Workflow Review", f"Quarterly {sc['dept']} Document Check"])

def _attachment_v3(diff, sc, channel, legit=False):
    if diff == "easy":
        return ""
    if legit:
        return sc.get("attach", "Internal_Notice.pdf") if channel in ("pdf", "calendar") else ""
    if diff == "medium":
        return sc.get("attach", "Summary.pdf") if channel == "pdf" else ""
    return "Official_" + sc.get("attach", "Protocol_Update_2026.pdf").replace(" ", "_")

def _body_v3(role_type, sc, sub, person, diff, channel, lang, legit=False, index=0):
    is_ar = lang == "Arabic"
    name, _ = person
    first = _first_name_v2(name)
    medium_name = _medium_display_name_v3(name)
    full_name = name
    link = _link_v3(diff, sc, channel, legit)
    sig = sc.get("sender") or _SIGNATURE_BY_SCENARIO_V3[role_type][index % len(_SIGNATURE_BY_SCENARIO_V3[role_type])]
    attach = _attachment_v3(diff, sc, channel, legit)
    ref = f"HSP-{202600 + index * 17}"

    if legit:
        if is_ar:
            return f"عزيزي/عزيزتي {medium_name}،\n\nهذا إشعار داخلي مجدول من {sc['dept']} بخصوص {sc['topic']}. لا يتطلب إدخال كلمة مرور، ولا يوجد تهديد بإغلاق الحساب، ويمكن التحقق منه من خلال أنظمة المستشفى الرسمية.\n\nمع التحية،\n{sig}"
        return f"Dear {medium_name},\n\nThis is a scheduled internal notice from {sc['dept']} about {sc['topic']}. It does not ask for a password, does not threaten account closure, and can be verified through official hospital systems.\n\nRegards,\n{sig}"

    if diff == "easy":
        greet = random.choice(_GENERIC_GREETINGS_V3[lang])
        if is_ar:
            return f"{greet}،\n\nحسابك في المستشفى سيتوقف اليوم. يجب إدخال اسم المستخدم وكلمة المرور الآن حتى لا يتم إغلاق الوصول. يوجد خطأ واضع في {sc['topic']} ويجب تصحيحه فورًا.\n\nالرابط المكشوف: {link}\n\nشكراً،\nفريق الدعم"
        return random.choice([
            f"{greet},\n\nYour hospital account will be closed TODAY. Enter your username and password now to keep access. This is requiered immediatly for {sc['topic']}.\n\nUpdate here: {link}\n\nThank You,\nSupport Team",
            f"{greet},\n\nWe found a problem with your hospital access. Verfy your login password NOW or your account will stop today.\n\nLogin link: {link}\n\nRegards,\nHospital Alert Team",
            f"{greet},\n\nFinal warning: your staff access is not confirmed. Send your credential update through this page immediatly.\n\n{link}\n\nSupport Desk",
        ])

    if diff == "medium":
        if is_ar:
            if channel == "pdf":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى مراجعة المرفق البسيط الخاص بـ {sc['topic']} من قسم {sc['dept']}. نحتاج تأكيد الإجراء خلال 48 ساعة لتجنب تأخير سير العمل.\n\nالمرفق: {attach}\n\nمع التحية،\n{sig}"
            if channel == "button":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيوجد تحديث مرتبط بـ {sc['topic']} في قسم {sc['dept']}. يرجى استخدام زر المراجعة أدناه لتأكيد البيانات خلال 48 ساعة.\n\n[BUTTON: مراجعة الطلب]\n\nمع التحية،\n{sig}"
            if channel == "calendar":
                return f"عزيزي/عزيزتي {medium_name}،\n\nتمت إضافة متابعة مجدولة بخصوص {sc['topic']} لقسم {sc['dept']}. يرجى تأكيد الحضور أو تحديث الحالة خلال 48 ساعة.\n\nرابط المتابعة: {link}\n\nمع التحية،\n{sig}"
            return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى تحديث أو تأكيد البيانات المرتبطة بـ {sc['topic']} لقسم {sc['dept']} خلال 48 ساعة. هذا الإجراء يساعد على استمرار سير العمل دون تأخير.\n\nرابط المراجعة: {link}\n\nمع التحية،\n{sig}"
        if channel == "pdf":
            return f"Dear {medium_name},\n\nPlease review the attached summary for {sc['topic']} from {sc['dept']}. Confirmation is requested within 48 hours so the department workflow is not delayed.\n\nAttachment: {attach}\n\nRegards,\n{sig}"
        if channel == "button":
            return f"Dear {medium_name},\n\nA department-level update is pending for {sc['topic']} in {sc['dept']}. Please use the review button to confirm the item within 48 hours.\n\n[BUTTON: Review Request]\n\nRegards,\n{sig}"
        if channel == "calendar":
            return f"Dear {medium_name},\n\nA scheduled follow-up has been added for {sc['topic']} in {sc['dept']}. Please confirm or update your status within 48 hours so planning can continue.\n\nFollow-up page: {link}\n\nRegards,\n{sig}"
        return f"Dear {medium_name},\n\nPlease confirm the pending update related to {sc['topic']} for {sc['dept']} within 48 hours. This helps the team keep the workflow current without delaying patient services.\n\nReview page: {link}\n\nRegards,\n{sig}"

    # Advanced: all indicators mandatory: full title/name, near-official domain, official attachment, QR, no explicit urgency.
    if is_ar:
        return f"عزيزي/عزيزتي {full_name}،\n\nضمن المراجعة التشغيلية الروتينية لقسم {sc['dept']}، يرجى الاطلاع على المستند الرسمي ({attach}) المتعلق بـ {sc['topic']}. يرتبط هذا الإجراء بمهمة يومية في وحدتكم ولا يتطلب أي إجراء عاجل.\n\nيرجى استخدام زر المراجعة الداخلي من جهاز تابع للمستشفى، أو مسح رمز QR داخل شبكة المستشفى فقط: [QR: {sc['path']}]\n\nالمرجع: {ref}\nجهة الاتصال: {sig}\n\nمع التحية،\n{sig}"
    return f"Dear {full_name},\n\nAs part of the routine operational review for {sc['dept']}, please review the official document ({attach}) related to {sc['topic']}. This item is linked to your daily role workflow and does not require urgent action.\n\nUse the internal review button from a hospital-managed device, or scan the QR code inside the hospital network only: [QR: {sc['path']}]\n\nReference: {ref}\nContact unit: {sig}\n\nSincerely,\n{sig}"

def _analysis_v3(diff, sc, channel, link, attach, lang):
    is_ar = lang == "Arabic"
    if is_ar:
        if diff == "easy":
            return [
                {"number":1,"title":"تحية عامة","description":"الرسالة لا تستخدم اسمًا أو لقبًا وظيفيًا، وهذا يطابق المستوى السهل ويجعلها أقل إقناعًا."},
                {"number":2,"title":"نطاق مزيف واضح","description":f"الرابط مكشوف ويظهر نطاقًا غير رسمي: {link}. في المستوى السهل يجب أن يكون الرابط واضح التزوير."},
                {"number":3,"title":"طلب مباشر للبيانات","description":"الرسالة تطلب اسم المستخدم أو كلمة المرور مباشرة، وهذا مؤشر تصيد واضح."},
                {"number":4,"title":"إلحاح وتهديد","description":"التهديد بإغلاق الحساب اليوم يضغط على المستخدم للتصرف بسرعة دون تحقق."},
            ]
        if diff == "medium":
            tech = "مرفق PDF بسيط" if channel == "pdf" else "زر بسيط" if channel == "button" else "رابط مراجعة" if channel == "link" else "دعوة/متابعة مجدولة"
            return [
                {"number":1,"title":"تخصيص جزئي","description":"التحية تستخدم لقبًا مهنيًا واسمًا أول فقط، فتبدو أكثر واقعية من السهل لكنها ليست كاملة مثل الصعب."},
                {"number":2,"title":"نطاق مشابه للرسمي","description":"النطاق يشبه نطاق المستشفى لكنه ليس نطاقًا رسميًا مؤكدًا."},
                {"number":3,"title":"طلب غير مباشر","description":"الرسالة تطلب تحديثًا أو تأكيدًا بدل طلب كلمة المرور صراحة، وهذا يناسب المستوى المتوسط."},
                {"number":4,"title":tech,"description":"العنصر التقني متوسط التعقيد ولا يحتوي QR، لأن QR ممنوع في السهل والمتوسط."},
                {"number":5,"title":"مهلة 48 ساعة","description":"الإلحاح موجود لكنه ليس تهديدًا مباشرًا، لذلك يحتاج المستخدم إلى انتباه أكبر."},
            ]
        return [
            {"number":1,"title":"اسم كامل ولقب وظيفي","description":"الرسالة تستخدم هوية مهنية كاملة، وهذا يزيد المصداقية ويطابق المستوى الصعب."},
            {"number":2,"title":"سياق مهمة يومية","description":f"المحتوى مرتبط بـ {sc['topic']} داخل {sc['dept']}، وليس طلبًا عامًا."},
            {"number":3,"title":"مرفق رسمي مسمى","description":f"المرفق ({attach}) يبدو رسميًا ومحددًا، وهذا مؤشر إلزامي في المستوى الصعب."},
            {"number":4,"title":"رمز QR إلزامي","description":"الـQR يخفي الوجهة الحقيقية، وهو المؤشر الحصري للمستوى الصعب في الإطار."},
            {"number":5,"title":"لا يوجد تهديد مباشر","description":"الصياغة روتينية واحترافية، لذلك يصعب اكتشافها مقارنة بالسهل والمتوسط."},
        ]
    if diff == "easy":
        return [
            {"number":1,"title":"Generic greeting","description":"The email uses a broad staff greeting rather than a healthcare title or name, matching the Easy level."},
            {"number":2,"title":"Obviously fake visible URL","description":f"The full link is visible and clearly suspicious: {link}."},
            {"number":3,"title":"Direct credential request","description":"It directly asks for username/password action, which is a clear phishing indicator."},
            {"number":4,"title":"Extreme urgency","description":"Same-day account closure pressure is designed to make the user act without checking."},
        ]
    if diff == "medium":
        tech = {"pdf":"simple PDF attachment", "button":"simple review button", "link":"look-alike review link", "calendar":"scheduled follow-up request"}.get(channel, "technical element")
        return [
            {"number":1,"title":"Partial professional personalization","description":"The email uses a healthcare/work title and first name, making it more believable than Easy but not as specific as Advanced."},
            {"number":2,"title":"Look-alike domain","description":"The sender domain resembles a hospital service but is not an official hospital domain."},
            {"number":3,"title":"Indirect request","description":"It asks for an update or confirmation rather than openly asking for a password."},
            {"number":4,"title":tech.title(),"description":"The technical element is moderate and contains no QR code, keeping it within the Intermediate rules."},
            {"number":5,"title":"Moderate 48-hour urgency","description":"The deadline creates pressure without the obvious threat seen in Easy emails."},
        ]
    return [
        {"number":1,"title":"Full title and name","description":"The recipient is addressed with a full professional identity, increasing credibility."},
        {"number":2,"title":"Daily role-specific context","description":f"The request is tied to {sc['topic']} in {sc['dept']}, which fits the recipient's healthcare role."},
        {"number":3,"title":"Named official attachment","description":f"The attachment ({attach}) appears formal and specific, which is required for Advanced."},
        {"number":4,"title":"Mandatory QR code","description":"The QR code can hide the destination and is the exclusive marker of the Advanced level."},
        {"number":5,"title":"Professional, low-pressure tone","description":"There is no explicit threat; the routine wording makes the email harder to detect."},
    ]

def _make_email_v3(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    diff = _enhanced_diff(difficulty)
    role_type, sc, sub, person = _pick_v3(role, index, assessment)
    legit = not bool(is_phishing)
    channel = _channel_v3(diff, sc, index, legit)
    domain = _domain_v3(diff, sc, legit)
    sender = sc.get("sender") or _SIGNATURE_BY_SCENARIO_V3[role_type][index % len(_SIGNATURE_BY_SCENARIO_V3[role_type])]
    frm = f"{sender} <updates@{domain}>"
    subject = _subject_v3(diff, sc, channel, language, legit)
    body = _body_v3(role_type, sc, sub, person, diff, channel, language, legit, index)
    link = _link_v3(diff, sc, channel, legit)
    attach = _attachment_v3(diff, sc, channel, legit)
    is_ar = language == "Arabic"
    attack_map = {
        "easy": "Credential Harvesting / Obvious Phishing",
        "medium": f"Look-alike Domain / {channel.title()} Phishing",
        "hard": "Advanced QR Phishing / Official Attachment",
    }
    attack_ar = {"easy":"تصيد واضح لبيانات الدخول", "medium":"تصيد متوسط بنطاق مشابه", "hard":"تصيد متقدم عبر QR ومرفق رسمي"}
    suspicious_text = "" if legit else ({"easy":"enter your username and password now", "medium":"confirm/update within 48 hours", "hard":"QR code and official attachment"}[diff] if not is_ar else {"easy":"إدخال اسم المستخدم وكلمة المرور الآن", "medium":"تحديث أو تأكيد خلال 48 ساعة", "hard":"رمز QR ومرفق رسمي"}[diff])
    why = "This email is risky because its indicators match the selected difficulty level while staying aligned with the healthcare role and scenario." if not is_ar else "هذه الرسالة خطرة لأنها تستخدم مؤشرات تصيد متوافقة مع مستوى الصعوبة ومتصلة بالدور الصحي المختار."
    if legit:
        why = "This message is legitimate because it avoids credential requests, external pressure, and suspicious urgency." if not is_ar else "هذه الرسالة شرعية لأنها لا تطلب بيانات دخول ولا تستخدم ضغطًا أو تهديدًا مشبوهًا."
    return {
        "email_type": ("رسالة شرعية" if legit and is_ar else "Legitimate Email" if legit else attack_ar[diff] if is_ar else attack_map[diff]),
        "attack_type": ("رسالة شرعية" if legit and is_ar else "Legitimate Email" if legit else attack_ar[diff] if is_ar else attack_map[diff]),
        "risk_level": "Safe" if legit else ("Critical" if diff == "hard" else "High" if diff == "medium" else "Medium"),
        "from": frm,
        "to": person[1],
        "subject": subject,
        "attachment": attach,
        "body": body,
        "suspicious_text": suspicious_text,
        "suspicious_link": "" if (legit or diff == "hard" or channel in ("pdf", "button")) else link,
        "is_phishing": not legit,
        "scenario_id": f"v3:{role_type}:{sc['sub']}:{sc['path']}:{diff}:{channel}:{index}",
        "subrole": sc["sub"],
        "indicators": [] if legit else _analysis_v3(diff, sc, channel, link, attach, language),
        "why_risky": why,
        "learning_tip": ("Check the sender domain, request type, and technical element before acting; QR codes appear only in advanced examples." if not is_ar else "تحقق من نطاق المرسل ونوع الطلب والعنصر التقني قبل التصرف؛ رموز QR تظهر فقط في أمثلة المستوى الصعب."),
    }

def _validate_v3(result, role, difficulty, is_phishing=True):
    if not isinstance(result, dict): return False
    diff = _enhanced_diff(difficulty)
    txt = "\n".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    attachment = str(result.get("attachment", ""))
    if is_phishing:
        if diff == "easy":
            if attachment or "[QR" in txt or "[BUTTON" in txt: return False
            if not re.search(r"http://", txt): return False
        if diff == "medium":
            if "[QR" in txt: return False
            # Exactly one technical path: link OR button OR simple attachment/calendar PDF.
            tech_count = int("http://" in txt) + int("[BUTTON" in txt) + int(bool(attachment))
            if tech_count != 1: return False
        if diff == "hard":
            if not attachment or "[QR" not in txt: return False
            if re.search(r"account will|closed TODAY|immediate password|Act NOW", txt, re.I): return False
    return True

# Final v3 overrides used by trainee-facing pages.
def generate_email(role, index, language, difficulty="medium"):
    for offset in (0, 7, 13):
        result = _make_email_v3(role, index + offset, language, difficulty, True, assessment=False)
        if _validate_v3(result, role, difficulty, True):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=True)
            except Exception: pass
            return result
    return result

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    for offset in (0, 9, 17):
        result = _make_email_v3(role, index + offset, language, difficulty, bool(is_phishing), assessment=True)
        if _validate_v3(result, role, difficulty, bool(is_phishing)):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
            except Exception: pass
            return result
    return result

def generate_other_email(index, language, difficulty):
    return generate_email("Other", index, language, difficulty)

def generate_other_assess_email(index, is_phishing, language, difficulty):
    return generate_assess_email("Other", index, is_phishing, language, difficulty)

# Default provider correction: OpenAI unless researcher manually changes it.
try:
    if not os.path.exists(_PROVIDER_CONFIG_PATH):
        st.session_state["ai_provider"] = "openai"
except Exception:
    pass

# =============================================================
# END SCENARIO ENGINE v3
# =============================================================


# =============================================================
# SCENARIO ENGINE v4 — Strict Framework Contract Fix
# -------------------------------------------------------------
# Fixes observed v3 issues:
# 1) Beginner/Easy can no longer display Intermediate indicators.
# 2) Link is generated ONCE and reused in body + suspicious_link, so
#    it cannot appear twice or drift to the signature area.
# 3) Button channel renders as a real markdown button for the renderer.
# 4) Analysis is generated from the exact email properties.
# =============================================================

def _stable_domain_v4(diff, sc, channel="link", legit=False):
    if legit:
        return "hospital.org"
    slug = re.sub(r"[^a-z0-9]", "", sc.get("path", "hospital").lower())[:18] or "portal"
    if diff == "easy":
        # Clearly fake, obvious, and not near-official.
        variants = [f"fake-{slug}-login.com", f"hospital-alert-{slug}.com", f"secure-{slug}-verify.info"]
    elif diff == "medium":
        # Look-alike, plausible but not official.
        variants = [f"hospital-{slug}-support.org", f"hospital-{slug}.net", f"{slug}-hospital-review.net"]
    else:
        # Near-official; used only behind QR/button, not shown as plain URL.
        variants = [f"hospital-{slug}.org.sa", f"hosp-{slug}.org.sa", f"hospital-services-{slug}.org.sa"]
    # Deterministic selection for a given scenario/channel to avoid body/link mismatch.
    return variants[(len(slug) + len(channel)) % len(variants)]

def _link_v4(diff, sc, channel, legit=False):
    if legit or diff == "hard":
        return ""
    return f"http://{_stable_domain_v4(diff, sc, channel, False)}/{sc['path']}"

def _display_name_v4(name, diff):
    if diff == "easy":
        return ""
    if diff == "hard":
        return name
    return _medium_display_name_v3(name)

def _subject_v4(diff, sc, channel, lang, legit=False):
    is_ar = lang == "Arabic"
    if legit:
        return (f"إشعار داخلي: {sc['dept']}" if is_ar else random.choice([
            f"Internal Notice: {sc['topic'].title()}",
            f"Scheduled Update: {sc['dept']}",
            f"Routine {sc['dept']} Notice",
        ]))
    if is_ar:
        if diff == "easy":
            return random.choice(["إجراء فوري مطلوب لحساب المستشفى", "تحذير: سيتم إغلاق الحساب اليوم", "تأكيد كلمة المرور فورًا"])
        if diff == "medium":
            return random.choice([f"مراجعة {sc['dept']} خلال 48 ساعة", f"تحديث مطلوب: {sc['topic']}", f"متابعة {sc['dept']} مطلوبة"])
        return random.choice([f"إجراء روتيني: {sc['topic']}", f"مراجعة رسمية لقسم {sc['dept']}", f"تأكيد مستند داخلي: {sc['dept']}"])
    if diff == "easy":
        return random.choice(["URGENT: Staff Account Will Close Today", "Immediate Password Verification Required", "Final Warning: Hospital Login Access"])
    if diff == "medium":
        return random.choice([f"{sc['topic'].title()} Review", f"{sc['dept']} Follow-Up Required", f"Pending {sc['topic'].title()} Confirmation"])
    return random.choice([f"Routine {sc['topic'].title()} Confirmation", f"Official {sc['dept']} Workflow Review", f"Quarterly {sc['dept']} Document Check"])

def _channel_v4(diff, sc, index, legit=False):
    if diff == "easy":
        return "link"
    if diff == "hard":
        return "qr_official"
    channels = sc.get("channels") or ["link", "pdf", "button", "calendar"]
    # Keep exactly one vector for Intermediate.
    return channels[index % len(channels)]

def _attachment_v4(diff, sc, channel, legit=False):
    if diff == "easy":
        return ""
    if legit:
        return sc.get("attach", "Internal_Notice.pdf") if channel in ("pdf", "calendar") else ""
    if diff == "medium":
        return sc.get("attach", "Summary.pdf") if channel == "pdf" else ""
    return "Official_" + sc.get("attach", "Protocol_Update_2026.pdf").replace(" ", "_")

def _body_v4(role_type, sc, sub, person, diff, channel, lang, legit=False, index=0, link="", attach=""):
    is_ar = lang == "Arabic"
    name, _ = person
    sig = sc.get("sender") or _SIGNATURE_BY_SCENARIO_V3[role_type][index % len(_SIGNATURE_BY_SCENARIO_V3[role_type])]
    medium_name = _medium_display_name_v3(name)
    full_name = name
    ref = f"HSP-{202600 + index * 19}"

    if legit:
        if is_ar:
            return f"عزيزي/عزيزتي {medium_name}،\n\nهذا إشعار داخلي مجدول من {sc['dept']} بخصوص {sc['topic']}. لا يطلب كلمة مرور، ولا يتضمن تهديدًا بإغلاق الحساب، ويمكن التحقق منه عبر أنظمة المستشفى الرسمية.\n\nمع التحية،\n{sig}"
        return f"Dear {medium_name},\n\nThis is a scheduled internal notice from {sc['dept']} about {sc['topic']}. It does not ask for a password, does not threaten account closure, and can be verified through official hospital systems.\n\nRegards,\n{sig}"

    if diff == "easy":
        greet = random.choice(_GENERIC_GREETINGS_V3[lang])
        if is_ar:
            return f"{greet}،\n\nسيتم إغلاق حساب المستشفى اليوم. أدخل اسم المستخدم وكلمة المرور الآن حتى لا يتوقف الوصول. هذا الإجراء مطلووب فورًا بسبب خطأ في نظام المستشفى.\n\nرابط الدخول: {link}\n\nشكراً،\nفريق الدعم"
        # Easy must stay generic, obvious, short, link-only, and contain visible errors.
        templates = [
            f"{greet},\n\nYour hospital account will be closed TODAY. Enter your username and password now to keep access. This is requiered immediatly for staff access.\n\nLogin page: {link}\n\nThank You,\nSupport Team",
            f"{greet},\n\nFinal warning: your hospital login is not confirmed. Verfy your password NOW or your access will stop today.\n\nUpdate link: {link}\n\nRegards,\nHospital Alert Team",
            f"{greet},\n\nWe found a problm with your staff account. Send your credential update through this page today or the account will close.\n\nSecure page: {link}\n\nSupport Desk",
        ]
        return templates[index % len(templates)]

    if diff == "medium":
        if is_ar:
            if channel == "pdf":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى مراجعة المرفق البسيط الخاص بـ {sc['topic']} من قسم {sc['dept']}. نحتاج تأكيد الإجراء خلال 48 ساعة حتى لا يتأخر سير العمل.\n\nمع التحية،\n{sig}"
            if channel == "button":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيوجد تحديث مرتبط بـ {sc['topic']} في قسم {sc['dept']}. يرجى استخدام زر المراجعة لتأكيد الطلب خلال 48 ساعة.\n\n[مراجعة الطلب]({link})\n\nمع التحية،\n{sig}"
            if channel == "calendar":
                return f"عزيزي/عزيزتي {medium_name}،\n\nتمت إضافة متابعة مجدولة بخصوص {sc['topic']} لقسم {sc['dept']}. يرجى تأكيد الحالة خلال 48 ساعة.\n\nرابط المتابعة: {link}\n\nمع التحية،\n{sig}"
            return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى تحديث أو تأكيد البيانات المرتبطة بـ {sc['topic']} لقسم {sc['dept']} خلال 48 ساعة. هذا الإجراء يساعد على استمرار سير العمل.\n\nرابط المراجعة: {link}\n\nمع التحية،\n{sig}"
        if channel == "pdf":
            return f"Dear {medium_name},\n\nPlease review the attached summary for {sc['topic']} from {sc['dept']}. Confirmation is requested within 48 hours so the department workflow is not delayed.\n\nRegards,\n{sig}"
        if channel == "button":
            return f"Dear {medium_name},\n\nA department-level update is pending for {sc['topic']} in {sc['dept']}. Please use the review button to confirm the item within 48 hours.\n\n[Review Request]({link})\n\nRegards,\n{sig}"
        if channel == "calendar":
            return f"Dear {medium_name},\n\nA scheduled follow-up has been added for {sc['topic']} in {sc['dept']}. Please confirm or update your status within 48 hours so planning can continue.\n\nFollow-up page: {link}\n\nRegards,\n{sig}"
        return f"Dear {medium_name},\n\nPlease confirm the pending update related to {sc['topic']} for {sc['dept']} within 48 hours. This helps the team keep the workflow current without delaying patient services.\n\nReview page: {link}\n\nRegards,\n{sig}"

    # Advanced: all mandatory indicators from the framework.
    if is_ar:
        return f"عزيزي/عزيزتي {full_name}،\n\nضمن المراجعة التشغيلية الروتينية لقسم {sc['dept']}، يرجى الاطلاع على المستند الرسمي ({attach}) المتعلق بـ {sc['topic']}. يرتبط هذا الإجراء بمهمة يومية في وحدتكم ولا يتطلب إجراءً عاجلًا.\n\nيرجى استخدام زر المراجعة الداخلي من جهاز تابع للمستشفى، أو مسح رمز QR داخل شبكة المستشفى فقط: [QR: {sc['path']}]\n\nالمرجع: {ref}\nجهة الاتصال: {sig}\n\nمع التحية،\n{sig}"
    return f"Dear {full_name},\n\nAs part of the routine operational review for {sc['dept']}, please review the official document ({attach}) related to {sc['topic']}. This item is linked to your daily role workflow and does not require urgent action.\n\nUse the internal review button from a hospital-managed device, or scan the QR code inside the hospital network only: [QR: {sc['path']}]\n\nReference: {ref}\nContact unit: {sig}\n\nSincerely,\n{sig}"

def _analysis_v4(diff, sc, channel, link, attach, lang, body=""):
    is_ar = lang == "Arabic"
    # Analysis is tied to what was actually rendered: link/pdf/button/QR + scenario + difficulty.
    has_pdf = bool(attach)
    has_button = "[" in body and "](" in body
    has_qr = "[QR" in body
    has_link = bool(link) and link in body
    if is_ar:
        if diff == "easy":
            return [
                {"number":1,"title":"تحية عامة","description":"الرسالة تخاطب مجموعة عامة من الموظفين ولا تستخدم اسمًا أو لقبًا مهنيًا، وهذا يطابق المستوى السهل."},
                {"number":2,"title":"رابط مكشوف وواضح التزوير","description":f"الرابط ظاهر داخل النص ويستخدم نطاقًا غير موثوق: {link}."},
                {"number":3,"title":"طلب مباشر لكلمة المرور","description":"النص يطلب إدخال اسم المستخدم وكلمة المرور مباشرة، وهذا مؤشر تصيد واضح."},
                {"number":4,"title":"إلحاح شديد","description":"التهديد بإغلاق الحساب اليوم يضغط على المستخدم للتصرف بسرعة."},
                {"number":5,"title":"لا يوجد مرفق أو QR","description":"هذا مناسب للمستوى السهل؛ الخطر هنا ظاهر من الرابط والطلب المباشر وليس من عناصر متقدمة."},
            ]
        if diff == "medium":
            tech_title = "مرفق PDF بسيط" if has_pdf else "زر مراجعة بسيط" if has_button else "رابط مشابه للرسمي" if has_link else "عنصر متابعة بسيط"
            tech_desc = f"العنصر المستخدم هو {tech_title}، ولا يوجد QR لأن QR محصور بالمستوى الصعب."
            return [
                {"number":1,"title":"تخصيص مهني جزئي","description":"التحية تستخدم لقبًا صحيًا أو وظيفيًا مع الاسم الأول فقط، فتبدو واقعية دون أن تصل لتفصيل المستوى الصعب."},
                {"number":2,"title":"سياق قسم محدد","description":f"الموضوع مرتبط بـ {sc['topic']} داخل {sc['dept']}، وهذا يجعله مناسبًا للدور الصحي."},
                {"number":3,"title":"طلب غير مباشر","description":"الرسالة تطلب مراجعة أو تأكيدًا خلال سير العمل بدل طلب كلمة المرور صراحة."},
                {"number":4,"title":tech_title,"description":tech_desc},
                {"number":5,"title":"مهلة 48 ساعة","description":"الإلحاح متوسط ومهني، وليس تهديدًا واضحًا مثل المستوى السهل."},
            ]
        return [
            {"number":1,"title":"اسم كامل ولقب وظيفي","description":"الرسالة تستخدم هوية مهنية كاملة مما يزيد المصداقية."},
            {"number":2,"title":"مهمة يومية محددة","description":f"المحتوى مرتبط بـ {sc['topic']} داخل {sc['dept']} وليس طلبًا عامًا."},
            {"number":3,"title":"مرفق رسمي مسمى","description":f"المرفق ({attach}) يبدو رسميًا ومحددًا."},
            {"number":4,"title":"رمز QR إلزامي","description":"رمز QR يخفي الوجهة الحقيقية وهو مؤشر المستوى الصعب حسب الإطار."},
            {"number":5,"title":"نبرة روتينية بلا تهديد","description":"الرسالة لا تستخدم تهديدًا مباشرًا، وهذا يجعلها أصعب في الاكتشاف."},
        ]
    if diff == "easy":
        return [
            {"number":1,"title":"Generic greeting","description":"The email addresses staff broadly instead of using a healthcare title or personal name, which matches the Easy level."},
            {"number":2,"title":"Clearly fake visible URL","description":f"The link is shown directly in the message and uses an obviously suspicious domain: {link}."},
            {"number":3,"title":"Direct credential request","description":"The body asks the user to enter or verify username/password details directly."},
            {"number":4,"title":"Extreme urgency and threat","description":"The same-day account closure warning is designed to pressure fast action."},
            {"number":5,"title":"No attachment or QR","description":"This stays within the Easy framework: the risk is obvious through the visible link and direct request, not an advanced technical element."},
        ]
    if diff == "medium":
        tech_title = "Simple PDF attachment" if has_pdf else "Simple review button" if has_button else "Look-alike review link" if has_link else "Simple workflow element"
        tech_desc = "The email uses exactly one moderate technical element and contains no QR code, keeping it within the Intermediate rules."
        return [
            {"number":1,"title":"Partial professional personalization","description":"The greeting uses a healthcare/work title and first name only, which is more believable than Easy but not as specific as Advanced."},
            {"number":2,"title":"Department-level healthcare context","description":f"The request is connected to {sc['topic']} in {sc['dept']}, so it fits the selected role."},
            {"number":3,"title":"Indirect request","description":"It asks for review, confirmation, or update rather than directly asking for a password."},
            {"number":4,"title":tech_title,"description":tech_desc},
            {"number":5,"title":"Moderate 48-hour urgency","description":"The deadline creates pressure without using the obvious threat style seen in Easy emails."},
        ]
    return [
        {"number":1,"title":"Full professional identity","description":"The recipient is addressed with a full title/name, increasing credibility."},
        {"number":2,"title":"Daily role-specific workflow","description":f"The message is tied to {sc['topic']} in {sc['dept']}, making it highly believable for the role."},
        {"number":3,"title":"Named official attachment","description":f"The attachment ({attach}) appears formal and specific, matching the Advanced framework."},
        {"number":4,"title":"Mandatory QR code","description":"The QR code hides the destination and is the exclusive marker of Advanced examples."},
        {"number":5,"title":"Professional low-pressure tone","description":"The email avoids blunt threats and uses routine wording, making it harder to detect."},
    ]

def _make_email_v4(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    diff = _enhanced_diff(difficulty)
    role_type, sc, sub, person = _pick_v3(role, index, assessment)
    legit = not bool(is_phishing)
    channel = _channel_v4(diff, sc, index, legit)
    domain = _stable_domain_v4(diff, sc, channel, legit)
    sender = sc.get("sender") or _SIGNATURE_BY_SCENARIO_V3[role_type][index % len(_SIGNATURE_BY_SCENARIO_V3[role_type])]
    frm = f"{sender} <updates@{domain}>"
    subject = _subject_v4(diff, sc, channel, language, legit)
    link = _link_v4(diff, sc, channel, legit)
    attach = _attachment_v4(diff, sc, channel, legit)
    body = _body_v4(role_type, sc, sub, person, diff, channel, language, legit, index, link, attach)
    is_ar = language == "Arabic"
    if legit:
        indicators = []
        why = "This message is legitimate because it avoids credential requests, suspicious external pressure, and threat-based urgency." if not is_ar else "هذه الرسالة شرعية لأنها لا تطلب بيانات دخول ولا تستخدم ضغطًا أو تهديدًا مشبوهًا."
        attack = "Legitimate Email" if not is_ar else "رسالة شرعية"
    else:
        indicators = _analysis_v4(diff, sc, channel, link, attach, language, body)
        why = (f"This email is risky because it uses {diff} phishing indicators inside a healthcare workflow about {sc['topic']}." if not is_ar else f"هذه الرسالة خطرة لأنها تستخدم مؤشرات تصيد من مستوى {diff} داخل سياق صحي مرتبط بـ {sc['topic']}.")
        attack = ({"easy":"Obvious Credential Harvesting", "medium":f"Intermediate {channel.title()} Phishing", "hard":"Advanced QR Phishing"}[diff] if not is_ar else {"easy":"تصيد واضح لبيانات الدخول", "medium":"تصيد متوسط", "hard":"تصيد متقدم عبر QR"}[diff])
    suspicious_text = "" if legit else ({"easy":"username and password", "medium":"within 48 hours", "hard":"QR code"}[diff] if not is_ar else {"easy":"اسم المستخدم وكلمة المرور", "medium":"خلال 48 ساعة", "hard":"رمز QR"}[diff])
    # suspicious_link must equal the exact URL in the body. For button, renderer will use it but not duplicate it.
    suspicious_link = "" if (legit or diff == "hard" or channel == "pdf") else link
    return {
        "email_type": attack,
        "attack_type": attack,
        "risk_level": "Safe" if legit else ("Critical" if diff == "hard" else "High" if diff == "medium" else "Medium"),
        "from": frm,
        "to": person[1],
        "subject": subject,
        "attachment": attach,
        "body": body,
        "suspicious_text": suspicious_text,
        "suspicious_link": suspicious_link,
        "is_phishing": not legit,
        "scenario_id": f"v4:{role_type}:{sc['sub']}:{sc['path']}:{diff}:{channel}:{index}",
        "subrole": sc["sub"],
        "indicators": indicators,
        "why_risky": why,
        "learning_tip": ("Check whether the request, sender, link, and technical element match the difficulty clues: Easy is obvious, Intermediate is semi-plausible, Advanced uses QR and formal documents." if not is_ar else "تحقق من الطلب والمرسل والرابط والعنصر التقني: السهل واضح، المتوسط شبه مقنع، والصعب يستخدم QR ومستندات رسمية."),
    }

def _validate_v4(result, role, difficulty, is_phishing=True):
    if not isinstance(result, dict):
        return False
    diff = _enhanced_diff(difficulty)
    body = str(result.get("body", ""))
    txt = "\n".join(str(result.get(k, "")) for k in ["from", "subject", "body", "attachment", "suspicious_link"])
    attach = str(result.get("attachment", "")).strip()
    link = str(result.get("suspicious_link", "")).strip()
    if is_phishing:
        if diff == "easy":
            if attach or "[QR" in txt or "](" in body or "[BUTTON" in body:
                return False
            if re.search(r"\b(Dr\.|Nurse|Pharm\.|Lab Specialist|Radiographer|Engineer|Analyst)\b", body):
                return False
            if "48 hours" in txt or "within 48" in txt:
                return False
            if not (link and link in body and link.startswith("http://")):
                return False
            if not re.search(r"password|username|credential", body, re.I):
                return False
        elif diff == "medium":
            if "[QR" in txt:
                return False
            tech_count = int(bool(link)) + int("](" in body) + int(bool(attach))
            # Button body also carries link; count it as button only.
            if "](" in body and link:
                tech_count -= 1
            if tech_count != 1:
                return False
            if not re.search(r"48 hours|خلال 48", txt, re.I):
                return False
        else:
            if not attach or "[QR" not in body:
                return False
            if re.search(r"closed TODAY|Act NOW|password now|will close today", txt, re.I):
                return False
    return True

# Final v4 overrides used by trainee-facing pages.
def generate_email(role, index, language, difficulty="medium"):
    for offset in (0, 5, 11, 17):
        result = _make_email_v4(role, index + offset, language, difficulty, True, assessment=False)
        if _validate_v4(result, role, difficulty, True):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=True)
            except Exception: pass
            return result
    return result

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    for offset in (0, 7, 13, 19):
        result = _make_email_v4(role, index + offset, language, difficulty, bool(is_phishing), assessment=True)
        if _validate_v4(result, role, difficulty, bool(is_phishing)):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
            except Exception: pass
            return result
    return result

def generate_other_email(index, language, difficulty):
    return generate_email("Other", index, language, difficulty)

def generate_other_assess_email(index, is_phishing, language, difficulty):
    return generate_assess_email("Other", index, is_phishing, language, difficulty)

# =============================================================
# END SCENARIO ENGINE v4
# =============================================================



# =============================================================
# SCENARIO CONTENT ENGINE v5 — Real 300 Topic Knowledge Base
# -------------------------------------------------------------
# This patch replaces the small v4 scenario pool with the full
# 300 Scenario Cards already defined in SCENARIO_LIBRARY.
# It keeps the existing UI and assessment flow unchanged.
# Key fix: Easy/Beginner no longer collapses into one generic
# "staff account/password" template. It is still Easy per the
# framework, but the subject/body now revolve around the selected
# healthcare scenario topic.
# =============================================================

_V5_DEPT_BY_SUBROLE = {
    "Doctor": "Medical Affairs",
    "Nurse": "Nursing Affairs",
    "Pharmacist": "Pharmacy Services",
    "Laboratory Specialist": "Laboratory Services",
    "Radiology Technician": "Radiology Services",
    "HR Officer": "Human Resources",
    "Medical Secretary": "Medical Administration",
    "Insurance Coordinator": "Insurance Office",
    "Procurement Officer": "Procurement Department",
    "Finance Officer": "Finance Department",
    "IT Support Engineer": "IT Helpdesk",
    "Network Engineer": "Network Operations",
    "Cybersecurity Analyst": "Cybersecurity Office",
    "Systems Administrator": "Systems Administration",
    "Clinical Informatics Specialist": "Clinical Informatics",
}

_V5_CHANNELS_BY_TOPIC = {
    "clinical": ["link", "pdf", "button", "calendar"],
    "admin": ["link", "pdf", "button", "reply"],
    "it": ["link", "button", "pdf", "ticket"],
}

_V5_PERSON_BY_SUBROLE = {
    "Doctor": [("Dr. Ahmed Alotaibi", "dr.ahmed.alotaibi@hospital.org"), ("Dr. Yousef Alghamdi", "dr.yousef.alghamdi@hospital.org"), ("Dr. Sara Almutairi", "dr.sara.almutairi@hospital.org")],
    "Nurse": [("Nurse Reem Alzahrani", "n.reem.alzahrani@hospital.org"), ("Nurse Noura Alshamri", "n.noura.alshamri@hospital.org"), ("Nurse Maha Alsubaie", "n.maha.alsubaie@hospital.org")],
    "Pharmacist": [("Pharm. Khalid Alqahtani", "ph.khalid.alqahtani@hospital.org"), ("Pharm. Sara Almutairi", "ph.sara.almutairi@hospital.org"), ("Pharm. Ziad Alharbi", "ph.ziad.alharbi@hospital.org")],
    "Laboratory Specialist": [("Lab Specialist Maha Alsubaie", "lab.maha.alsubaie@hospital.org"), ("Lab Specialist Faisal Alzahrani", "lab.faisal.alzahrani@hospital.org")],
    "Radiology Technician": [("Radiographer Faisal Alzahrani", "rad.faisal.alzahrani@hospital.org"), ("Radiology Tech Reem Alzahrani", "rad.reem.alzahrani@hospital.org")],
    "default": [("Ahmed Alotaibi", "a.ahmed.alotaibi@hospital.org"), ("Maha Alsubaie", "a.maha.alsubaie@hospital.org"), ("Khalid Alqahtani", "a.khalid.alqahtani@hospital.org")],
}

_V5_AR_LABELS = {
    "Doctor": "دكتور", "Nurse": "ممرضة", "Pharmacist": "صيدلي", "Laboratory Specialist": "أخصائي مختبر", "Radiology Technician": "فني أشعة",
    "HR Officer": "موظف موارد بشرية", "Medical Secretary": "سكرتير طبي", "Insurance Coordinator": "منسق تأمين", "Procurement Officer": "موظف مشتريات", "Finance Officer": "موظف مالية",
    "IT Support Engineer": "مهندس دعم تقني", "Network Engineer": "مهندس شبكات", "Cybersecurity Analyst": "محلل أمن سيبراني", "Systems Administrator": "مسؤول أنظمة", "Clinical Informatics Specialist": "أخصائي معلوماتية صحية"
}

def _v5_slug(text):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).lower()).strip("-")
    s = re.sub(r"-+", "-", s)
    return (s[:42] or "hospital-task").strip("-")

def _v5_pick_full_card(role, index, assessment=False):
    role_type0 = _enhanced_role_type(role)
    if role_type0 == "other":
        role_type = ["clinical", "admin", "it"][(index + (2 if assessment else 0)) % 3]
    else:
        role_type = role_type0
    cards = SCENARIO_LIBRARY.get(role_type, SCENARIO_LIBRARY["clinical"])
    # Use a prime stride to cover all 100 before repeating.
    pos = ((index * 17) + (31 if assessment else 0)) % len(cards)
    base = cards[pos]
    sub = base.get("sub_role", "Staff")
    dept = _V5_DEPT_BY_SUBROLE.get(sub, base.get("sender", "Hospital Department"))
    topic = base.get("scenario", "hospital workflow review")
    path = _v5_slug(topic)
    attach = re.sub(r"[^A-Za-z0-9]+", "_", topic.title()).strip("_")[:36] + "_Summary.pdf"
    channels = list(_V5_CHANNELS_BY_TOPIC.get(role_type, ["link", "pdf", "button"]))
    sc = {
        "sub": sub,
        "topic": topic,
        "dept": dept,
        "sender": base.get("sender") or dept,
        "task": base.get("action") or f"review {topic}",
        "path": path,
        "channels": channels,
        "attach": attach,
        "source_id": base.get("id", "SC")
    }
    people = _V5_PERSON_BY_SUBROLE.get(sub, _V5_PERSON_BY_SUBROLE["default"])
    person = people[(index + len(topic)) % len(people)]
    return role_type, sc, sub, person

def _v5_domain(diff, sc, channel="link", legit=False):
    if legit:
        return "hospital.org"
    slug = _v5_slug(sc.get("topic", sc.get("path", "task"))).replace("-", "")[:22]
    if diff == "easy":
        variants = [
            f"fake-{slug}-login.com",
            f"urgent-{slug}-access.info",
            f"hospital-{slug}-alert.com",
            f"verify-{slug}-today.net",
        ]
    elif diff == "medium":
        variants = [
            f"hospital-{slug}-review.net",
            f"{slug}-hospital-support.org",
            f"hospital-workflow-{slug}.net",
            f"dept-{slug}-hospital.org",
        ]
    else:
        variants = [
            f"kfmc-{slug}.org.sa",
            f"mohservices-{slug}.org.sa",
            f"hospitaldept-{slug}.org.sa",
        ]
    return variants[(len(slug) + len(channel) + len(sc.get("sender", ""))) % len(variants)]

def _v5_link(diff, sc, channel, legit=False):
    if legit or diff == "hard" or channel in ("pdf", "reply"):
        return ""
    return f"http://{_v5_domain(diff, sc, channel, False)}/{sc['path']}"

def _v5_subject(diff, sc, channel, lang, legit=False):
    is_ar = lang == "Arabic"
    topic_en = str(sc["topic"]).title()
    dept = sc["dept"]
    if legit:
        return f"Internal Notice: {topic_en}" if not is_ar else f"إشعار داخلي: {sc['topic']}"
    if is_ar:
        if diff == "easy":
            opts = [
                f"تحذير عاجل: تعطل وصول {sc['topic']}",
                f"تأكيد فوري مطلوب: {sc['topic']}",
                f"إغلاق وصول المستشفى اليوم: {sc['topic']}",
                f"تنبيه نهائي: صفحة {dept}",
            ]
        elif diff == "medium":
            opts = [f"مراجعة مطلوبة: {sc['topic']}", f"متابعة {dept} خلال 48 ساعة", f"تحديث قسم {dept}: {sc['topic']}"]
        else:
            opts = [f"إجراء روتيني: {sc['topic']}", f"مراجعة رسمية لقسم {dept}", f"تأكيد مستند داخلي: {sc['topic']}"]
        return opts[(len(sc['topic']) + len(channel)) % len(opts)]
    if diff == "easy":
        opts = [
            f"URGENT: {topic_en} Access Will Stop Today",
            f"Immediate Verification Required: {topic_en}",
            f"Final Warning: Hospital {topic_en} Page",
            f"Action Required Today: {dept} Access",
        ]
    elif diff == "medium":
        opts = [f"{topic_en} Review", f"{dept} Follow-Up Required", f"Pending {topic_en} Confirmation", f"48-Hour {dept} Update"]
    else:
        opts = [f"Routine {topic_en} Confirmation", f"Official {dept} Workflow Review", f"Quarterly {dept} Document Check"]
    return opts[(len(sc['topic']) + len(channel)) % len(opts)]

def _v5_body(role_type, sc, sub, person, diff, channel, lang, legit=False, index=0, link="", attach=""):
    is_ar = lang == "Arabic"
    name, _email = person
    sig = sc.get("sender") or sc.get("dept") or "Hospital Department"
    topic = sc.get("topic", "hospital workflow")
    dept = sc.get("dept", "Hospital Department")
    medium_name = _medium_display_name_v3(name)
    ref = f"HSP-{202600 + index * 37}"

    if legit:
        if is_ar:
            return f"عزيزي/عزيزتي {medium_name}،\n\nهذا إشعار داخلي آمن من {dept} بخصوص {topic}. لا يطلب كلمة مرور، ولا يتضمن تهديدًا، ويمكن التحقق منه عبر أنظمة المستشفى الرسمية.\n\nمع التحية،\n{sig}"
        return f"Dear {medium_name},\n\nThis is a scheduled internal notice from {dept} about {topic}. It does not ask for a password, does not threaten account closure, and can be verified through official hospital systems.\n\nRegards,\n{sig}"

    if diff == "easy":
        greet = random.choice(_GENERIC_GREETINGS_V3[lang])
        if is_ar:
            templates = [
                f"{greet}،\n\nصفحة المستشفى الخاصة بـ {topic} ستتوقف اليوم. أدخل اسم المستخدم وكلمة المرور الآن حتى لا يتم إيقاف الوصول. هذا الإجراء مطلووب فورًا بسبب خطأ في نظام المستشفى.\n\nرابط الدخول: {link}\n\nفريق دعم المستشفى",
                f"{greet}،\n\nتم العثور على مشكله في وصول المستشفى المرتبط بـ {topic}. أرسل بيانات الدخول الآن أو سيتم إغلاق الصفحة اليوم.\n\nصفحة التحديث: {link}\n\nمكتب الدعم",
                f"{greet}،\n\nتحذير نهائي: لم يتم تأكيد حسابك لصفحة {topic}. أكّد كلمة المرور فورًا حتى لا يتوقف الوصول اليوم.\n\nالرابط: {link}\n\nفريق تنبيه المستشفى",
            ]
            return templates[index % len(templates)]
        templates = [
            f"{greet},\n\nThe hospital page for {topic} will stop TODAY. Enter your username and password now to keep access. This is requiered immediatly for staff access.\n\nLogin page: {link}\n\nThank You,\nHospital Support Team",
            f"{greet},\n\nWe found a problm with the hospital access page for {topic}. Send your credential update through this page today or the access will close.\n\nSecure page: {link}\n\nSupport Desk",
            f"{greet},\n\nFinal warning: your access to {topic} is not confirmed. Verfy your password NOW or the hospital page will stop today.\n\nUpdate link: {link}\n\nRegards,\nHospital Alert Team",
            f"{greet},\n\nThe {dept} page for {topic} needs immediat action. Enter username and password to avoid same-day access closure.\n\nAccess page: {link}\n\nHospital Support Desk",
        ]
        return templates[index % len(templates)]

    if diff == "medium":
        if is_ar:
            if channel == "pdf":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى مراجعة المرفق البسيط الخاص بـ {topic} من قسم {dept}. نحتاج تأكيد الحالة خلال 48 ساعة حتى لا يتأخر سير العمل.\n\nمع التحية،\n{sig}"
            if channel == "button":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيوجد تحديث مرتبط بـ {topic} في قسم {dept}. يرجى استخدام زر المراجعة لتأكيد الطلب خلال 48 ساعة.\n\n[مراجعة الطلب]({link})\n\nمع التحية،\n{sig}"
            if channel == "reply":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى الرد على هذه الرسالة لتأكيد حالة {topic} لقسم {dept} خلال 48 ساعة. لا ترسل كلمة مرور؛ نحتاج فقط تأكيدًا تشغيليًا.\n\nمع التحية،\n{sig}"
            return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى تأكيد البيانات المرتبطة بـ {topic} لقسم {dept} خلال 48 ساعة. هذا الإجراء يساعد على استمرار سير العمل.\n\nرابط المراجعة: {link}\n\nمع التحية،\n{sig}"
        if channel == "pdf":
            return f"Dear {medium_name},\n\nPlease review the attached summary for {topic} from {dept}. Confirmation is requested within 48 hours so the department workflow is not delayed.\n\nRegards,\n{sig}"
        if channel == "button":
            return f"Dear {medium_name},\n\nA department-level update is pending for {topic} in {dept}. Please use the review button to confirm the item within 48 hours.\n\n[Review Request]({link})\n\nRegards,\n{sig}"
        if channel == "reply":
            return f"Dear {medium_name},\n\nPlease reply to confirm the status of {topic} for {dept} within 48 hours. Do not send a password; only an operational confirmation is required.\n\nRegards,\n{sig}"
        return f"Dear {medium_name},\n\nPlease confirm the pending update related to {topic} for {dept} within 48 hours. This helps the team keep the workflow current without delaying patient services.\n\nReview page: {link}\n\nRegards,\n{sig}"

    # Advanced
    if is_ar:
        return f"عزيزي/عزيزتي {name}،\n\nضمن المراجعة التشغيلية الروتينية لقسم {dept}، يرجى الاطلاع على المستند الرسمي ({attach}) المتعلق بـ {topic}. يرتبط هذا الإجراء بمهمة يومية في وحدتكم ولا يتطلب إجراءً عاجلًا.\n\nيرجى استخدام زر المراجعة الداخلي من جهاز تابع للمستشفى، أو مسح رمز QR داخل شبكة المستشفى فقط: [QR: {sc['path']}]\n\nالمرجع: {ref}\nجهة الاتصال: {sig}\n\nمع التحية،\n{sig}"
    return f"Dear {name},\n\nAs part of the routine operational review for {dept}, please review the official document ({attach}) related to {topic}. This item is linked to your daily role workflow and does not require urgent action.\n\nUse the internal review button from a hospital-managed device, or scan the QR code inside the hospital network only: [QR: {sc['path']}]\n\nReference: {ref}\nContact unit: {sig}\n\nSincerely,\n{sig}"

def _v5_channel(diff, sc, index, legit=False):
    if diff == "easy":
        return "link"
    if diff == "hard":
        return "qr_official"
    channels = sc.get("channels") or ["link", "pdf", "button", "reply"]
    return channels[index % len(channels)]

def _make_email_v5(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    diff = _enhanced_diff(difficulty)
    role_type, sc, sub, person = _v5_pick_full_card(role, index, assessment)
    legit = not bool(is_phishing)
    channel = _v5_channel(diff, sc, index, legit)
    domain = _v5_domain(diff, sc, channel, legit)
    sender = sc.get("sender") or sc.get("dept") or "Hospital Department"
    frm = f"{sender} <updates@{domain}>"
    subject = _v5_subject(diff, sc, channel, language, legit)
    link = _v5_link(diff, sc, channel, legit)
    attach = _attachment_v4(diff, sc, channel, legit)
    if diff != "easy" and attach:
        attach = sc.get("attach") if diff == "medium" else "Official_" + sc.get("attach", "Protocol_Update.pdf")
    body = _v5_body(role_type, sc, sub, person, diff, channel, language, legit, index, link, attach)
    is_ar = language == "Arabic"
    if legit:
        indicators = []
        why = "This message is legitimate because it avoids credential requests, suspicious external pressure, and threat-based urgency." if not is_ar else "هذه الرسالة شرعية لأنها لا تطلب بيانات دخول ولا تستخدم ضغطًا أو تهديدًا مشبوهًا."
        attack = "Legitimate Email" if not is_ar else "رسالة شرعية"
    else:
        indicators = _analysis_v4(diff, sc, channel, link, attach, language, body)
        why = (f"This email is risky because it uses {diff} phishing indicators inside a healthcare workflow about {sc['topic']}." if not is_ar else f"هذه الرسالة خطرة لأنها تستخدم مؤشرات تصيد من مستوى {diff} داخل سياق صحي مرتبط بـ {sc['topic']}.")
        attack = ({"easy":"Obvious Credential Harvesting", "medium":f"Intermediate {channel.title()} Phishing", "hard":"Advanced QR Phishing"}[diff] if not is_ar else {"easy":"تصيد واضح لبيانات الدخول", "medium":"تصيد متوسط", "hard":"تصيد متقدم عبر QR"}[diff])
    suspicious_text = "" if legit else ({"easy":"username and password", "medium":"within 48 hours", "hard":"QR code"}[diff] if not is_ar else {"easy":"اسم المستخدم وكلمة المرور", "medium":"خلال 48 ساعة", "hard":"رمز QR"}[diff])
    suspicious_link = "" if (legit or diff == "hard" or channel in ("pdf", "reply")) else link
    return {
        "email_type": attack,
        "attack_type": attack,
        "risk_level": "Safe" if legit else ("Critical" if diff == "hard" else "High" if diff == "medium" else "Medium"),
        "from": frm,
        "to": person[1],
        "subject": subject,
        "attachment": attach,
        "body": body,
        "suspicious_text": suspicious_text,
        "suspicious_link": suspicious_link,
        "is_phishing": not legit,
        "scenario_id": f"v5:{role_type}:{sc.get('source_id')}:{sc['sub']}:{sc['path']}:{diff}:{channel}:{index}",
        "subrole": sc["sub"],
        "indicators": indicators,
        "why_risky": why,
        "learning_tip": ("Focus on whether the message matches the scenario and difficulty: Easy is obvious, Intermediate is semi-plausible, and Advanced hides risk in formal workflow." if not is_ar else "ركز على توافق الرسالة مع السيناريو ومستوى الصعوبة: السهل واضح، المتوسط شبه مقنع، والصعب يخفي الخطر داخل سير عمل رسمي."),
    }

def generate_email(role, index, language, difficulty="medium"):
    for offset in (0, 17, 29, 43):
        result = _make_email_v5(role, index + offset, language, difficulty, True, assessment=False)
        if _validate_v4(result, role, difficulty, True):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=True)
            except Exception: pass
            return result
    return result

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    for offset in (0, 19, 37, 53):
        result = _make_email_v5(role, index + offset, language, difficulty, bool(is_phishing), assessment=True)
        if _validate_v4(result, role, difficulty, bool(is_phishing)):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
            except Exception: pass
            return result
    return result

def generate_other_email(index, language, difficulty):
    return generate_email("Other", index, language, difficulty)

def generate_other_assess_email(index, is_phishing, language, difficulty):
    return generate_assess_email("Other", index, is_phishing, language, difficulty)

# =============================================================
# END SCENARIO CONTENT ENGINE v5
# =============================================================


# =============================================================
# EMAIL GENERATION ENGINE v6 — Template Diversity Library
# -------------------------------------------------------------
# Keeps the existing UI and scenario cards, but adds large internal
# libraries for subjects, greetings, body structures, credential requests,
# CTA labels, closings, explanations and learning tips. This fixes the
# repeated "account/password" style while preserving the difficulty framework.
# =============================================================

_V6_EASY_GREETINGS_EN = [
    "Dear Hospital Staff", "Dear Clinical Team", "Dear Healthcare Employee", "Attention Clinical Staff",
    "Dear Department Team", "Dear Hospital User", "Good Morning", "Dear Staff Member", "Dear Employee",
    "Attention Hospital Employee", "Dear Ward Team", "Dear Clinical User", "Dear Care Team",
    "Attention Staff Member", "Dear Hospital Team", "Dear Service User", "Dear Team Member",
    "To All Clinical Staff", "Dear Healthcare Team", "Attention Hospital Staff"
]
_V6_EASY_GREETINGS_AR = [
    "عزيزي/عزيزتي موظف المستشفى", "فريق الرعاية الصحية", "تنبيه لموظفي القسم", "عزيزي/عزيزتي عضو الفريق الصحي",
    "فريق المستشفى", "تنبيه عاجل لموظفي المستشفى", "عزيزي/عزيزتي الموظف", "فريق القسم السريري"
]

_V6_CREDENTIALS_EN = [
    "username and password", "hospital login details", "staff portal password", "employee ID and password",
    "hospital email credentials", "staff PIN and password", "network login", "portal username and password",
    "authentication code and password", "clinical system password", "hospital account details", "user ID and password",
    "access code", "identity verification details", "staff account credentials"
]
_V6_CREDENTIALS_AR = [
    "اسم المستخدم وكلمة المرور", "بيانات الدخول للمستشفى", "رقم الموظف وكلمة المرور", "رمز الدخول وكلمة المرور",
    "بيانات حساب الموظف", "كلمة مرور النظام السريري"
]

_V6_CTA_EN = [
    "Login page", "Access page", "Secure page", "Update link", "Verification page", "Staff portal",
    "Clinical portal", "Department page", "Review page", "Hospital access page", "Continue here", "Action page",
    "Workflow page", "Confirmation page", "System page", "Urgent update page"
]
_V6_CTA_AR = ["رابط الدخول", "صفحة التحديث", "صفحة التحقق", "بوابة الموظفين", "صفحة القسم", "رابط المراجعة"]

_V6_EASY_SUBJECTS_EN = [
    "Immediate Action Required: {Topic}", "URGENT: {Topic} Access Will Stop Today", "Final Warning: {Topic} Page",
    "Action Required Today: {Dept} Access", "Hospital Verification Required: {Topic}", "Staff Access Alert: {Topic}",
    "{Dept} Page Locked Today", "Immediate Password Verification Required", "Important: {Topic} Login Notice",
    "Final Reminder: {Dept} System", "TODAY ONLY: {Topic} Update", "Hospital Portal Problem: {Topic}",
    "Clinical Access Warning: {Topic}", "{Topic} Confirmation Failed", "Required Now: {Dept} Access",
    "Staff Login Problem: {Topic}", "Hospital Account Notice: {Topic}", "Service Access Warning: {Dept}",
    "Action Needed: {Topic} Review", "Access Closure Notice: {Topic}", "Verify Staff Access for {Topic}",
    "Urgent Clinical Page Update", "Department Access Hold: {Topic}", "Hospital Workflow Login Alert",
    "Security Check Needed for {Topic}"
]
_V6_EASY_SUBJECTS_AR = [
    "إجراء عاجل مطلوب: {topic}", "تحذير نهائي: صفحة {topic}", "إغلاق وصول اليوم: {dept}",
    "تأكيد فوري مطلوب: {topic}", "مشكلة في وصول المستشفى: {topic}"
]

_V6_EASY_BODY_TEMPLATES_EN = [
    "The hospital page for {topic} will stop TODAY. Enter your {credential} now to keep access. This is requiered immediatly for staff access.",
    "We found a problm with the hospital access page for {topic}. Send your {credential} through this page today or the access will close.",
    "Final warning: your access to {topic} is not confirmed. Verfy your {credential} NOW or the hospital page will stop today.",
    "The {dept} page for {topic} needs immediat action. Enter your {credential} to avoid same-day access closure.",
    "Your hospital record for {topic} is pending. Add your {credential} now or the service will be blocked TODAY.",
    "The clinical page about {topic} has a login error. Use your {credential} now to fix it before your access stop today.",
    "Staff access to {topic} is marked as unverifyed. Confirm your {credential} immediately to keep using the page.",
    "A hospital system error was found in {topic}. Re-enter your {credential} today or your staff access will close.",
    "Your profile is missing on the {topic} page. Update your {credential} NOW to avoid service stop today.",
    "The {dept} access list did not confirm your login. Submit your {credential} immediatly or access will be removed today.",
    "Hospital staff must check {topic} today. Enter your {credential} on the page or your access will be disabled.",
    "The page for {topic} is expired. Verfy your {credential} now to restore access before end of day.",
    "Your clinical access for {topic} will be deleted today. Send {credential} now to continue working.",
    "The hospital needs a fast re-check for {topic}. Enter your {credential} now. Do not delay this update.",
    "Account mismatch found for {topic}. Use the page to confirm {credential} or the account will close TODAY.",
    "Your {dept} access is not safe. Enter your {credential} now to secure the hospital page.",
    "The patient workflow page for {topic} cannot continue. Add your {credential} today to keep access.",
    "System notice: {topic} access failed. Confirm your {credential} NOW or staff access will be stopped.",
    "The hospital page for {topic} need verification. Submit your {credential} immediatly to keep department access.",
    "Your staff portal linked to {topic} is on hold. Enter {credential} today to reactivate it.",
    "The {dept} login check is incomplete. Please update your {credential} now or service will stop.",
    "Clinical service for {topic} requires a quick login check. Enter {credential} today to prevent closure.",
    "Your access to {topic} is under final review. Submit {credential} now to keep the page open.",
    "Urgent staff notice: {topic} is blocked for users who do not verify {credential} today.",
    "A hospital alert was created for {topic}. Add {credential} now to prevent same-day account closure."
]
_V6_EASY_BODY_TEMPLATES_AR = [
    "صفحة المستشفى الخاصة بـ {topic} ستتوقف اليوم. أدخل {credential} الآن حتى لا يتم إيقاف الوصول. هذا الإجراء مطلووب فورًا.",
    "تم العثور على مشكله في صفحة {topic}. أرسل {credential} اليوم أو سيتم إغلاق الوصول.",
    "تحذير نهائي: لم يتم تأكيد وصولك إلى {topic}. أكّد {credential} الآن حتى لا يتوقف الوصول اليوم.",
    "صفحة {dept} الخاصة بـ {topic} تحتاج إجراء فوري. أدخل {credential} لتجنب إغلاق الوصول اليوم."
]

_V6_MEDIUM_BODY_TEMPLATES_EN = [
    "A department-level update is pending for {topic} in {dept}. Please confirm the item within 48 hours so the workflow can continue.",
    "Please review the latest notice about {topic} for {dept}. Confirmation is requested within 48 hours.",
    "The {dept} team is reconciling records related to {topic}. Please complete the requested review within 48 hours.",
    "A follow-up item for {topic} has been assigned to your department queue. Please confirm or update the status within 48 hours.",
    "The current {topic} workflow requires staff acknowledgement within 48 hours to keep the department record current.",
    "Please check the pending {topic} entry for {dept}. The review window closes in 48 hours.",
    "A routine department confirmation is needed for {topic}. Please respond or review the item within 48 hours.",
    "The {dept} list for {topic} was updated today. Please verify the entry within 48 hours.",
    "Please review the provided summary for {topic}. This helps {dept} avoid workflow delays within the next 48 hours.",
    "A scheduled review was opened for {topic}. Please complete the department confirmation within 48 hours.",
    "The {dept} office requests confirmation of {topic} before the 48-hour review window ends.",
    "A pending operational note about {topic} requires your acknowledgement within 48 hours."
]

_V6_LEGIT_BODY_TEMPLATES_EN = [
    "This is a scheduled internal notice from {dept} about {topic}. It does not ask for a password, does not threaten account closure, and can be verified through official hospital systems.",
    "Please note that {dept} has published a routine update about {topic}. No login details are required, and the information is available through the hospital intranet.",
    "This message confirms that the {topic} update has been posted for staff awareness. It contains no external link and no request for credentials.",
    "The {dept} team is sharing a routine reminder about {topic}. Staff may review the details through the official hospital system when convenient.",
    "This is an internal operational bulletin regarding {topic}. The notice is informational only and does not require password verification.",
    "A routine staff communication about {topic} has been issued by {dept}. For questions, contact the department through official hospital channels.",
    "The hospital has scheduled an internal review related to {topic}. This email is for awareness and does not request any sensitive information.",
    "Please be aware of the upcoming {topic} activity managed by {dept}. No action is required outside official hospital platforms.",
    "This is a safe internal update concerning {topic}. It avoids urgency, external domains, attachments, and credential requests.",
    "The {dept} office is notifying staff about {topic}. Verification can be performed only through the official hospital system if needed."
]
_V6_LEGIT_BODY_TEMPLATES_AR = [
    "هذا إشعار داخلي آمن من {dept} بخصوص {topic}. لا يطلب كلمة مرور ولا يستخدم تهديدًا أو رابطًا خارجيًا مشبوهًا.",
    "نود إعلامكم بتحديث روتيني من {dept} حول {topic}. لا توجد أي حاجة لإدخال بيانات دخول أو إرسال معلومات حساسة.",
    "هذه رسالة داخلية للتوعية فقط بخصوص {topic}. يمكن التحقق منها عبر الأنظمة الرسمية للمستشفى."
]

_V6_CLOSINGS_EN = [
    "Hospital Support Desk", "Clinical Support Team", "Hospital Alert Team", "Department Support Office", "Clinical Operations Desk",
    "Medical Affairs Office", "Patient Safety Office", "Quality Management Office", "Clinical Governance", "Hospital Administration",
    "Nursing Affairs", "Laboratory Services", "Radiology Administration", "Pharmacy Safety Unit", "Credentialing Office",
    "Outpatient Services", "Infection Prevention Unit", "Medication Safety Office", "Patient Access Center", "Clinical Education Center"
]
_V6_CLOSINGS_AR = ["مكتب دعم المستشفى", "فريق الدعم السريري", "إدارة الجودة", "الشؤون الطبية", "قسم سلامة المرضى", "إدارة القسم"]

_V6_TIPS_EN = [
    "Check whether the sender, link, and requested action match normal hospital workflow.",
    "Hospital systems should not request passwords through email links.",
    "Treat same-day account closure threats as a strong phishing signal.",
    "A clinical topic can still be phishing if it asks for credentials.",
    "Verify unusual clinical requests through the official hospital system before acting.",
    "Do not enter staff credentials on pages reached from unexpected emails.",
    "Look for mismatch between the clinical topic and the login domain.",
    "Visible fake URLs and direct password requests are clear warning signs.",
    "Report suspicious hospital workflow emails to the security team.",
    "Legitimate hospital notices rarely combine urgency with password requests."
]
_V6_TIPS_AR = [
    "تحقق من المرسل والرابط والطلب قبل التصرف.",
    "أنظمة المستشفى لا تطلب كلمة المرور عبر روابط بريدية.",
    "التهديد بإغلاق الحساب في نفس اليوم علامة تصيد قوية.",
    "قد يكون الموضوع صحيًا لكنه تصيدي إذا طلب بيانات الدخول."
]

_V6_DOMAIN_PREFIXES = ["secure", "verify", "urgent", "staff", "access", "update", "review", "portal", "login", "confirm", "today", "clinical", "dept", "workflow"]
_V6_DOMAIN_SUFFIXES = ["login.com", "access.info", "verify.net", "alert.com", "review.org", "portal.co", "check.net", "service.info", "update.com", "confirm.org"]


def _v6_pick(items, index, salt=0):
    if not items:
        return ""
    return items[(index * 7 + salt * 13) % len(items)]


def _v6_topic_title(sc):
    return str(sc.get("topic", "hospital workflow")).title()


def _v6_domain(diff, sc, channel="link", legit=False, index=0):
    if legit:
        return "hospital.org"
    slug = _v5_slug(sc.get("topic", sc.get("path", "task"))).replace("-", "")[:24]
    deptslug = _v5_slug(sc.get("dept", "hospital")).replace("-", "")[:14]
    if diff == "easy":
        # deliberately obvious but varied
        variants = [
            f"{_v6_pick(_V6_DOMAIN_PREFIXES,index,1)}-{slug}-{_v6_pick(_V6_DOMAIN_SUFFIXES,index,2)}",
            f"fake-{slug}-{_v6_pick(_V6_DOMAIN_SUFFIXES,index,3)}",
            f"hospital-{deptslug}-{_v6_pick(_V6_DOMAIN_SUFFIXES,index,4)}",
            f"{slug}-hospital-{_v6_pick(_V6_DOMAIN_SUFFIXES,index,5)}",
            f"{_v6_pick(_V6_DOMAIN_PREFIXES,index,6)}-{deptslug}-{slug}.info",
            f"{slug}-{_v6_pick(_V6_DOMAIN_PREFIXES,index,7)}-page.net",
        ]
    elif diff == "medium":
        variants = [
            f"hospital-{slug}-review.net", f"{slug}-hospital-support.org", f"workflow-{deptslug}-{slug}.net",
            f"dept-{slug}-hospital.org", f"hospitalservice-{slug}.org", f"review-{slug}-hospital.net"
        ]
    else:
        variants = [f"mohservices-{slug}.org.sa", f"hospitaldept-{slug}.org.sa", f"kfmc-{deptslug}-{slug}.org.sa"]
    return variants[index % len(variants)]


def _v6_link(diff, sc, channel, legit=False, index=0):
    if legit or diff == "hard" or channel in ("pdf", "reply"):
        return ""
    return f"http://{_v6_domain(diff, sc, channel, False, index)}/{_v5_slug(sc.get('topic','review'))}"


def _v6_subject(diff, sc, channel, lang, legit=False, index=0):
    is_ar = lang == "Arabic"
    topic = sc.get("topic", "hospital workflow")
    dept = sc.get("dept", "Hospital Department")
    if legit:
        if is_ar:
            return _v6_pick([f"إشعار داخلي: {topic}", f"تحديث روتيني من {dept}", f"معلومة داخلية: {topic}"], index)
        patterns = [
            "Internal Notice: {Topic}", "Scheduled Update: {Topic}", "Routine {Dept} Notice", "Staff Bulletin: {Topic}",
            "Monthly {Dept} Review", "Information Only: {Topic}", "Department Memo: {Topic}", "Clinical Bulletin: {Topic}",
            "Awareness Notice: {Topic}", "Routine Workflow Note: {Topic}"
        ]
        return _v6_pick(patterns, index).format(Topic=_v6_topic_title(sc), Dept=dept)
    if is_ar:
        return _v6_pick(_V6_EASY_SUBJECTS_AR if diff == "easy" else [f"مراجعة مطلوبة: {topic}", f"متابعة {dept} خلال 48 ساعة", f"تحديث قسم {dept}: {topic}"], index).format(topic=topic, dept=dept)
    if diff == "easy":
        return _v6_pick(_V6_EASY_SUBJECTS_EN, index).format(Topic=_v6_topic_title(sc), Dept=dept)
    if diff == "medium":
        patterns = [
            "{Topic} Review", "{Dept} Follow-Up Required", "Pending {Topic} Confirmation", "48-Hour {Dept} Update",
            "Department Workflow Review: {Topic}", "Updated {Dept} Procedure", "Clinical Documentation Review", "{Topic} Acknowledgement Needed",
            "New {Dept} Checklist", "Operational Notice: {Topic}", "Patient Workflow Revision", "Department Policy Update"
        ]
        return _v6_pick(patterns, index).format(Topic=_v6_topic_title(sc), Dept=dept)
    patterns = [
        "Routine {Topic} Confirmation", "Official {Dept} Workflow Review", "Quarterly {Dept} Document Check",
        "Annual Clinical Governance Review", "Regulatory Policy Confirmation: {Topic}", "Quality Assurance Documentation",
        "Electronic Clinical Record Validation", "Department Compliance Audit", "Routine Internal Review: {Topic}"
    ]
    return _v6_pick(patterns, index).format(Topic=_v6_topic_title(sc), Dept=dept)


def _v6_body(role_type, sc, sub, person, diff, channel, lang, legit=False, index=0, link="", attach=""):
    is_ar = lang == "Arabic"
    name, _email = person
    topic = sc.get("topic", "hospital workflow")
    dept = sc.get("dept", "Hospital Department")
    sig = _v6_pick(_V6_CLOSINGS_AR if is_ar else _V6_CLOSINGS_EN, index, len(topic))
    ref = f"HSP-{202600 + index * 37}"

    if legit:
        if is_ar:
            greeting = f"عزيزي/عزيزتي {_medium_display_name_v3(name)}"
            body = _v6_pick(_V6_LEGIT_BODY_TEMPLATES_AR, index).format(topic=topic, dept=dept)
            return f"{greeting}،\n\n{body}\n\nمع التحية،\n{sig}"
        greeting = f"Dear {_medium_display_name_v3(name)}"
        body = _v6_pick(_V6_LEGIT_BODY_TEMPLATES_EN, index).format(topic=topic, dept=dept)
        return f"{greeting},\n\n{body}\n\nRegards,\n{sig}"

    if diff == "easy":
        if is_ar:
            greeting = _v6_pick(_V6_EASY_GREETINGS_AR, index, len(topic))
            cred = _v6_pick(_V6_CREDENTIALS_AR, index, len(dept))
            cta = _v6_pick(_V6_CTA_AR, index, 3)
            body = _v6_pick(_V6_EASY_BODY_TEMPLATES_AR, index, len(topic)).format(topic=topic, dept=dept, credential=cred)
            return f"{greeting}،\n\n{body}\n\n{cta}: {link}\n\n{sig}"
        greeting = _v6_pick(_V6_EASY_GREETINGS_EN, index, len(topic))
        cred = _v6_pick(_V6_CREDENTIALS_EN, index, len(dept))
        cta = _v6_pick(_V6_CTA_EN, index, len(cred))
        body = _v6_pick(_V6_EASY_BODY_TEMPLATES_EN, index, len(topic)).format(topic=topic, dept=dept, credential=cred)
        return f"{greeting},\n\n{body}\n\n{cta}: {link}\n\n{sig}"

    medium_name = _medium_display_name_v3(name)
    if diff == "medium":
        if is_ar:
            if channel == "pdf":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى مراجعة المرفق البسيط الخاص بـ {topic} من قسم {dept}. نحتاج تأكيد الحالة خلال 48 ساعة حتى لا يتأخر سير العمل.\n\nمع التحية،\n{sig}"
            if channel == "button":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيوجد تحديث مرتبط بـ {topic} في قسم {dept}. يرجى استخدام زر المراجعة لتأكيد الطلب خلال 48 ساعة.\n\n[مراجعة الطلب]({link})\n\nمع التحية،\n{sig}"
            if channel == "reply":
                return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى الرد لتأكيد حالة {topic} لقسم {dept} خلال 48 ساعة. لا ترسل كلمة مرور؛ نحتاج فقط تأكيدًا تشغيليًا.\n\nمع التحية،\n{sig}"
            return f"عزيزي/عزيزتي {medium_name}،\n\nيرجى تأكيد البيانات المرتبطة بـ {topic} لقسم {dept} خلال 48 ساعة. هذا الإجراء يساعد على استمرار سير العمل.\n\nرابط المراجعة: {link}\n\nمع التحية،\n{sig}"
        body = _v6_pick(_V6_MEDIUM_BODY_TEMPLATES_EN, index, len(topic)).format(topic=topic, dept=dept)
        if channel == "pdf":
            return f"Dear {medium_name},\n\n{body}\n\nPlease review the attached summary file.\n\nRegards,\n{sig}"
        if channel == "button":
            return f"Dear {medium_name},\n\n{body}\n\n[Review Request]({link})\n\nRegards,\n{sig}"
        if channel == "reply":
            return f"Dear {medium_name},\n\n{body}\n\nA reply confirmation is sufficient; do not send passwords or patient identifiers.\n\nRegards,\n{sig}"
        return f"Dear {medium_name},\n\n{body}\n\nReview page: {link}\n\nRegards,\n{sig}"

    # Advanced
    full_name = name
    if is_ar:
        return f"عزيزي/عزيزتي {full_name}،\n\nضمن مراجعة تشغيلية روتينية لقسم {dept}، يرجى الاطلاع على المستند الرسمي ({attach}) المتعلق بـ {topic}. يرتبط هذا الإجراء بمهمة يومية في وحدتكم ولا يتطلب إجراءً عاجلًا.\n\nيرجى استخدام زر المراجعة الداخلي من جهاز تابع للمستشفى، أو مسح رمز QR داخل شبكة المستشفى فقط: [QR: {sc['path']}]\n\nالمرجع: {ref}\nجهة الاتصال: {sig}\n\nمع التحية،\n{sig}"
    adv_bodies = [
        "As part of the routine operational review for {dept}, please review the official document ({attach}) related to {topic}. This item is linked to your daily role workflow and does not require urgent action.",
        "The {dept} office is completing a scheduled governance check for {topic}. Please review the official document ({attach}) through the internal channel when convenient.",
        "A formal quality review related to {topic} has been opened for {dept}. The attached document ({attach}) is provided for routine confirmation only.",
        "Please review the official {dept} packet ({attach}) concerning {topic}. This is part of a planned internal audit and carries no emergency deadline.",
        "The hospital is conducting a routine validation of {topic} in {dept}. Please access the official document ({attach}) through the approved internal workflow."
    ]
    body = _v6_pick(adv_bodies, index, len(topic)).format(topic=topic, dept=dept, attach=attach)
    return f"Dear {full_name},\n\n{body}\n\nUse the internal review button from a hospital-managed device, or scan the QR code inside the hospital network only: [QR: {sc['path']}]\n\nReference: {ref}\nContact unit: {sig}\n\nSincerely,\n{sig}"


def _analysis_v6(diff, sc, channel, link, attach, lang, body=""):
    is_ar = lang == "Arabic"
    topic = sc.get("topic", "hospital workflow")
    dept = sc.get("dept", "Hospital Department")
    if is_ar:
        return _analysis_v4(diff, sc, channel, link, attach, lang, body)
    if diff == "easy":
        # Still maps to the framework, but each explanation is tied to the rendered topic/link/body.
        domain = link.replace("http://", "").split("/")[0] if link else "the displayed link"
        return [
            {"number":1,"title":"Generic healthcare greeting","description":f"The greeting avoids a named recipient or professional title, while the message claims to involve {topic}."},
            {"number":2,"title":"Visible fake domain","description":f"The URL is visible and points to {domain}, which is not an official hospital domain."},
            {"number":3,"title":"Direct credential request","description":"The email asks for login information such as a password, username, PIN, ID, or staff credentials."},
            {"number":4,"title":"Same-day pressure","description":f"The message threatens loss of access today to force action before the recipient verifies the {dept} request."},
            {"number":5,"title":"Easy-level technical pattern","description":"There is no QR code or attachment; the risk is intentionally obvious through the link, wording, and credential request."},
        ]
    if diff == "medium":
        tech_title = "PDF attachment" if attach else "review button" if "](" in body else "look-alike link" if link else "reply request"
        return [
            {"number":1,"title":"Partial professional context","description":f"The message uses department context ({dept}) and the scenario ({topic}) to appear plausible."},
            {"number":2,"title":"Indirect action request","description":"It requests review, confirmation, acknowledgement, or status update rather than openly asking for a password."},
            {"number":3,"title":"Moderate 48-hour urgency","description":"The 48-hour window creates pressure without the obvious account-closure threat used in Easy examples."},
            {"number":4,"title":tech_title.title(),"description":"Only one moderate technical element is used, which matches the Intermediate framework."},
            {"number":5,"title":"Healthcare workflow framing","description":f"The request is embedded in a realistic clinical workflow about {topic}."},
        ]
    return [
        {"number":1,"title":"Full professional workflow","description":f"The email is tied to a specific {dept} task about {topic}, making it role-specific."},
        {"number":2,"title":"Official-looking attachment","description":f"The attachment name ({attach}) appears formal and relevant to the workflow."},
        {"number":3,"title":"QR-based access path","description":"The QR code hides the destination URL, which is the required Advanced technical marker."},
        {"number":4,"title":"Low-pressure tone","description":"The email avoids obvious threats, making the risk harder to spot."},
        {"number":5,"title":"Near-official presentation","description":"The sender and wording resemble routine hospital governance or quality procedures."},
    ]


def _make_email_v6(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    diff = _enhanced_diff(difficulty)
    role_type, sc, sub, person = _v5_pick_full_card(role, index, assessment)
    legit = not bool(is_phishing)
    channel = _v5_channel(diff, sc, index, legit)
    if diff == "easy":
        channel = "link"
    domain = _v6_domain(diff, sc, channel, legit, index)
    sender = sc.get("sender") or sc.get("dept") or "Hospital Department"
    frm = f"{sender} <updates@{domain}>"
    subject = _v6_subject(diff, sc, channel, language, legit, index)
    link = _v6_link(diff, sc, channel, legit, index)
    attach = _attachment_v4(diff, sc, channel, legit)
    if diff == "easy":
        attach = ""
    elif diff != "easy" and attach:
        attach = sc.get("attach") if diff == "medium" else "Official_" + sc.get("attach", "Protocol_Update.pdf")
    body = _v6_body(role_type, sc, sub, person, diff, channel, language, legit, index, link, attach)
    is_ar = language == "Arabic"
    if legit:
        indicators = []
        why = "This message is legitimate because it does not ask for credentials, avoids threat-based urgency, and keeps the workflow inside official hospital channels." if not is_ar else "هذه الرسالة شرعية لأنها لا تطلب بيانات دخول ولا تستخدم تهديدًا أو رابطًا خارجيًا مشبوهًا."
        attack = "Legitimate Email" if not is_ar else "رسالة شرعية"
    else:
        indicators = _analysis_v6(diff, sc, channel, link, attach, language, body)
        why_starters = [
            "This message is risky because", "The main risk is that", "Multiple phishing clues appear because",
            "This email should be treated as suspicious because", "The scenario becomes risky because"
        ]
        starter = _v6_pick(why_starters, index, len(sc.get("topic", "")))
        why = (f"{starter} it uses {diff} phishing indicators inside a healthcare workflow about {sc['topic']}." if not is_ar else f"هذه الرسالة خطرة لأنها تستخدم مؤشرات تصيد من مستوى {diff} داخل سياق صحي مرتبط بـ {sc['topic']}.")
        attack = ({"easy":"Obvious Credential Harvesting", "medium":f"Intermediate {channel.title()} Phishing", "hard":"Advanced QR Phishing"}[diff] if not is_ar else {"easy":"تصيد واضح لبيانات الدخول", "medium":"تصيد متوسط", "hard":"تصيد متقدم عبر QR"}[diff])
    suspicious_text = "" if legit else ({"easy":"credential request", "medium":"within 48 hours", "hard":"QR code"}[diff] if not is_ar else {"easy":"طلب بيانات الدخول", "medium":"خلال 48 ساعة", "hard":"رمز QR"}[diff])
    suspicious_link = "" if (legit or diff == "hard" or channel in ("pdf", "reply")) else link
    return {
        "email_type": attack,
        "attack_type": attack,
        "risk_level": "Safe" if legit else ("Critical" if diff == "hard" else "High" if diff == "medium" else "Medium"),
        "from": frm,
        "to": person[1],
        "subject": subject,
        "attachment": attach,
        "body": body,
        "suspicious_text": suspicious_text,
        "suspicious_link": suspicious_link,
        "is_phishing": not legit,
        "scenario_id": f"v6:{role_type}:{sc.get('source_id')}:{sc['sub']}:{sc['path']}:{diff}:{channel}:{index}",
        "subrole": sc["sub"],
        "indicators": indicators,
        "why_risky": why,
        "learning_tip": _v6_pick(_V6_TIPS_AR if is_ar else _V6_TIPS_EN, index, len(sc.get("topic", ""))),
    }


def generate_email(role, index, language, difficulty="medium"):
    for offset in (0, 17, 29, 43, 61):
        result = _make_email_v6(role, index + offset, language, difficulty, True, assessment=False)
        if _validate_v4(result, role, difficulty, True):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=True)
            except Exception: pass
            return result
    return result


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    for offset in (0, 19, 37, 53, 71):
        result = _make_email_v6(role, index + offset, language, difficulty, bool(is_phishing), assessment=True)
        if _validate_v4(result, role, difficulty, bool(is_phishing)):
            try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
            except Exception: pass
            return result
    return result


def generate_other_email(index, language, difficulty):
    return generate_email("Other", index, language, difficulty)


def generate_other_assess_email(index, is_phishing, language, difficulty):
    return generate_assess_email("Other", index, is_phishing, language, difficulty)

# =============================================================
# END EMAIL GENERATION ENGINE v6
# =============================================================



# =============================================================
# API-FIRST DYNAMIC GENERATION ENGINE v10
# -------------------------------------------------------------
# Purpose:
# - Trainee-facing emails are generated by the selected API provider.
# - The 300-card scenario library remains the content backbone.
# - Difficulty follows the approved progressive framework.
# - Easy is obvious but not one fixed short template: body shape,
#   sender, subject, department, wording, and scenario vary each run.
# - AI Tutor Analysis is generated with the same email and must cite
#   visible facts from that specific email.
# =============================================================

_V10_BODY_SHAPES_EN = [
    "notice with background, impact, action, link, closing",
    "workflow alert with affected item, operational consequence, required action, link, closing",
    "department update with short context, failed status, required credential step, link, closing",
    "service interruption warning with affected healthcare workflow, direct request, visible URL, closing",
    "portal record mismatch with explanation, staff action, deadline, visible URL, closing",
    "queue access warning with scenario detail, sensitive request, visible URL, support closing",
]
_V10_BODY_SHAPES_AR = [
    "إشعار مع خلفية، أثر، إجراء، رابط، توقيع",
    "تنبيه سير عمل مع العنصر المتأثر، نتيجة تشغيلية، إجراء مطلوب، رابط، توقيع",
    "تحديث قسم مع سياق قصير، حالة فشل، خطوة بيانات دخول، رابط، توقيع",
    "تحذير تعطل خدمة مع سير عمل صحي متأثر، طلب مباشر، رابط ظاهر، توقيع",
]

_V10_EASY_REQUESTS_EN = [
    "enter username and password", "verify staff password", "confirm employee ID and password",
    "add portal password", "submit staff PIN and password", "update login credentials",
]
_V10_EASY_REQUESTS_AR = [
    "إدخال اسم المستخدم وكلمة المرور", "تأكيد كلمة مرور الموظف", "تحديث بيانات الدخول",
    "إرسال الرقم الوظيفي وكلمة المرور", "إضافة كلمة مرور البوابة",
]

_V10_EASY_PRESSURE_EN = [
    "access will stop TODAY", "the page will be disabled today", "your department access will close today",
    "service access will be blocked before the end of the day", "the hospital record will be locked today",
]
_V10_EASY_PRESSURE_AR = [
    "سيتم إيقاف الوصول اليوم", "ستتعطل الصفحة اليوم", "سيُغلق وصول القسم اليوم", "سيتم حجب الخدمة قبل نهاية اليوم",
]

_V10_MEDIUM_CHANNELS = ["lookalike_link", "simple_button", "simple_pdf", "reply_confirmation"]
_V10_HARD_CHANNELS = ["official_attachment_button", "qr_with_document", "sharepoint_style_review", "microsoft365_style_notice"]


def _v10_language_name(language):
    return "Arabic" if language == "Arabic" else "English"


def _v10_select_card(role, index, assessment=False):
    # reuse the validated 300-card picker so every role has 100+ ideas
    return _v5_pick_full_card(role, index + random.randint(0, 999), assessment)


def _v10_diff_contract(diff, phishing=True, is_ar=False):
    if is_ar:
        if not phishing:
            return "شرعي: نطاق رسمي hospital.org أو moh.gov.sa فقط، لا طلب كلمة مرور، لا تهديد، لا رابط خارجي مشبوه، نبرة طبيعية."
        if diff == "easy":
            return "سهل: تحية عامة فقط؛ رابط خام ظاهر وواضح التزوير؛ لا QR؛ لا مرفقات؛ طلب مباشر لكلمة المرور/بيانات الدخول؛ إلحاح اليوم؛ أخطاء واضحة قليلة؛ 7-10 أسطر حتى لا يكون قصيرًا جدًا."
        if diff == "medium":
            return "متوسط: تحية باسم أول أو لقب مهني؛ نطاق قريب من الرسمي لكن قابل للكشف؛ لا QR؛ عنصر تقني واحد فقط مثل زر أو PDF بسيط أو رابط مراجعة؛ طلب غير مباشر؛ مهلة 24-72 ساعة؛ خطأ خفيف واحد."
        return "صعب: اسم كامل ولقب دقيق؛ نبرة احترافية بلا تهديد؛ لا طلب كلمة مرور مباشر؛ مرفق رسمي إلزامي؛ QR أو زر رسمي حسب السيناريو؛ نطاق شبه رسمي لا يحتوي secure/update/verify/login/reset؛ تفاصيل يومية دقيقة."
    if not phishing:
        return "Legitimate: official hospital.org or moh.gov.sa domain only; no password request; no threat; no suspicious external link; normal professional tone."
    if diff == "easy":
        return "Easy: generic greeting only; raw visible obviously fake URL; no QR; no attachment; direct password/credential request; same-day urgency; a few obvious errors; 7-10 readable lines so it is not too short."
    if diff == "medium":
        return "Intermediate: first name or professional title; look-alike but detectable domain; no QR; exactly one technical element such as a simple button, PDF, or review link; indirect request; 24-72 hour deadline; one subtle error."
    return "Advanced: full name and precise title; polished professional tone with no threat; no direct password request; official attachment required; QR or official button depending on scenario; near-official domain without secure/update/verify/login/reset; detailed daily-workflow context."


def _v10_prompt(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    diff = _enhanced_diff(difficulty)
    is_ar = language == "Arabic"
    role_type, sc, sub, person = _v10_select_card(role, index, assessment)
    shape = random.choice(_V10_BODY_SHAPES_AR if is_ar else _V10_BODY_SHAPES_EN)
    channel = ("official_notice" if not is_phishing else (random.choice(_V10_MEDIUM_CHANNELS) if diff == "medium" else random.choice(_V10_HARD_CHANNELS) if diff == "hard" else "visible_raw_link"))
    request = random.choice(_V10_EASY_REQUESTS_AR if is_ar else _V10_EASY_REQUESTS_EN)
    pressure = random.choice(_V10_EASY_PRESSURE_AR if is_ar else _V10_EASY_PRESSURE_EN)
    seed = random.randint(100000, 999999)
    recipient = person[1]
    lang_rule = "اكتب كل شيء بالعربية فقط." if is_ar else "Write everything in English only."
    label = "تصيد" if is_ar and is_phishing else "شرعي" if is_ar else "PHISHING" if is_phishing else "LEGITIMATE"
    schema = """
{
  "email_type": "specific type",
  "attack_type": "specific attack or Legitimate Email",
  "risk_level": "Safe/Medium/High/Critical",
  "from": "Display Name <email@domain>",
  "to": "recipient@hospital.org",
  "subject": "subject",
  "attachment": "filename or empty string",
  "body": "full email body with paragraphs",
  "suspicious_text": "exact suspicious phrase from body or empty if legitimate",
  "suspicious_link": "exact suspicious URL or empty",
  "is_phishing": true,
  "scenario_id": "short id",
  "subrole": "subrole",
  "indicators": [
    {"number":1,"title":"specific clue","description":"must cite a visible phrase/domain/sender/request from the email"},
    {"number":2,"title":"specific clue","description":"must cite a visible phrase/domain/sender/request from the email"},
    {"number":3,"title":"specific clue","description":"must cite a visible phrase/domain/sender/request from the email"}
  ],
  "why_risky": "grounded explanation",
  "learning_tip": "short practical tip"
}
"""
    if is_ar:
        return f"""
أنت مولد محتوى توعوي بالتصيد لمستشفى سعودي. يجب أن يكون التوليد من API وليس قالبًا ثابتًا.
{lang_rule}

التصنيف المطلوب: {label}
الدور المختار: {role_type}
الدور الداخلي: {sub}
بطاقة السيناريو من قاعدة الـ300:
- الموضوع: {sc.get('topic')}
- القسم: {sc.get('dept')}
- المرسل المنطقي: {sc.get('sender')}
- المهمة: {sc.get('task')}
- المسار/المفتاح: {sc.get('path')}
المستلم: {recipient}

إطار الصعوبة الإلزامي:
{_v10_diff_contract(diff, is_phishing, is_ar)}

تنويع المحتوى الإلزامي:
- شكل الرسالة هذه المرة: {shape}
- قناة الهجوم/الإجراء: {channel}
- الطلب الحساس إذا كان سهلًا: {request}
- الضغط الزمني إذا كان سهلًا: {pressure}
- رقم تنويع: {seed}

قواعد حاسمة:
- اربط كل جملة بالموضوع الصحي المحدد. لا تجعل الرسالة كلها "حساب عام" فقط.
- المستوى السهل لا يعني رسالة قصيرة جدًا؛ اجعلها واضحة لكن فيها تفاصيل سياق صحي بسيطة.
- لا تستخدم QR أو مرفق في السهل.
- لا تستخدم QR في المتوسط.
- لا تكرر نفس بداية أو نهاية الرسالة.
- التحليل يجب أن يطابق البريد نفسه: لا تكتب تحليلًا عامًا.
- suspicious_text يجب أن يكون عبارة موجودة حرفيًا داخل body.
- suspicious_link يجب أن يكون الرابط نفسه الموجود في body إذا كان الرابط ظاهرًا.
- أخرج JSON فقط بالمخطط التالي، دون Markdown.
{schema}
"""
    return f"""
You are generating phishing-awareness content for a Saudi hospital. The content must be API-generated, not a fixed template.
{lang_rule}

Required label: {label}
Selected role: {role_type}
Internal sub-role: {sub}
Scenario card from the 300-card knowledge base:
- Topic: {sc.get('topic')}
- Department: {sc.get('dept')}
- Logical sender: {sc.get('sender')}
- Task: {sc.get('task')}
- Path/key: {sc.get('path')}
Recipient: {recipient}

Mandatory difficulty framework:
{_v10_diff_contract(diff, is_phishing, is_ar)}

Mandatory content diversity:
- Email structure this time: {shape}
- Attack/action channel: {channel}
- Sensitive request if Easy: {request}
- Pressure phrase if Easy: {pressure}
- Diversity seed: {seed}

Critical rules:
- Tie every paragraph to the specific healthcare topic. Do not make the message only a generic account notice.
- Easy does not mean too short; make it obvious but include simple healthcare context.
- No QR and no attachment in Easy.
- No QR in Intermediate.
- Do not reuse the same opening or closing style.
- The analysis must match this exact email. Do not write generic analysis.
- suspicious_text must be an exact phrase from body.
- suspicious_link must be the same URL visible in body when a raw visible link is used.
- Return JSON only using this schema. No Markdown.
{schema}
"""


def _v10_parse_or_none(data):
    try:
        if "error" in data:
            return None
        raw = data["choices"][0]["message"]["content"].strip()
        return parse_json_response(raw)
    except Exception:
        return None


def _v10_fix_result(result, role, index, language, difficulty, is_phishing, assessment=False):
    diff = _enhanced_diff(difficulty)
    is_ar = language == "Arabic"
    if not isinstance(result, dict):
        return None
    # Extract role_type BEFORE calling normalize_learning_analysis (which requires it)
    try:
        role_type, sc, sub, person = _v5_pick_full_card(role, index, assessment)
    except Exception:
        role_type, sc, sub, person = "other", {}, "Staff", ("", "staff@hospital.org")
    result = normalize_learning_analysis(result, role_type, diff, is_ar) if "indicators" in result else result
    result = clean_result(result, is_ar)
    # ensure required identity fields are present
    try:
        result["to"] = person[1]
        result.setdefault("subrole", sub)
        result.setdefault("scenario_id", f"api-v10:{role_type}:{sc.get('source_id','SC')}:{diff}:{index}:{random.randint(1000,9999)}")
    except Exception:
        pass
    result["is_phishing"] = bool(is_phishing)
    if not is_phishing:
        result["risk_level"] = "Safe"
        result.setdefault("suspicious_text", "")
        result.setdefault("suspicious_link", "")
        result.setdefault("indicators", [])
    # Easy/medium safety cleanup
    txt = "\n".join(str(result.get(k,"")) for k in ["body","attachment","suspicious_link"])
    if diff in ("easy", "medium") and "[QR" in txt:
        result["body"] = re.sub(r"\[QR[^\]]*\]", "", str(result.get("body", ""))).strip()
    if diff == "easy":
        result["attachment"] = ""
        # If link is missing, create one from the shown scenario id/body topic but keep API body mostly intact.
        link = str(result.get("suspicious_link", "")).strip()
        body = str(result.get("body", ""))
        if not link:
            slug = re.sub(r"[^a-z0-9]+", "-", str(result.get("email_type") or "hospital-update").lower()).strip("-")[:28] or "hospital-update"
            link = f"http://fake-{slug}-login.com/{slug}"
            result["suspicious_link"] = link
        if link and link not in body:
            result["body"] = _insert_before_signature(body, link)
    return result


def _v10_generate_api(role, index, language, difficulty="medium", is_phishing=True, assessment=False):
    is_ar = language == "Arabic"
    last_issues = []
    for attempt in range(3):
        prompt = _v10_prompt(role, index + attempt * 101, language, difficulty, is_phishing, assessment)
        if last_issues:
            prompt += build_retry_guidance(last_issues, is_ar)
        data = call_groq(prompt, max_tokens=3200)
        result = _v10_parse_or_none(data)
        result = _v10_fix_result(result, role, index + attempt * 101, language, difficulty, is_phishing, assessment) if result else None
        if result:
            issues = get_generation_quality_issues(result, _enhanced_diff(difficulty), bool(is_phishing))
            # For API-first mode, reject only serious structural issues; accept stylistic variety.
            severe_words = ["QR", "Attachment", "Legitimate item", "must not contain", "must include", "must show"]
            severe = [i for i in issues if any(w.lower() in i.lower() for w in severe_words)]
            if not severe:
                try: evaluate_and_log_auto_scores(result, _enhanced_diff(difficulty), language, is_phishing=bool(is_phishing))
                except Exception: pass
                return result
            last_issues = issues
    # fail-open with a clear API error instead of silently reverting to fixed templates
    return {"error": "API generation did not pass the difficulty/role guardrails. Please press Generate/Try again."}


def generate_email(role, index, language, difficulty="medium"):
    return _v10_generate_api(role, index, language, difficulty, True, assessment=False)


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    return _v10_generate_api(role, index, language, difficulty, bool(is_phishing), assessment=True)


def generate_other_email(index, language, difficulty):
    return generate_email("Other", index, language, difficulty)


def generate_other_assess_email(index, is_phishing, language, difficulty):
    return generate_assess_email("Other", index, is_phishing, language, difficulty)

# =============================================================
# END API-FIRST DYNAMIC GENERATION ENGINE v10
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

# تم الاستبدال في الأسفل
