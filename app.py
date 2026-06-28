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

def load_persistent_provider(default="groq"):
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
    try:
        with open(_RUNS_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def save_run(record):
    """Append one holistic run-rating record and persist to disk."""
    runs = load_runs()
    runs.append(record)
    try:
        with open(_RUNS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=2)
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
              ("ai_provider", load_persistent_provider("groq")),
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
    Strong 9-criterion difficulty contract.
    Goal: Beginner / Intermediate / Advanced must be visibly different in BOTH Arabic and English.
    """
    if is_ar:
        if is_phishing:
            rules = {
                "easy": """
مستوى مبتدئ — اجعل التصيد واضحًا جدًا ومناسبًا للمبتدئين عبر 9 معايير إلزامية:
1) النطاق: مزيف وواضح، بنطاق جديد بالكامل، ولا تستخدم نطاقًا سبق استعماله.
2) الأخطاء: ضع بالضبط خطأين إملائيين/لغويين واضحين في جسم الرسالة.
3) الإلحاح: تهديد مباشر وصريح خلال ساعات قليلة أو اليوم نفسه.
4) التحية: عامة فقط مثل "عزيزي الموظف" أو "Dear Staff".
5) المرسل: جهة عامة أو اسم قسم غير دقيق.
6) الطلب الحساس: طلب واضح لكلمة مرور/بيانات دخول/تحديث حساب عبر رابط.
7) المعرفة الداخلية: لا تستخدم تفاصيل داخلية حقيقية؛ الرسالة عامة.
8) التعقيد: ناقل هجوم واحد فقط: رابط أو مرفق، وليس الاثنين معًا.
9) التكتيك النفسي: خوف وإلحاح مباشر بصياغة سهلة الاكتشاف.

ممنوع في المبتدئ: التفاصيل الدقيقة جدًا، اللغة الرسمية المبالغ فيها، أو النطاقات القريبة جدًا من الرسمية.
""",
                "medium": """
مستوى متوسط — اجعل التصيد متوسط الوضوح عبر 9 معايير إلزامية:
1) النطاق: يبدو قريبًا من بيئة العمل لكنه يحتوي فرقًا واضحًا عند التدقيق، ويجب أن يكون جديدًا بالكامل.
2) الأخطاء: خطأ واحد فقط وخفيف في جسم الرسالة.
3) الإلحاح: مهني ومتوسط؛ موعد خلال 24–72 ساعة، بدون تهديد عدواني.
4) التحية: شبه مخصصة: الاسم الأول أو المسمى الوظيفي.
5) المرسل: قسم أو موظف يبدو معقولًا لكنه غير مثالي.
6) الطلب الحساس: طلب غير معتاد لكنه يبدو ممكنًا في العمل.
7) المعرفة الداخلية: تفصيل داخلي خفيف مثل اسم قسم أو نظام.
8) التعقيد: ناقل أساسي واحد مع لمسة هندسة اجتماعية.
9) التكتيك النفسي: مصداقية مهنية + ضغط زمني خفيف.

ممنوع في المتوسط: كل الحروف الكبيرة، التهديد الشديد، أو طلب كلمة المرور بشكل فاضح جدًا.
""",
                "hard": """
مستوى متقدم — اجعل التصيد شبه شرعي وصعب الاكتشاف عبر 9 معايير إلزامية:
1) النطاق: قريب بذكاء من جهة عمل/منصة صحية، لكن ليس مطابقًا للرسمي، وجديد بالكامل.
   لا تستخدم كلمات مكشوفة في النطاق مثل: secure, update, verify, login, reset, password, urgent.
2) الأخطاء: صفر أخطاء إملائية أو لغوية.
3) الإلحاح: مهذب وخفي، بدون تهديد، بدون حروف كبيرة، بدون "ACT NOW".
4) التحية: مخصصة بالاسم والمسمى/الدور المناسب.
5) المرسل: شخص أو قسم واقعي جدًا مع توقيع مهني.
6) الطلب الحساس: لا تطلب كلمة المرور مباشرة؛ اجعل الخطر عبر إجراء يبدو طبيعيًا مثل مراجعة حالة، تأكيد امتثال، أو فتح بوابة.
7) المعرفة الداخلية: تفاصيل سياقية محددة لكن غير مبالغ فيها.
8) التعقيد: يمكن دمج رابط + سياق إداري/سريري طبيعي، لكن بدون فوضى.
9) التكتيك النفسي: سلطة/ثقة/روتين مهني، وليس خوفًا مباشرًا.

ممنوع في المتقدم: التحية العامة، الأخطاء، التهديدات، النطاقات الفاضحة، أو طلب "أدخل كلمة المرور" مباشرة.
""",
            }
        else:
            rules = {
                "easy": """
رسالة شرعية سهلة:
1) نطاق رسمي فقط hospital.org أو moh.gov.sa.
2) لا أخطاء.
3) لا تهديد.
4) تحية واضحة.
5) مرسل رسمي.
6) لا طلب بيانات حساسة.
7) تفاصيل بسيطة.
8) لا روابط خارجية.
9) هدف إداري/سريري واضح وآمن.
""",
                "medium": """
رسالة شرعية متوسطة:
1) نطاق رسمي فقط.
2) لا أخطاء.
3) موعد أو إجراء طبيعي.
4) تحية شبه مخصصة.
5) مرسل مناسب.
6) لا طلب كلمة مرور أو بيانات حساسة.
7) تفاصيل عمل واقعية.
8) قد تشير للإنترانت أو رقم تحويلة، بدون رابط خارجي.
9) تبدو مهمة لكن آمنة.
""",
                "hard": """
رسالة شرعية متقدمة:
1) نطاق رسمي فقط.
2) لا أخطاء.
3) قد تكون عاجلة مهنيًا لكن بدون تهديد.
4) تحية مخصصة.
5) مرسل واقعي جدًا.
6) لا بيانات حساسة.
7) تفاصيل دقيقة ومهنية.
8) لا رابط خارجي مشبوه.
9) قد تشبه التصيد ظاهريًا لكنها آمنة تمامًا عند الفحص.
""",
            }
    else:
        if is_phishing:
            rules = {
                "easy": """
BEGINNER difficulty — make the phishing obvious through all 9 criteria:
1) Domain realism: clearly fake, brand-new, and not reused.
2) Spelling: include exactly two obvious spelling/grammar mistakes in the email body.
3) Urgency: direct threat within hours or today.
4) Greeting: generic only, such as "Dear Staff" or "Dear Team".
5) Sender credibility: vague department or suspicious generic sender.
6) Sensitive request: obvious password/credential/account-update request through a link.
7) Insider knowledge: none; generic message.
8) Attack complexity: one attack vector only: either link or attachment, not both.
9) Psychological tactic: blunt fear and urgency.

Do not make Beginner subtle, polished, or highly realistic.
""",
                "medium": """
INTERMEDIATE difficulty — mixed red flags through all 9 criteria:
1) Domain realism: workplace-plausible but imperfect, brand-new, and not reused.
2) Spelling: exactly one subtle spelling/grammar mistake in the body.
3) Urgency: moderate workplace deadline within 24–72 hours; no aggressive threats.
4) Greeting: semi-personal, using first name or role.
5) Sender credibility: plausible but not perfect.
6) Sensitive request: unusual but possible in workplace context.
7) Insider knowledge: light department/system detail.
8) Attack complexity: one main vector plus light social engineering.
9) Psychological tactic: professional credibility plus mild deadline pressure.

Do not use all-caps, severe threats, or an obviously fake domain in Intermediate.
""",
                "hard": """
ADVANCED difficulty — almost legitimate and hard to detect through all 9 criteria:
1) Domain realism: intelligently close to a healthcare/workplace service, but not official, brand-new, and not reused.
   Do NOT use obvious domain words: secure, update, verify, login, reset, password, urgent.
2) Spelling: zero spelling or grammar mistakes.
3) Urgency: polite and subtle only; no threats, no all-caps, no "ACT NOW".
4) Greeting: personalized with full name and correct role/title.
5) Sender credibility: realistic person or department with professional signature.
6) Sensitive request: do NOT ask directly for a password; make the risky action look like a normal workflow.
7) Insider knowledge: specific realistic context, not excessive.
8) Attack complexity: natural combination of link + professional process is allowed.
9) Psychological tactic: authority, trust, routine compliance, or professional responsibility.

Advanced must NOT look like Beginner. Avoid generic greeting, obvious spelling errors, obvious fake domains, direct password requests, and aggressive threats.
""",
            }
        else:
            rules = {
                "easy": """
Legitimate Beginner item:
1) official hospital.org or moh.gov.sa domain only,
2) no mistakes,
3) no urgency threat,
4) clear greeting,
5) official sender,
6) no sensitive data request,
7) simple workplace context,
8) no external link,
9) clearly safe.
""",
                "medium": """
Legitimate Intermediate item:
1) official domain only,
2) no mistakes,
3) normal deadline,
4) semi-personal greeting,
5) plausible sender,
6) no credentials/payment request,
7) realistic workflow detail,
8) may mention intranet or extension number but no external link,
9) important but safe.
""",
                "hard": """
Legitimate Advanced item:
1) official domain only,
2) no mistakes,
3) may be professionally urgent but not threatening,
4) personalized greeting,
5) realistic sender,
6) no sensitive request,
7) detailed healthcare context,
8) no suspicious external link,
9) may look important but remains safe under inspection.
""",
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
    diff_rule = get_dynamic_difficulty_rules(difficulty, is_phishing=True, is_ar=is_ar)

    if is_ar:
        return f"""
أنت مولّد أمثلة تدريبية للتوعية بالتصيد في بيئة مستشفى سعودي.

المطلوب: ولّد مثال تعلم واحد فقط لتصيد إلكتروني.

قواعد مهمة جدًا:
- لا تستخدم أي قالب ثابت.
- لا تستخدم أي نطاق من أمثلة محفوظة أو نطاقات تكررت سابقًا.
- اختر فكرة جديدة من الصفر: نظام، مرسل، سبب، رابط، رسالة، وتحليل.
- يجب أن تكون الفكرة مناسبة للدور والسياق، لكنها غير مكررة.
- ممنوع استخدام النص الحرفي: suspicious_link داخل body. ضع رابطًا حقيقي الشكل.
- أخرج JSON فقط بدون Markdown.

السياق:
{role_context}
المستلم: {recipient_email}
رقم عشوائي لكسر التكرار: {seed}
{avoid_topics}{avoid_domains}
قواعد الصعوبة:
{diff_rule}

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

Critical rules:
- Do NOT use a fixed template.
- Do NOT use memorized example domains or domains already used in this session.
- Invent a fresh scenario from scratch: system, sender, reason, domain, message, and AI analysis.
- The idea must fit the role context but must not repeat previous topics.
- Never write the literal placeholder suspicious_link inside body. Use a realistic-looking URL.
- Return JSON only. No Markdown.

Context:
{role_context}
Recipient: {recipient_email}
Anti-repeat random seed: {seed}
{avoid_topics}{avoid_domains}
Difficulty rules:
{diff_rule}

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
- اختر سيناريو جديدًا من الصفر ومناسبًا للدور.
- يجب أن يكون الاختبار متوازنًا: الرسائل الشرعية آمنة فعلًا، ورسائل التصيد فيها علامات حسب مستوى الصعوبة.
- ممنوع استخدام النص الحرفي: suspicious_link داخل body.
- أخرج JSON فقط بدون Markdown.
{official}

السياق:
{role_context}
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
- Invent a fresh scenario from scratch that fits the role.
- The assessment must be balanced: legitimate emails must be truly safe, phishing emails must show red flags according to difficulty.
- Never write the literal placeholder suspicious_link inside body. Use a realistic-looking URL when phishing needs a link.
- Return JSON only. No Markdown.
{official}

Context:
{role_context}
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
        return
    m["speed"].append(round(speed_sec, 2))
    if json_success:
        m["json_ok"] += 1
    else:
        m["json_fail"] += 1
    if content_hash and content_hash not in m["hashes"]:
        m["hashes"].append(content_hash)
    save_metrics_file(st.session_state["metrics"])

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
                        {"role": "user", "content": prompt},
                        # Prefill: forcing the assistant turn to already start
                        # with "{" makes Claude continue directly inside the
                        # JSON object instead of opening with prose/markdown,
                        # which was the main cause of unparseable replies.
                        {"role": "assistant", "content": "{"}
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
                # Re-attach the "{" we prefilled on the assistant turn — the
                # API only echoes back the continuation, not the prefix.
                text = "{" + text
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
    for attempt in range(2):
        try:
            prompt = build_prompt(role, index, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data['error'].get('message', str(data['error']))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, is_ar)
            if (result.get("suspicious_link") or "").strip() and result["suspicious_link"] not in (result.get("body") or ""):
                result["body"] = (result.get("body") or "") + "\n" + result["suspicious_link"]

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), True)
            if not last_issues or attempt == 1:
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
    for attempt in range(2):
        try:
            prompt = build_assess_prompt(role, index, is_phishing, language) + build_retry_guidance(last_issues, is_ar)
            data = call_groq(prompt, max_tokens=2400)
            if "error" in data:
                return {"error": data['error'].get('message', str(data['error']))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, is_ar)
            result["is_phishing"] = bool(is_phishing)
            if (result.get("suspicious_link") or "").strip() and result["suspicious_link"] not in (result.get("body") or ""):
                result["body"] = (result.get("body") or "") + "\n" + result["suspicious_link"]

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), is_phishing)
            if not last_issues or attempt == 1:
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
    for attempt in range(2):
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
                    result["body"] = (result.get("body") or "") + f'\n{result["suspicious_link"]}'

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), True)
            if not last_issues or attempt == 1:
                remember_generated_artifacts(role_type, "learn", result)
                return result
        except json.JSONDecodeError as e:
            if attempt == 1:
                return {"error": f"JSON parse error: {e}"}
            last_issues = [f"invalid JSON: {e}"]
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Generation failed quality checks."}


def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        return generate_other_assess_email(index, is_phishing, language, difficulty)

    is_ar = (language == "Arabic")
    last_issues = []
    for attempt in range(2):
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
                    result["body"] = (result.get("body") or "") + f'\n{result["suspicious_link"]}'

            last_issues = get_generation_quality_issues(result, st.session_state.get("difficulty", "medium"), is_phishing)
            if not last_issues or attempt == 1:
                remember_generated_artifacts(role_type, f"assess_{is_phishing}", result)
                return result
        except json.JSONDecodeError:
            if attempt == 1:
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

    # --------------------------------------------------------
    # NEW: detect a "[QR Code: label]" / "[QR: label]" placeholder
    # inside the body. Instead of deleting it, swap it for a unique
    # token that stays exactly where the model put it — so the real
    # QR image ends up rendered IN PLACE (not dumped at the very
    # bottom after the signature).
    # --------------------------------------------------------
    qr_label, has_qr = "", False
    qr_match = re.search(r'\[\s*QR(?:\s*Code)?\s*:?\s*([^\]]*)\]', body_raw, re.I)
    if qr_match:
        has_qr = True
        qr_label = qr_match.group(1).strip()
        body_raw = body_raw[:qr_match.start()] + "@@QR_TOKEN@@" + body_raw[qr_match.end():]

    # --------------------------------------------------------
    # NEW: detect a markdown-style "[Button label](https://...)"
    # link inside the body. Same idea: swap for a token so the real
    # clickable button renders IN PLACE of where the link was
    # written, not always at the bottom of the email.
    # --------------------------------------------------------
    link_label, link_url, has_link_button = "", "", False
    link_match = re.search(r'\[([^\]]{1,80})\]\s*\(\s*(https?://[^\)\s]+)\s*\)', body_raw)
    if link_match:
        has_link_button = True
        link_label = link_match.group(1).strip()
        link_url   = link_match.group(2).strip()
        body_raw   = body_raw[:link_match.start()] + "@@LINK_TOKEN@@" + body_raw[link_match.end():]

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
    # detect a standalone line that is exactly the suspicious_link,
    # strip it — the model didn't follow the no-repeat instruction,
    # but the UI shouldn't show the same link twice.
    # --------------------------------------------------------
    if (has_qr or has_link_button) and suspicious_link:
        bare_link_pattern = re.escape(suspicious_link)
        body_raw = re.sub(rf'^[ \t]*{bare_link_pattern}[ \t]*$\n?', '', body_raw, flags=re.MULTILINE)
        body_raw = re.sub(r'[ \t]*\n[ \t]*\n[ \t]*\n+', '\n\n', body_raw).strip()

    # Legacy fallback: if neither a QR nor a link-button placeholder was found,
    # keep the original behaviour of appending the raw suspicious_link as text.
    if suspicious_link and suspicious_link not in body_raw and not has_qr and not has_link_button:
        link_bare = re.sub(r'^https?://', '', suspicious_link)
        if link_bare not in body_raw:
            body_raw = body_raw.rstrip() + f'\n\n{suspicious_link}'

    has_attachment  = bool((email.get("attachment") or "").strip())

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
    link_block_html = ""
    if has_link_button:
        link_badge = make_badge(next_badge()) if show_badges else ""
        link_block_html = f"""
<div style="margin:.8rem 0;direction:{bd};">
  <a href="{html_lib.escape(link_url)}" target="_blank" rel="noopener"
     style="display:inline-flex;align-items:center;gap:.5rem;border:1px solid rgba(37,99,235,.55);
            border-radius:8px;padding:.5rem 1.1rem;background:rgba(37,99,235,.18);color:#93C5FD;
            font-size:.92rem;font-weight:700;text-decoration:none;">
    {link_badge}🔗 {html_lib.escape(link_label)}
  </a>
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
                    f'border:1px solid rgba(37,99,235,.5);border-radius:8px;padding:.4rem .8rem;'
                    f'background:rgba(37,99,235,.15);color:#93C5FD;font-size:.88rem;margin:.4rem 0;">'
                    f'{b_att}📎 {att_val}</div>')

    st.markdown(f"""
<div style="background:#0F172A;border:1px solid rgba(37,99,235,.5);
            border-radius:16px 16px 0 0;overflow:hidden;">
  <div style="background:#1E293B;padding:.6rem 1rem;display:flex;gap:8px;align-items:center;">
    <div style="width:12px;height:12px;border-radius:50%;background:#FF5F57;"></div>
    <div style="width:12px;height:12px;border-radius:50%;background:#FFBD2E;"></div>
    <div style="width:12px;height:12px;border-radius:50%;background:#28C840;"></div>
  </div>
  <div style="padding:1rem 1.6rem .5rem;font-size:.92rem;color:#CBD5E1;
              direction:{bd};text-align:{ta};
              font-family:{email_font};">
    <table style="width:100%;border-collapse:collapse;direction:{bd};">
      <tr style="vertical-align:top;">
        <td style="color:#64748B;font-weight:700;padding:0 8px 6px 0;white-space:nowrap;width:70px;">{fl}</td>
        <td style="color:#E2E8F0;padding:0 0 6px 0;word-break:break-all;">{b_from}{from_val}</td>
      </tr>
      <tr style="vertical-align:middle;">
        <td style="color:#64748B;font-weight:700;padding:0 8px 6px 0;white-space:nowrap;">{tl}</td>
        <td style="color:#93C5FD;padding:0 0 6px 0;direction:ltr;text-align:{('right' if bd=='rtl' else 'left')};overflow:hidden;text-overflow:ellipsis;">{to_val}</td>
      </tr>
      <tr style="vertical-align:top;">
        <td style="color:#64748B;font-weight:700;padding:0 8px 6px 0;white-space:nowrap;">{sl}</td>
        <td style="color:#E2E8F0;padding:0 0 6px 0;word-break:break-word;">{b_subj}{subj_val}</td>
      </tr>
    </table>
    {att_html}
  </div>
</div>
<div style="background:#0F172A;border:1px solid rgba(37,99,235,.5);border-top:none;
            border-radius:0 0 16px 16px;padding:.8rem 1.6rem 1.4rem;
            font-family:{email_font};
            font-size:.92rem;color:#CBD5E1;
            line-height:2;direction:{bd};text-align:{ta};
            box-shadow:0 20px 60px rgba(0,0,0,.5);">
  {body_html}
</div>""", unsafe_allow_html=True)


def page_home():
    is_arabic      = st.session_state["language"] == "Arabic"
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
            cur_provider = st.session_state.get("ai_provider", "groq")
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

    if st.session_state.get("cache_version",0) < 13:
        st.session_state["emails"]={}; st.session_state["cache_version"]=13

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
            st.session_state["emails"][idx] = generate_email(st.session_state["role"],idx,st.session_state["language"])
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
            st.session_state["assess_emails"][idx]=generate_assess_email(st.session_state["role"],idx,pattern[idx],st.session_state["language"])
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
        st.session_state["cache_version"] = int(__import__("time").time()) % 99999
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
    background:rgba(15,23,42,.5)!important;
    border:1px solid rgba(255,255,255,.15)!important;
    border-radius:8px!important;
    box-shadow:none!important;
}}
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

    _persist_pk = load_persistent_provider("groq")
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

        cur = st.session_state.get("ai_provider", "groq")

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
                        st.session_state["cache_version"] = int(__import__("time").time()) % 99999
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

        col_diff, col_lang = st.columns(2)
        with col_diff:
            st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#D1FAE5;margin-bottom:.5rem;">{T("difficulty_lvl")}</div>', unsafe_allow_html=True)
            diff_opts = {"easy": f"🟢 {T('easy')}", "medium": f"🟡 {T('medium')}", "hard": f"🔴 {T('hard')}"}
            cur_diff = st.session_state.get("difficulty", "medium")
            for dk, dl in diff_opts.items():
                is_d = cur_diff == dk
                label = f"{dl} ✓" if is_d else dl
                if st.button(label, key=f"adm_diff_{dk}", use_container_width=True,
                             type="primary" if is_d else "secondary"):
                    if not is_d:
                        st.session_state["difficulty"] = dk
                        st.session_state["emails"] = {}
                        st.session_state["cache_version"] = int(__import__("time").time()) % 99999
                        st.rerun()

        with col_lang:
            st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#D1FAE5;margin-bottom:.5rem;">{T("language_lbl")}</div>', unsafe_allow_html=True)
            cur_lang = st.session_state.get("language", "English")
            lang_display = {"English": "English", "Arabic": "العربية"}
            for lk in ["English", "Arabic"]:
                is_l = cur_lang == lk
                label = f"{lang_display[lk]} ✓" if is_l else lang_display[lk]
                if st.button(label, key=f"adm_lang_{lk}", use_container_width=True,
                             type="primary" if is_l else "secondary"):
                    if not is_l:
                        st.session_state["language"] = lk
                        st.session_state["emails"] = {}
                        st.rerun()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        if st.button(T('clear_cache'), use_container_width=True):
            st.session_state["emails"] = {}
            st.session_state.pop("assess_emails", None)
            st.session_state["cache_version"] = int(__import__("time").time()) % 99999
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
        for p in PROV_ORDER:
            meta = PROV_META[p]
            m = get_m(p)
            speeds = m.get("speed", [])
            total_j = m.get("json_ok",0) + m.get("json_fail",0)
            json_rate = int(m.get("json_ok",0)/total_j*100) if total_j > 0 else None
            calls = m.get("calls",0)
            err_rate = int(m.get("errors",0)/calls*100) if calls > 0 else None
            hashes = m.get("hashes",[])

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
                    all_runs.pop(i)
                    try:
                        with open(_RUNS_FILE_PATH, "w", encoding="utf-8") as f:
                            json.dump(all_runs, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
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

def get_generation_plan(role_type, is_phishing=True, is_ar=False, phase="learn"):
    if is_phishing:
        items = ATTACK_PLAYBOOK.get(role_type, ATTACK_PLAYBOOK["other"])
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
    plan = get_generation_plan(role_type, is_phishing=True, is_ar=is_ar, phase="learn")
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
- اختر نطاقًا جديدًا واقعي الشكل. في المتقدم لا تستخدم كلمات مكشوفة في النطاق.
- لا تستخدم النص الحرفي suspicious_link داخل body.
- يجب أن يكون التحليل عميقًا ومتنوعًا، وليس Domain/Urgency/Spelling دائمًا.
- أخرج JSON فقط.

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
- Invent a new realistic-looking domain. For Advanced, avoid obvious domain words.
- Never write the literal placeholder suspicious_link inside body.
- The analysis must be varied and deep, not always Domain/Urgency/Spelling.
- Return JSON only.

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
    plan = get_generation_plan(role_type, is_phishing=is_phishing, is_ar=is_ar, phase="assess")
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
        sep = "\n\n" if body.strip() else ""
        result["body"] = (body.rstrip() + sep + "[QR: Scan to continue]").strip()

    elif wants_link and not has_qr_marker and not has_btn_marker:
        link = _as_str(result.get("suspicious_link")).strip() or "https://example-training-only.invalid/verify"
        result["suspicious_link"] = link
        sep = "\n\n" if body.strip() else ""
        result["body"] = (body.rstrip() + sep + "[Open Link](" + link + ")").strip()

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
                "easy": "مبتدئ: تصيد واضح جدًا. تحية عامة، نطاق مزيف مكشوف، طلب بيانات حساس مباشر، رابط واضح أو مرفق مشبوه، تهديد مباشر، ويمكن وجود خطأين فقط.",
                "medium": "متوسط: تصيد مقنع جزئيًا. تفاصيل عمل واقعية، تحية شبه شخصية، إلحاح مهني خفيف 24-72 ساعة، مؤشران واضحان فقط، ولا تستخدم تهديدًا مبالغًا أو كلمات إنجليزية كثيرة.",
                "hard": "متقدم: تصيد قريب من الشرعي. تحية شخصية، لا أخطاء إملائية، لا تهديد مباشر، لا طلب كلمة مرور صريح، استخدم سياقًا داخليًا واقعيًا وناقلًا غير واضح مثل مرفق/QR/MFA/رد/مستند مشترك/مكالمة."
            }.get(difficulty, "متوسط")
        return {
            "easy": "شرعي مبتدئ: رسمي وواضح من hospital.org أو moh.gov.sa، لا رابط خارجي، لا بيانات حساسة، لا تهديد.",
            "medium": "شرعي متوسط: رسمي مع تفاصيل عمل واقعية وموعد طبيعي، قد يذكر الإنترانت أو التحويلة، دون طلب بيانات حساسة.",
            "hard": "شرعي متقدم: يبدو مهمًا ومهنيًا لكنه آمن؛ نطاق رسمي، تفاصيل دقيقة، لا رابط مشبوه، لا تهديد، لا بيانات دخول."
        }.get(difficulty, "شرعي متوسط")
    if is_phishing:
        return {
            "easy": "Beginner: obvious phishing. Generic greeting, obvious fake domain, direct sensitive request, clear link or risky attachment, direct threat, and at most two spelling mistakes.",
            "medium": "Intermediate: partly convincing phishing. Realistic workplace detail, semi-personal greeting, mild professional urgency of 24-72 hours, only two clear red flags, no extreme threat or heavy all-caps.",
            "hard": "Advanced: near-legitimate phishing. Personalized greeting, no spelling mistakes, no direct password request, no blunt threat, and a subtle vector such as attachment, QR, MFA, reply, shared document, phone callback, or contextual portal."
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
    {"home":page_home,"login":page_login,"learning":page_learning,"complete":page_complete,
     "assessment":page_assessment,"results":page_results,"report":page_report}.get(pg, page_home)()

# تم الاستبدال في الأسفل
