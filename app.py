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

import streamlit as st
import json
import requests
import os
import re
import html as html_lib
import random

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

for k, v in [("language","English"),("page","home"),("role",""),
              ("example_index",0),("emails",{}),("difficulty","medium"),
              ("user_name",""),("user_email",""),
              ("ai_provider", load_persistent_provider("groq")),
              ("admin_authenticated",False),
              ("metrics",{}),  # {provider: {speed:[], json_ok:int, json_fail:int, errors:int, calls:int, hashes:[]}}
              ("manual_ratings",{}),  # {provider: {quality:[], difficulty:[], arabic:[], medical:[]}}
             ]:
    if k not in st.session_state:
        st.session_state[k] = v

_nav = st.query_params.get("nav", "")
if _nav in ("login", "register"):
    st.session_state["login_mode"] = _nav
    st.session_state["page"] = "login"
    _lang = st.query_params.get("lang", "")
    if _lang in ("Arabic", "English"):
        st.session_state["language"] = _lang
    st.query_params.clear()

def set_language(lang):
    st.session_state["language"] = lang
    st.session_state["lang_explicitly_chosen"] = True

def t(en, ar):
    return ar if st.session_state["language"] == "Arabic" else en

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


# ══════════════════════════════════════════════════════════════
# STATIC TEMPLATES للـ Other Role
# 6 إيميلات جاهزة: 2 Admin + 2 IT + 2 Clinical
# اللـ LLM يولّد فقط الـ AI Analysis
# ══════════════════════════════════════════════════════════════
OTHER_TEMPLATES = [
    # 0 — ADMIN: رواتب
    {
        "type": "admin",
        "email": {
            "email_type": "Payroll Phishing",
            "from": "HR Department <noreply@hospital-hr-portal.xyz>",
            "to": "m.sultan.alghamdi@hospital.org",
            "subject": "URGENT: Your Salary Payment On Hold — Update Required",
            "attachment": "",
            "body": "Dear Staff,\n\nThis is an urgent notice from the HR Department regarding your monthly salary payment.\n\nDue to a recent banking system migration, we require all employees to re-verify their bank account details (IBAN) to ensure uninterrupted salary disbursement.\n\nYour salary payment for this month is currently ON HOLD and will not be processed until you complete the verification.\n\nPLEASE CLICK THE LINK BELOW TO UPDATE YOUR BANK DETIALS IMEDIATLY:\nhttps://hospital-salary-update.xyz/verify-iban\n\nFAILURE TO COMPLETE THIS PROCESS BEFORE END OF BUSINESS TODAY WILL RESULT IN YOUR SALARY BEING DELAYED BY 30 DAYS.\n\nBest regards,\nMohammed Al-Zahrani\nHR Manager\nnoreply@hospital-hr-portal.xyz",
            "suspicious_text": "PLEASE CLICK THE LINK BELOW TO UPDATE YOUR BANK DETIALS IMEDIATLY",
            "suspicious_link": "https://hospital-salary-update.xyz/verify-iban"
        }
    },
    # 1 — IT: Network Security
    {
        "type": "it",
        "email": {
            "email_type": "Network Credential Phishing",
            "from": "IT Security Team <noreply@hospital-netsecure.xyz>",
            "to": "t.bandar.althubaiti@hospital.org",
            "subject": "URGENT: Suspicious Login Detected — Re-Verify Your Network Account NOW",
            "attachment": "",
            "body": "Dear Staff,\n\nOur security monitoring system has detected SUSPISIOUS LOGIN ACTIVITY on your hospital network account from an unrecognized device.\n\nTo protect the hospital network and patient data, your account has been TEMPORARILY SUSPENDED until you re-verify your identity.\n\nPLEASE RE-VERIFY YOUR ACCOUNT IMEDIATELY by clicking the link below:\nhttps://hospital-netsecure.xyz/account-verify\n\nEnter your full network credentials (username and password) to restore access.\n\nACT NOW! YOUR ACCOUNT WILL BE PERMANENTLY LOCKED IF YOU DO NOT VERIFY WITHIN THE NEXT 2 HOURS.\n\nSincerely,\nIT Security Team\nnoreply@hospital-netsecure.xyz",
            "suspicious_text": "PLEASE RE-VERIFY YOUR ACCOUNT IMEDIATELY",
            "suspicious_link": "https://hospital-netsecure.xyz/account-verify"
        }
    },
    # 2 — CLINICAL: MOH Protocol
    {
        "type": "clinical",
        "email": {
            "email_type": "MOH Phishing",
            "from": "MOH Clinical Alerts <noreply@moh-clinical-updates.net>",
            "to": "dr.sarah.almutairi@hospital.org",
            "subject": "URGENT: New MOH Infection Control Protocol — Mandatory Compliance Required",
            "attachment": "",
            "body": "Dear Staff,\n\nThe Ministry of Health has issued a CRITICAL new infection control protocol following reports of increased MRSA cases across Saudi hospitals.\n\nAll clinical staff are REQUIRED to review and confirm compliance with the new protocol BEFORE END OF TODAY.\n\nACT NOW! YOUR ACCOUNT WILL BE SUSPENDED IF YOU DO NOT COMPLY.\n\nPlease urgentley click the link below to access the new protocal and enter your credintials to confirm compliance:\nhttps://moh-protocol-update.totally-fake.net/mrsa\n\nFAILURE TO COMPLY WILL RESULT IN DISCIPLINARY ACTION AND ACCOUNT CLOSURE.\n\nBest regards,\nDr. Khalid Al-Otaibi\nMOH Clinical Protocols Team\nnoreply@moh-clinical-updates.net",
            "suspicious_text": "click the link below to access the new protocal and enter your credintials",
            "suspicious_link": "https://moh-protocol-update.totally-fake.net/mrsa"
        }
    },
    # 3 — ADMIN: فاتورة مورد
    {
        "type": "admin",
        "email": {
            "email_type": "Invoice Fraud",
            "from": "Gulf Medical Supplies <billing@gulf-med-supplies.xyz>",
            "to": "m.reem.alsabiei@hospital.org",
            "subject": "URGENT: Medical Equipment Invoice SAR 125,000 — Payment Required TODAY",
            "attachment": "Medical_Invoice_GMS_2024.pdf",
            "body": "Dear Staff,\n\nPlease find attached the invoice for the recent delivery of ICU monitoring equipment (Order #GMS-4521).\n\nTotal amount due: SAR 125,000\nPayment deadline: TODAY\n\nWE HAVE RECIVED NOTIFCATION that your payment is overdue. Failure to pay by end of business today will result in IMMEDIATE SUSPENSION of your supply contract and legal action.\n\nTo verify the invoice and process payment, please login to our supplier portal:\nhttps://pay.gulf-med-supplies.xyz/login?inv=GMS4521\n\nPlease enter your username and password to complete the payment process.\n\nBest regards,\nAhmed Al-Farsi\nBilling Department\nGulf Medical Supplies",
            "suspicious_text": "WE HAVE RECIVED NOTIFCATION that your payment is overdue",
            "suspicious_link": "https://pay.gulf-med-supplies.xyz/login?inv=GMS4521"
        }
    },
    # 4 — IT: SSL Certificate
    {
        "type": "it",
        "email": {
            "email_type": "SSL Certificate Phishing",
            "from": "IT Department <noreply@hospital-ssl-renewal.net>",
            "to": "t.rania.almalki@hospital.org",
            "subject": "CRITICAL: Staff Intranet SSL Certificate EXPIRED — Immediate Action Required",
            "attachment": "",
            "body": "Dear Staff,\n\nYour atention is requierd IMEDIATLY!\n\nOur IT Department has detected that the SSL certificate for the Hospital Staff Intranet has EXPIRED TODAY.\n\nAll staff access to the intranet will be TERMINATED unless the certificate is renewed within the next hour.\n\nPlease click the link below to renew your SSL certificate and re-enter your login credintials:\nhttps://hospital-ssl-renewal.net/renew-cert\n\nYou will need to provide your FULL CREDENTIALS including username and password to complete the renewal process.\n\nSINCERELY,\nIT Department\nnoreply@hospital-ssl-renewal.net",
            "suspicious_text": "re-enter your login credintials",
            "suspicious_link": "https://hospital-ssl-renewal.net/renew-cert"
        }
    },
    # 5 — CLINICAL: EMR Credentials
    {
        "type": "clinical",
        "email": {
            "email_type": "EMR Credential Phishing",
            "from": "EMR Support Team <support@hospital-emr-secure.xyz>",
            "to": "n.noura.alshamri@hospital.org",
            "subject": "URGENT: EMR System Migration — Re-Verify Your Login Credintials NOW",
            "attachment": "",
            "body": "Dear Staff,\n\nOUR EMR SYSTEM IS UNDERGOING CRITICAL MIGRATION AND YOUR LOGIN CREDINTIALS ARE NO LONGER VALID.\n\nYou must re-verify your account IMEDIATLY to maintain access to patient records.\n\nPLEASE CLICK THIS LINK TO UPDATE YOUR ACCOUNT:\nhttp://emr-secure-update.totally-fake.net/verify-now\n\nIF YOU DO NOT TAKE ACTION NOW, YOUR ACCOUNT WILL BE CLOSED AND YOU WILL NOT BE ABLE TO ACCESS ANY PATIENT RECORDS.\n\nACT NOW! DEADLINE TODAY! THESE CHANGES ARE URGANT.\n\nPLEASE ENTER YOUR FULL CREDENTIALS TO VERIFY YOUR ACCOUNT.\n\nSincerely,\nEMR Support Team\nsupport@hospital-emr-secure.xyz",
            "suspicious_text": "PLEASE ENTER YOUR FULL CREDENTIALS TO VERIFY YOUR ACCOUNT",
            "suspicious_link": "http://emr-secure-update.totally-fake.net/verify-now"
        }
    },
]

# Static templates للـ Assessment (Other) — 5 phishing + 5 legit
OTHER_ASSESS_PHISHING = [
    # Admin
    {
        "email_type": "Payroll Phishing",
        "from": "Payroll System <noreply@hr-payroll-update.xyz>",
        "to": "m.abdullah.alqahtani@hospital.org",
        "subject": "ACTION REQUIRED: Verify Your Bank Details to Receive Salary",
        "attachment": "",
        "body": "Dear Staff,\n\nYour salary for this month cannot be processed due to a system update.\n\nPlease verify your IBAN and bank details via the link below before end of day to avoid salary delay:\nhttps://hr-payroll-update.xyz/verify\n\nFAILURE TO COMPLY WILL RESULT IN 30-DAY SALARY DELAY.\n\nHR Department",
        "suspicious_text": "verify your IBAN and bank details via the link below",
        "suspicious_link": "https://hr-payroll-update.xyz/verify",
        "explanation": "Legitimate HR departments never ask employees to verify bank details via email links. The domain hr-payroll-update.xyz is not an official hospital domain."
    },
    # IT
    {
        "email_type": "VPN Phishing",
        "from": "IT Security <noreply@hospital-vpn-update.net>",
        "to": "t.yusuf.aljuhani@hospital.org",
        "subject": "URGENT: Your VPN Access Will Be Terminated in 1 Hour",
        "attachment": "",
        "body": "Dear Staff,\n\nYour FortiClient VPN access is about to EXPIRE. You must re-authenticate IMEDIATLY.\n\nClick here to update your VPN credintials:\nhttps://hospital-vpn-update.net/reauth\n\nACT NOW OR LOSE NETWORK ACCESS TODAY.\n\nIT Security Team",
        "suspicious_text": "Click here to update your VPN credintials",
        "suspicious_link": "https://hospital-vpn-update.net/reauth",
        "explanation": "The domain hospital-vpn-update.net is fake. Legitimate VPN re-authentication is done through official IT channels, not email links."
    },
    # Clinical
    {
        "email_type": "EMR Phishing",
        "from": "EMR Support <support@emr-secure.net>",
        "to": "dr.sarah.almutairi@hospital.org",
        "subject": "URGENT: Access Your EMR Credintials NOW",
        "attachment": "",
        "body": "Dear Staff,\n\nOUR EMR SYSTEM REQUIRES YOU TO RE-VERIFY YOUR CREDINTIALS IMMEDIATELY.\n\nPLEASE CLICK THIS LINK TO UPDATE YOUR ACCOUNT:\nhttp://emr-secure.net/verify-now\n\nACT NOW! DEADLINE TODAY!\n\nEMR Support Team",
        "suspicious_text": "PLEASE CLICK THIS LINK TO UPDATE YOUR ACCOUNT",
        "suspicious_link": "http://emr-secure.net/verify-now",
        "explanation": "The domain emr-secure.net is not the official hospital EMR system. Legitimate EMR systems never ask for credentials via email."
    },
    # Admin
    {
        "email_type": "Invoice Fraud",
        "from": "MedSupply Co. <billing@medsupply-invoices.xyz>",
        "to": "m.hind.alrashidi@hospital.org",
        "subject": "OVERDUE: Medical Supply Invoice SAR 98,000 — Pay NOW",
        "attachment": "Invoice_MedSupply_2024.pdf",
        "body": "Dear Staff,\n\nYour payment for medical supplies (Order #MS-7823) is OVERDUE.\n\nAmount: SAR 98,000\n\nLogin to process payment:\nhttps://medsupply-invoices.xyz/pay\n\nEnter your credentials to complete payment. Account will be suspended if not paid TODAY.\n\nMedSupply Billing Team",
        "suspicious_text": "Enter your credentials to complete payment",
        "suspicious_link": "https://medsupply-invoices.xyz/pay",
        "explanation": "Legitimate suppliers do not ask for login credentials to process invoices. The domain medsupply-invoices.xyz is not an official supplier domain."
    },
    # IT
    {
        "email_type": "SSL Phishing",
        "from": "IT Dept <noreply@hospital-ssl-cert.net>",
        "to": "t.lama.alumari@hospital.org",
        "subject": "CRITICAL: Hospital Website SSL Certificate Expired TODAY",
        "attachment": "",
        "body": "Dear Staff,\n\nThe hospital website SSL certificate has EXPIRED TODAY.\n\nPlease renew it IMEDIATLY by clicking:\nhttps://hospital-ssl-cert.net/renew\n\nYou must enter your full admin credintials to complete renewal.\n\nIT Department",
        "suspicious_text": "enter your full admin credintials to complete renewal",
        "suspicious_link": "https://hospital-ssl-cert.net/renew",
        "explanation": "SSL certificates are renewed by IT administrators through official channels, not via email links. The domain hospital-ssl-cert.net is fake."
    },
]

OTHER_ASSESS_LEGIT = [
    # Clinical
    {
        "email_type": "Legitimate",
        "from": "Head Nurse <nurse.head@hospital.org>",
        "to": "n.noura.alshamri@hospital.org",
        "subject": "Next Week Shift Schedule Update",
        "attachment": "",
        "body": "Dear Staff,\n\nPlease find below the updated shift schedule for next week.\n\nMonday: Dr. Al-Zahrani (0800-1600), Nurse Al-Otaibi (0800-1600)\nTuesday: Dr. Al-Rashidi (0800-1600), Nurse Al-Harbi (1200-2000)\n\nIf you have any conflicts, please contact me by Thursday.\n\nBest regards,\nFatima Al-Zahrani\nHead Nurse\nfatima.alzahrani@hospital.org",
        "suspicious_link": "",
        "explanation": "This is a legitimate shift schedule email. It comes from an official @hospital.org address, contains no suspicious links, and makes no requests for credentials."
    },
    # Admin
    {
        "email_type": "Legitimate",
        "from": "HR Department <hr@hospital.org>",
        "to": "m.sultan.alghamdi@hospital.org",
        "subject": "Upcoming Mandatory Fire Safety Training — June 22",
        "attachment": "",
        "body": "Dear Staff,\n\nThis is a reminder that mandatory Fire Safety and Emergency Evacuation training will take place on June 22, 2024 from 9:00 AM to 12:00 PM in Conference Room B.\n\nAll administrative staff are required to attend.\n\nFor any questions, please contact the HR department at hr@hospital.org or ext. 1234.\n\nBest regards,\nMohammed Al-Hussain\nHR Department\nhr@hospital.org",
        "suspicious_link": "",
        "explanation": "This is a legitimate training notification. It comes from the official @hospital.org domain, provides clear training details, and does not request any sensitive information."
    },
    # IT
    {
        "email_type": "Legitimate",
        "from": "IT Department <it@hospital.org>",
        "to": "t.rania.almalki@hospital.org",
        "subject": "Scheduled Server Maintenance — Saturday 3:00 AM to 6:00 AM",
        "attachment": "",
        "body": "Dear Staff,\n\nWe would like to inform you that scheduled server maintenance will take place this Saturday between 3:00 AM and 6:00 AM.\n\nDuring this period, the following services will be temporarily unavailable:\n- EMR system\n- Staff intranet\n- Email (limited)\n\nNo action is required from your end. All systems will be restored automatically after maintenance.\n\nFor urgent support during maintenance, contact the IT helpdesk at ext. 5555.\n\nBest regards,\nIT Department\nit@hospital.org",
        "suspicious_link": "",
        "explanation": "This is a legitimate IT maintenance notice. It comes from the official @hospital.org domain, provides specific maintenance details, and requires no action or credentials from the recipient."
    },
    # Clinical
    {
        "email_type": "Legitimate",
        "from": "Infection Control Team <ic@hospital.org>",
        "to": "dr.sarah.almutairi@hospital.org",
        "subject": "Updated Hand Hygiene Guidelines — Q2 2024",
        "attachment": "",
        "body": "Dear Clinical Staff,\n\nThe Infection Control Team has updated the Hand Hygiene guidelines in line with the latest MOH recommendations for Q2 2024.\n\nKey updates:\n- Alcohol-based hand rub to be used before and after every patient contact\n- Glove usage protocols updated for ICU and oncology wards\n- New hand hygiene audit schedule starting July 1\n\nPlease review the updated guidelines on the hospital intranet under Infection Control > Guidelines.\n\nFor questions, contact the Infection Control Team at ic@hospital.org.\n\nBest regards,\nDr. Sara Al-Mahmoud\nInfection Control Officer\nic@hospital.org",
        "suspicious_link": "",
        "explanation": "This is a legitimate infection control update. It comes from the official @hospital.org domain, refers staff to the official intranet, and requests no credentials or personal information."
    },
    # Admin
    {
        "email_type": "Legitimate",
        "from": "Procurement Department <procurement@hospital.org>",
        "to": "m.reem.alsabiei@hospital.org",
        "subject": "Medical Supply Order #MS-2024-089 Confirmed and Dispatched",
        "attachment": "",
        "body": "Dear Staff,\n\nWe are pleased to confirm that your medical supply order has been processed and dispatched.\n\nOrder details:\n- Order Number: MS-2024-089\n- Items: Surgical gloves, sterile dressings, IV catheters\n- Estimated delivery: 3-5 business days\n\nIf you have any questions about your order, please contact the Procurement Department at procurement@hospital.org or ext. 2200.\n\nBest regards,\nSarah Al-Otaibi\nProcurement Officer\nprocurement@hospital.org",
        "suspicious_link": "",
        "explanation": "This is a legitimate order confirmation. It comes from the official @hospital.org domain, provides order details, and does not request any sensitive information or credentials."
    },
]


def get_recipient(role, index, language):
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    pool = EN_NAMES.get(role_type, EN_NAMES["clinical"])
    return pool[index % len(pool)]

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
    ],
    "it": [
        {"en": "SCENARIO: Fake VPN credential update — claim the hospital VPN gateway requires urgent re-authentication. IMPORTANT: vary the VPN system name (Cisco AnyConnect/FortiClient/Pulse Secure), the suspicious portal URL, and the urgency reason each time.", "ar": "السيناريو: تحديث مزيف لبيانات الـ VPN. مهم: غيّر اسم النظام (Cisco/FortiClient) والرابط والسبب في كل مرة."},
        {"en": "SCENARIO: Fake SSL certificate expiry — claim the hospital website or portal SSL certificate has expired. IMPORTANT: vary the affected system (hospital website/patient portal/EMR login/staff intranet), renewal deadline, and suspicious link each time.", "ar": "السيناريو: تنبيه مزيف بانتهاء شهادة SSL. مهم: غيّر النظام المتأثر (موقع/بوابة/EMR) والموعد والرابط في كل مرة."},
        {"en": "SCENARIO: Fake IT helpdesk remote access — impersonate IT helpdesk claiming a critical server issue requires remote access credentials immediately.", "ar": "السيناريو: مكتب مساعدة مزيف يطلب بيانات الوصول عن بُعد لحل مشكلة خادم حرجة."},
        {"en": "SCENARIO: CIO impersonation — impersonate the Chief Information Officer urgently requesting server admin credentials or asking to disable security settings.", "ar": "السيناريو: انتحال هوية مدير تقنية المعلومات يطلب بيانات الخادم أو تعطيل إعدادات الأمان."},
        {"en": "SCENARIO: Fake software license renewal — claim a critical hospital software license is expiring in 24 hours and requires immediate renewal via a suspicious portal.", "ar": "السيناريو: تجديد مزيف لترخيص برنامج حيوي ينتهي خلال 24 ساعة."},
        {"en": "SCENARIO: Fake firewall policy update — send a malicious Word document claiming to contain a new mandatory firewall security policy requiring macro enablement.", "ar": "السيناريو: سياسة جدار ناري مزيفة — مستند Word يطلب تفعيل الماكرو."},
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

# FIX 1: build_prompt — upgraded to llama-3.3-70b-versatile
# and enhanced difficulty rules with more detail
# =============================================================
def build_prompt(role, index, language):
    is_ar      = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info  = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    role_desc, role_ctx, role_type = role_info
    seed = st.session_state.get("cache_version", 13)
    import time
    session_seed = abs(hash(str(seed) + str(index) + str(time.time()))) % 99999

    role_guidance = {
        "clinical": (
            "Doctors, nurses, pharmacists, lab technicians, radiologists in a Saudi hospital.",
            "EMR systems, patient records, lab results, clinical schedules, pharmacy, medical devices, "
            "surgery lists, vaccination records, ICU data, telemedicine, clinical protocols, MOH alerts, "
            "medical training, infection control, blood bank, patient transfers.",
            "Choose freely: credential theft link, malicious PDF/Excel/Word attachment, executive impersonation, "
            "fake MOH/hospital alert, fake medical system update — MUST be medical/clinical content only."
        ),
        "admin": (
            "Medical secretaries, receptionists, patient records clerks, insurance coordinators, "
            "billing specialists, procurement officers, hospital administrators in Saudi healthcare.",
            "Patient appointments, medical records, health insurance claims, hospital billing, "
            "medical procurement, supplier invoices, staff policies, MOH compliance, accreditation, "
            "patient registration, treatment authorizations, surgery scheduling.",
            "Choose freely: fake appointment/insurance/billing portal link, malicious patient records PDF/Excel, "
            "doctor/CEO impersonation, fake MOH audit, fake supplier invoice — MUST be healthcare admin content only."
        ),
        "it": (
            "IT specialists, informatics officers, system administrators, cybersecurity staff in a Saudi hospital.",
            "Hospital network, VPN, servers, EMR system, cloud backup, SSL certificates, firewall, "
            "software licenses, IT helpdesk, endpoint security, database administration, network monitoring.",
            "Choose freely: VPN/cloud/helpdesk credential theft link, malicious IT policy PDF/Excel, "
            "CIO/CISO impersonation, fake SSL/firewall/license alert — MUST be healthcare IT content only."
        ),
        "other": (
            "A general hospital employee in Saudi Arabia (could be any department).",
            "Any hospital area: clinical (patient records, EMR), administrative (billing, insurance, payroll), or IT (network, systems, helpdesk).",
            "Use the MANDATORY SCENARIO provided — it rotates across all three role types for maximum variety."
        ),
    }

    # للـ Other: نحدد وظيفة دقيقة + قسم محدد + بريد مناسب لكل index
    OTHER_JOB_PROFILES = [
        # 0 — Admin
        {
            "r_desc": "a hospital billing and insurance coordinator (administrative staff)",
            "r_ctx": "payroll, IBAN updates, health insurance (Tawuniya/Bupa/AXA), supplier invoices, procurement, hospital billing, staff HR notifications",
            "r_guidance": "Generate an ADMINISTRATIVE phishing email. Topic MUST be one of: fake payroll/IBAN update, fake health insurance verification, fake supplier invoice, fake CEO financial request, fake HR notification. NEVER use clinical or IT topics.",
            "recipient": "m.sultan.alghamdi@hospital.org"
        },
        # 1 — IT
        {
            "r_desc": "a hospital network and systems support technician (IT staff)",
            "r_ctx": "VPN access, hospital network accounts, SSL certificates, firewall, software licenses, IT helpdesk tickets, server maintenance, cybersecurity alerts",
            "r_guidance": "Generate an IT/TECHNICAL phishing email. Topic MUST be one of: fake VPN re-authentication, fake SSL certificate expiry, fake network security alert, fake software license renewal, fake IT helpdesk request. NEVER use clinical or administrative topics.",
            "recipient": "t.bandar.althubaiti@hospital.org"
        },
        # 2 — Clinical
        {
            "r_desc": "a hospital pharmacist or lab technician (clinical staff)",
            "r_ctx": "pharmacy dispensing system, lab results portal, patient medication records, clinical protocols, MOH drug circulars, infection control updates",
            "r_guidance": "Generate a CLINICAL phishing email. Topic MUST be one of: fake pharmacy/lab system credential update, fake MOH clinical protocol alert, fake patient records access, fake medical director request. NEVER use administrative or IT topics.",
            "recipient": "dr.khalid.alanazi@hospital.org"
        },
        # 3 — Admin
        {
            "r_desc": "a hospital medical procurement and supply chain officer (administrative staff)",
            "r_ctx": "medical equipment procurement, supplier contracts, purchase orders, delivery confirmations, inventory management, MOH procurement compliance",
            "r_guidance": "Generate an ADMINISTRATIVE phishing email. Topic MUST be one of: fake urgent supplier invoice (SAR amount), fake procurement portal login, fake contract renewal, fake supply order confirmation with malicious attachment. NEVER use clinical or IT topics.",
            "recipient": "m.reem.alsabiei@hospital.org"
        },
        # 4 — IT
        {
            "r_desc": "a hospital cybersecurity and systems administrator (IT staff)",
            "r_ctx": "hospital firewall, server administration, database backups, endpoint security, Active Directory, cloud backup systems, EMR server maintenance",
            "r_guidance": "Generate an IT/TECHNICAL phishing email. Topic MUST be one of: fake CIO/CISO urgent server request, fake firewall/security policy update, fake cloud backup credential alert, fake Active Directory password expiry, fake database admin request. NEVER use clinical or administrative topics.",
            "recipient": "t.nadia.alsalmi@hospital.org"
        },
        # 5 — Clinical
        {
            "r_desc": "a hospital radiologist or medical imaging technician (clinical staff)",
            "r_ctx": "PACS imaging system, radiology reports, patient scan results, imaging department scheduling, MOH radiology protocols, medical imaging equipment",
            "r_guidance": "Generate a CLINICAL phishing email. Topic MUST be one of: fake PACS system credential update, fake urgent patient scan results PDF, fake radiology department alert, fake MOH imaging protocol update. NEVER use administrative or IT topics.",
            "recipient": "dr.fahad.aldosari@hospital.org"
        },
    ]

    _dummy_guidance = {
    }
    if role_type == "other":
        profile = OTHER_JOB_PROFILES[index % len(OTHER_JOB_PROFILES)]
        r_desc     = profile["r_desc"]
        r_ctx      = profile["r_ctx"]
        r_guidance = profile["r_guidance"]
        # override الـ recipient للـ other حسب الـ profile
        st.session_state[f"_other_recipient_{index}"] = profile["recipient"]
    else:
        r_desc, r_ctx, r_guidance = role_guidance.get(role_type, role_guidance["clinical"])

    # FIX 3: Enhanced difficulty rules — more detailed for both languages
    if is_ar:
        diff_rules = {
            "easy": (
                "مستوى مبتدئ — العلامات يجب أن تكون واضحة جداً ولا تخطئها:\n"
                "- نطاق مزيف واضح تماماً (مثل hosp1tal-updates.xyz أو hospital.totally-fake.net أو secur3-login.com)\n"
                "- خطأين إملائيين واضحين على الأقل في نص الرسالة (مثل: 'تسجيل الدخوول' أو 'عزيزي الموظفف')\n"
                "- إلحاح مبالغ فيه بعبارات تحذيرية كبيرة (تصرف الآن! سيتم إغلاق حسابك خلال ساعة! موعد نهائي اليوم!)\n"
                "- تحية عامة فقط: 'عزيزي الموظف' أو 'عزيزي المستخدم' — ممنوع استخدام الاسم أو المسمى الوظيفي\n"
                "- طلب صريح ومشبوه جداً (شارك كلمة المرور، أدخل بياناتك الكاملة، أرسل رقم الهوية)\n"
                "- عنوان المرسل واضح الزيف (مثل: noreply@hospital-secure.xyz)"
            ),
            "medium": (
                "مستوى متوسط — صعوبة معتدلة، بعض العلامات واضحة وبعضها يحتاج تمعّناً:\n"
                "- نطاق مشبوه نسبياً لكن ليس واضح الزيف تماماً (مثل hospital-hr-portal.net أو moh-notifications.com)\n"
                "- أسلوب شبه مهني مع علامة تحذيرية واحدة أو اثنتين في الصياغة\n"
                "- خطأ إملائي واحد بسيط أو جملة غير طبيعية في السياق\n"
                "- إلحاح معتدل ('يرجى الرد بنهاية الأسبوع' أو 'يجب التحديث قبل يوم الاثنين')\n"
                "- تحية شبه شخصية (اللقب الوظيفي صح لكن الاسم أحياناً خاطئ أو عام)\n"
                "- الطلب غير عادي لكن ليس مستحيلاً في بيئة العمل"
            ),
            "hard": (
                "مستوى متقدم — العلامات خفية جداً، الرسالة تبدو حقيقية تقريباً:\n"
                "- نطاق يشبه الحقيقي مع تغيير بسيط جداً لا يُلاحَظ بسهولة (مثل hosp1tal.org أو hospital-sa.net أو moh.gov-sa.com)\n"
                "- لغة عربية فصحى مهنية سليمة تماماً، صفر أخطاء إملائية أو نحوية\n"
                "- إلحاح خفيف ومهني جداً ('نرجو الاطلاع قبل نهاية يوم العمل' أو 'للحفاظ على أمان حسابك')\n"
                "- تحية بالاسم الكامل والمسمى الوظيفي الدقيق\n"
                "- علامة تحذيرية واحدة فقط وخفية للغاية — كل شيء آخر يبدو حقيقياً تماماً\n"
                "- المحتوى ذو صلة مباشرة بعمل المستلم ويوحي بمعرفة داخلية"
            ),
        }
    else:
        diff_rules = {
            "easy": (
                "BEGINNER difficulty — red flags must be VERY obvious and easy to spot:\n"
                "- Clearly fake domain (e.g. hosp1tal-updates.xyz, hospital.totally-fake.net, secur3-login.com)\n"
                "- At least 2 obvious spelling/grammar mistakes in the body text\n"
                "- Aggressive ALL-CAPS urgency with alarming language (ACT NOW! YOUR ACCOUNT WILL BE CLOSED! DEADLINE TODAY!)\n"
                "- Generic greeting only: 'Dear Staff' or 'Dear User' — never use recipient's name or job title\n"
                "- Blatantly suspicious request (share your password, enter full credentials, send your ID number)\n"
                "- Sender address obviously fake (e.g. noreply@hospital-secure.xyz)"
            ),
            "medium": (
                "INTERMEDIATE difficulty — some flags obvious, some require careful reading:\n"
                "- Slightly suspicious domain that looks almost real (e.g. hospital-hr-portal.net, moh-notifications.com)\n"
                "- Mostly professional tone with 1-2 red flags in wording\n"
                "- One minor spelling error or awkward sentence that feels slightly off\n"
                "- Moderate urgency with a deadline ('Please respond by end of week' or 'Update required before Monday')\n"
                "- Semi-personal greeting — correct job title but name is generic or slightly wrong\n"
                "- Request is unusual but not impossible in a workplace context"
            ),
            "hard": (
                "ADVANCED difficulty — red flags extremely subtle, email looks almost completely legitimate:\n"
                "- Nearly real domain with only one tiny character change (e.g. hosp1tal.org, hospital-sa.net, moh.gov-sa.com)\n"
                "- Perfect professional English, zero spelling or grammar errors\n"
                "- Subtle, polite urgency only ('Kindly review before end of business day' or 'To keep your account secure')\n"
                "- Personalised greeting with full name and exact job title\n"
                "- Only ONE subtle red flag — everything else looks completely legitimate\n"
                "- Content directly relevant to recipient's work, implying insider knowledge"
            ),
        }

    diff_rule = diff_rules.get(difficulty, diff_rules["medium"])

    if is_ar:
        lang_rule = (
            "اللغة: عربية فصحى فقط في كل النصوص (subject/body/indicators/why_risky/learning_tip).\n"
            "استثناء: عناوين البريد الإلكتروني والروابط (http://...) تبقى لاتينية.\n"
            "ممنوع: أي حرف لاتيني داخل النصوص العربية.\n"
            "حقل 'to': البريد الإلكتروني فقط بدون أي نص."
        )
        from_ex  = "اسم المرسل <fake@domain.com>"
        body_ex  = "نص الرسالة بالعربية الفصحى"
        ind_t_ex = "عنوان المؤشر"
        ind_d_ex = "وصف تقني تفصيلي"
    else:
        lang_rule = "Language: English only throughout. No Arabic or foreign characters in text fields."
        from_ex  = "Sender Name <fake@domain.com>"
        body_ex  = "email body in English"
        ind_t_ex = "indicator title"
        ind_d_ex = "detailed technical explanation"

    # FIX 7: Get forced scenario for this index
    forced = FORCED_SCENARIOS.get(role_type, FORCED_SCENARIOS["admin"])
    forced_scenario = forced[index % len(forced)]
    scenario_instruction = forced_scenario["ar"] if is_ar else forced_scenario["en"]

    return f"""You are a cybersecurity expert creating phishing awareness training for Saudi healthcare.

TRAINING EXAMPLE #{index + 1} of 6 | Variety seed: {session_seed}

━━━ TARGET ━━━
Role: {r_desc}
Context: {r_ctx}

━━━ YOUR TASK — MANDATORY SCENARIO ━━━
You MUST generate this EXACT scenario type — do NOT substitute or change it:
{scenario_instruction}

This scenario is NON-NEGOTIABLE. Generate the email body, subject, and sender to match this specific scenario exactly.

━━━ DIFFICULTY ━━━
{diff_rule}

━━━ LANGUAGE ━━━
{lang_rule}

━━━ FORMAT RULES ━━━
- body: plain text only, use \\n for line breaks, NO HTML. Keep it concise: 100-180 words maximum.
- "to": email address only, nothing else
- If attack uses a link: put URL in "suspicious_link" AND verbatim in body
- If attack uses attachment: put filename in "attachment" (e.g. file.pdf, data.xlsx)
- If social engineering only: "suspicious_link":"", "attachment":""
- Each indicator "description": ONE short sentence, max ~20 words
- "why_risky": maximum 2 short sentences
- "learning_tip": ONE short, practical sentence
- Be concise everywhere. Do not pad any field with extra explanation beyond what's asked.

━━━ RETURN ONLY VALID JSON ━━━
CRITICAL: Output ONLY the JSON. No text before or after.
{{"email_type":"attack type name","from":"{from_ex}","to":"employee@hospital.org","subject":"subject line","attachment":"filename or empty","body":"{body_ex}","suspicious_text":"most suspicious phrase","suspicious_link":"url or empty","indicators":[{{"number":1,"title":"{ind_t_ex}","description":"{ind_d_ex}"}},{{"number":2,"title":"{ind_t_ex}","description":"{ind_d_ex}"}},{{"number":3,"title":"{ind_t_ex}","description":"{ind_d_ex}"}}],"why_risky":"why dangerous for this role","learning_tip":"practical tip for this role"}}"""

# =============================================================
# FIX 2 + FIX 3: build_assess_prompt — tokens raised to 1200,
# difficulty rules expanded to match build_prompt detail level
# =============================================================
def build_assess_prompt(role, index, is_phishing, language):
    is_ar      = (language == "Arabic")
    difficulty = st.session_state.get("difficulty", "medium")
    role_info  = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    role_desc, role_ctx, role_type = role_info
    seed = st.session_state.get("cache_version", 13)
    import time
    session_seed = abs(hash(str(seed) + str(index) + str(is_phishing) + str(time.time()))) % 99999

    role_guidance = {
        "clinical": (
            "a nurse, doctor, pharmacist, or lab technician in a Saudi hospital",
            "EMR, patient records, lab results, pharmacy, clinical schedules, MOH alerts, medical devices, surgery"
        ),
        "admin": (
            "a medical secretary, receptionist, patient records clerk, insurance coordinator, or billing specialist in Saudi healthcare",
            "patient appointments, medical records, health insurance, hospital billing, medical procurement, MOH compliance"
        ),
        "it": (
            "an IT specialist, system administrator, or cybersecurity officer in a Saudi hospital",
            "hospital network, VPN, servers, EMR system, cloud backup, SSL, firewall, software licenses, IT helpdesk"
        ),
        "other": (
            "a general hospital employee in Saudi Arabia (any department)",
            "clinical areas (EMR, patient records), administrative tasks (billing, insurance, payroll), or IT systems (network, helpdesk)"
        ),
    }
    r_desc, r_ctx = role_guidance.get(role_type, role_guidance["other"])

    # FIX 3+10: diff_rules مطابقة للتعلم — مخصصة لكل دور
    # ══════════════════════════════════════════════════════
    # diff_rules — مطابقة لقسم التعلم تماماً
    # مخصصة لكل دور في كل مستوى (AR + EN)
    # ══════════════════════════════════════════════════════
    role_domains = {
        "admin":    {"easy": "hosp1tal-hr.xyz / moh-pay.net",
                     "medium": "hospital-hr-portal.net / moh-billing.com",
                     "hard":   "moh.gov-sa.com / hosp1tal.org"},
        "clinical": {"easy": "emr-secure.xyz / medrecords.net",
                     "medium": "emr-health-sa.net / moh-clinic.com",
                     "hard":   "hosp1tal-clinic.org / moh.gov.sa-health.com"},
        "it":       {"easy": "vpn-update.xyz / sysadmin-alert.net",
                     "medium": "vpn-hospital-sa.net / itsupport-moh.com",
                     "hard":   "hosp1tal-it.org / moh-itsupport.sa.com"},
        "other":    {"easy": "hospital-alert.xyz / hosp1tal-secure.net",
                     "medium": "hospital-portal-sa.net / moh-staff.com",
                     "hard":   "hosp1tal.org / moh.gov-sa.com"},
    }
    rd = role_domains.get(role_type, role_domains["admin"])

    if is_ar:
        diff_rules = {
            "easy": (
                "مستوى مبتدئ — العلامات يجب أن تكون واضحة جداً ولا تخطئها:\n"
                f"- نطاق مزيف واضح تماماً مناسب للدور (مثل {rd['easy']})\n"
                "- خطأين إملائيين واضحين على الأقل في نص الرسالة\n"
                "- إلحاح مبالغ فيه بعبارات كبيرة (تصرف الآن! حسابك سيُغلق! موعد نهائي اليوم!)\n"
                "- تحية عامة فقط: 'عزيزي الموظف' أو 'عزيزي الفريق' — ممنوع الاسم\n"
                "- طلب صريح ومشبوه جداً (شارك كلمة المرور، أدخل بياناتك كاملة)\n"
                "- عنوان المرسل واضح الزيف"
            ),
            "medium": (
                "مستوى متوسط — صعوبة معتدلة، بعض العلامات تحتاج تمعّناً:\n"
                f"- نطاق مشبوه نسبياً مناسب للدور (مثل {rd['medium']})\n"
                "- أسلوب شبه مهني مع علامة تحذيرية واحدة أو اثنتين فقط\n"
                "- خطأ إملائي واحد بسيط فقط في النص\n"
                "- إلحاح معتدل ('يرجى الرد بنهاية الأسبوع') — ممنوع ALL CAPS\n"
                "- تحية شبه شخصية (اللقب صح لكن الاسم أحياناً عام أو خاطئ)\n"
                "- الطلب غير عادي لكن ليس مستحيلاً في بيئة العمل"
            ),
            "hard": (
                "مستوى متقدم — العلامات خفية جداً، الرسالة تبدو حقيقية تقريباً:\n"
                f"- نطاق يشبه الحقيقي مع تغيير بسيط جداً مناسب للدور (مثل {rd['hard']})\n"
                "- لغة عربية فصحى مهنية سليمة تماماً، صفر أخطاء إملائية\n"
                "- صفر ALL CAPS — أسلوب مهني هادئ تماماً\n"
                "- إلحاح خفيف ومهني فقط ('نرجو الاطلاع قبل نهاية يوم العمل')\n"
                "- تحية بالاسم الكامل والمسمى الوظيفي الدقيق\n"
                "- علامة تحذيرية واحدة فقط وخفية — كل شيء آخر يبدو حقيقياً تماماً"
            ),
        }
    else:
        diff_rules = {
            "easy": (
                "BEGINNER difficulty — red flags VERY obvious and easy to spot:\n"
                f"- Clearly fake domain suited to the role (e.g. {rd['easy']})\n"
                "- At least 2 obvious spelling/grammar mistakes in the body\n"
                "- Aggressive ALL-CAPS urgency (ACT NOW! DEADLINE TODAY! ACCOUNT WILL BE CLOSED!)\n"
                "- Generic greeting only: 'Dear Staff' or 'Dear Team' — NEVER use name\n"
                "- Blatantly suspicious request (share password, enter full credentials)\n"
                "- Sender address obviously fake"
            ),
            "medium": (
                "INTERMEDIATE difficulty — some flags obvious, some need careful reading:\n"
                f"- Slightly suspicious domain suited to role (e.g. {rd['medium']})\n"
                "- Mostly professional tone with 1-2 red flags only\n"
                "- EXACTLY 1 minor spelling mistake — just one subtle error\n"
                "- Moderate urgency only: 'Please respond by end of week' — NO ALL-CAPS\n"
                "- Semi-personal greeting matching role (correct title, name slightly off)\n"
                "- Request unusual but not impossible in workplace context"
            ),
            "hard": (
                "ADVANCED difficulty — red flags extremely subtle, email looks almost completely real:\n"
                f"- Nearly real domain with ONE tiny change suited to role (e.g. {rd['hard']})\n"
                "- ZERO spelling or grammar mistakes — perfect professional language\n"
                "- ZERO ALL-CAPS — completely normal professional tone throughout\n"
                "- Subtle polite urgency only: 'Kindly review before end of business day'\n"
                "- Full name + exact job title matching the role in greeting\n"
                "- ONLY ONE subtle red flag (the domain) — everything else perfectly legitimate"
            ),
        }
    diff_rule = diff_rules.get(difficulty, diff_rules["medium"])

    # تعريف task_p و task_l — يُستخدمان كـ "Additional context" في الـ prompt
    if is_ar:
        task_p = f"ولّد رسالة تصيد إلكتروني واقعية تستهدف {r_desc}. اتبع السيناريو الإجباري أعلاه بدقة."
        task_l = f"ولّد بريد إلكتروني شرعي وطبيعي من بيئة عمل {r_desc}. استخدم نطاق رسمي (@hospital.org أو @moh.gov.sa). لا علامات تصيد إطلاقاً."
    else:
        task_p = f"Generate a realistic phishing email targeting {r_desc}. Follow the MANDATORY SCENARIO above exactly."
        task_l = f"Generate a realistic legitimate workplace email for {r_desc}. Use official domain (@hospital.org or @moh.gov.sa). Zero suspicious elements — must look completely normal."

    task = task_p if is_phishing else task_l

    # FIX 7b: Forced scenario for assessment based on index + is_phishing
    # ══════════════════════════════════════════════
    # ASSESSMENT SCENARIOS — مخصصة لكل دور مع تنويع
    # ══════════════════════════════════════════════
    assess_scenarios = {
        "admin": {
            True: [
                "MANDATORY PHISHING — Admin/Billing: Fake supplier invoice for medical equipment. Vary: supplier name (MedSupply Co./Gulf Medical/Al-Rashid Medical), equipment type (surgical instruments/lab supplies/radiology equipment/ICU monitors), invoice amount (SAR 75,000–200,000), PDF filename. Target: billing or procurement admin.",
                "MANDATORY PHISHING — Admin/Insurance: Fake health insurance portal — re-verify staff coverage. Vary: provider name (Tawuniya/Bupa Arabia/MedGulf/AXA), claim type (annual renewal/coverage update/reimbursement), suspicious link URL. Target: insurance coordinator.",
                "MANDATORY PHISHING — Admin/HR: Fake payroll system — salary on hold until bank details updated. Vary: bank detail type (IBAN/account number/branch code), deadline (end of month/within 48h/before 15th), sender name. Target: HR or billing staff.",
                "MANDATORY PHISHING — Admin/Executive: Hospital CEO or Director impersonation — urgent financial transfer or sensitive payroll data request. Vary: director name, amount, urgency reason. Pure social engineering, no link needed. Target: admin manager.",
                "MANDATORY PHISHING — Admin/Procurement: Fake medical procurement portal — supplier contract must be renewed via suspicious link. Vary: supplier type, contract value, deadline, suspicious URL. Target: procurement officer.",
            ],
            False: [
                "MANDATORY LEGITIMATE — Admin: Routine weekly patient appointment schedule reminder from department head. Official @hospital.org sender. No links, no requests, no urgency.",
                "MANDATORY LEGITIMATE — Admin/HR: Upcoming mandatory staff training notice (fire safety/CPR/MOH compliance). Official @hospital.org sender. Informational only.",
                "MANDATORY LEGITIMATE — Admin/Procurement: Approved medical supply order confirmed and dispatched. Official @hospital.org sender. No suspicious links.",
                "MANDATORY LEGITIMATE — Admin/Payroll: Monthly payslip notification from official HR system. Official @hospital.org sender. Standard routine notification.",
                "MANDATORY LEGITIMATE — Admin: Departmental meeting invitation from manager about next week. Official @hospital.org sender. Normal business communication.",
            ],
        },
        "clinical": {
            True: [
                "MANDATORY PHISHING — Clinical/EMR: Fake EMR credential harvest. Vary: system name (EMR/Patient Portal/Clinical System/HealthRecord), suspicious link URL, ONE spelling mistake (credintials OR urgant OR acces OR imediatly — never use recived). Address to Dr. or Nurse.",
                "MANDATORY PHISHING — Clinical/PDF: Malicious patient lab results PDF. Vary: patient department (ICU/oncology/cardiology/radiology/pediatrics), patient case reference, PDF filename (patient_results_XXXX.pdf), doctor name. Include one spelling mistake.",
                "MANDATORY PHISHING — Clinical/MOH: Fake MOH clinical protocol requiring immediate link click. Vary: protocol topic (infection control/COVID update/MRSA alert/vaccination/antimicrobial resistance), MOH official name, suspicious link URL.",
                "MANDATORY PHISHING — Clinical/Impersonation: Medical director or chief of staff impersonation — urgent patient data or system credentials request. Vary: director name, specialty (Surgery/Internal Medicine/Emergency/Oncology), specific request. Pure social engineering.",
                "MANDATORY PHISHING — Clinical/Excel: Malicious clinical duty roster Excel. Vary: schedule period (next month/Ramadan/Q2/holiday coverage), head nurse name, Excel filename. Request to enable macros.",
            ],
            False: [
                "MANDATORY LEGITIMATE — Clinical: Next week shift schedule update from head nurse. Official @hospital.org sender. No links, no requests. Normal clinical communication.",
                "MANDATORY LEGITIMATE — Clinical: Patient case review reminder for ward round or MDT meeting. Official @hospital.org sender. Routine clinical workflow, no suspicious elements.",
                "MANDATORY LEGITIMATE — Clinical/MOH: MOH mandatory training reminder (CPD/BLS/infection control). Official @moh.gov.sa or @hospital.org sender. Informational only.",
                "MANDATORY LEGITIMATE — Clinical: Updated infection control guidelines from infection control team. Official @hospital.org sender. Policy update only, no links.",
                "MANDATORY LEGITIMATE — Clinical: Department meeting invitation from medical director about clinical protocols. Official @hospital.org sender. Normal professional communication.",
            ],
        },
        "it": {
            True: [
                "MANDATORY PHISHING — IT/VPN: Fake VPN re-authentication alert. Vary: VPN name (Cisco AnyConnect/FortiClient/Pulse Secure/GlobalProtect), suspicious portal URL, urgency reason (security update/certificate renewal/mandatory re-auth). Target: IT specialist.",
                "MANDATORY PHISHING — IT/SSL: Fake SSL certificate expiry for hospital system. Vary: affected system (hospital website/patient portal/EMR login/staff intranet/lab system), renewal deadline, suspicious link URL. Target: system administrator.",
                "MANDATORY PHISHING — IT/Helpdesk: Fake helpdesk ticket requesting remote access or credentials. Vary: ticket reference number, reported issue (server outage/network fault/EMR performance), requester name. Target: IT helpdesk staff.",
                "MANDATORY PHISHING — IT/CIO: CIO or CISO impersonation — urgent server credentials or disable security settings. Vary: executive name, specific system (firewall/server/database), urgency reason. Pure social engineering. Target: IT specialist.",
                "MANDATORY PHISHING — IT/License: Fake software license renewal portal. Vary: software name (antivirus/EMR/Windows Server/database license), expiry urgency (24h/end of day/this week), suspicious renewal URL. Target: IT admin.",
            ],
            False: [
                "MANDATORY LEGITIMATE — IT: Scheduled server maintenance notice for next weekend. Official @hospital.org sender. Informational only, no credentials needed.",
                "MANDATORY LEGITIMATE — IT: Software update announcement for hospital systems (antivirus/Windows/EMR patch). Official @hospital.org sender. Standard IT notification.",
                "MANDATORY LEGITIMATE — IT: Network upgrade scheduled notification from IT department. Official @hospital.org sender. Informational, no action required.",
                "MANDATORY LEGITIMATE — IT/Helpdesk: IT helpdesk ticket resolution confirmation — issue resolved. Official @hospital.org sender. Closing notification only.",
                "MANDATORY LEGITIMATE — IT: Cybersecurity awareness training reminder for IT staff. Official @hospital.org or @moh.gov.sa sender. Training schedule only.",
            ],
        },
        "other": {
            True: [
                "MANDATORY PHISHING — Mixed/Admin: Fake supplier invoice for medical equipment — urgent payment request. Vary: supplier name, equipment type, invoice amount (SAR 50,000–150,000), PDF filename. Target: general hospital employee.",
                "MANDATORY PHISHING — Mixed/Clinical: Fake hospital system login credential harvest. Vary: system name (EMR/staff portal/scheduling system), suspicious URL, ONE spelling mistake. Target: general hospital employee.",
                "MANDATORY PHISHING — Mixed/IT: Fake hospital network or cybersecurity alert — urgent credential update. Vary: alert type (security breach/VPN expiry/account lockout), suspicious portal URL. Target: general hospital employee.",
                "MANDATORY PHISHING — Mixed/Admin: Fake payroll notification — salary on hold until bank details updated. Vary: bank detail type, urgency deadline, sender name. Target: general hospital employee.",
                "MANDATORY PHISHING — Mixed/Clinical: Fake MOH health directive — immediate acknowledgment via link. Vary: directive topic (vaccination/safety/compliance), MOH official name, suspicious URL. Target: general hospital employee.",
            ],
            False: [
                "MANDATORY LEGITIMATE — Mixed: Routine weekly work schedule update from department head. Official @hospital.org sender. No suspicious elements.",
                "MANDATORY LEGITIMATE — Mixed/HR: Staff training reminder from HR (safety/compliance/professional development). Official @hospital.org sender. Informational only.",
                "MANDATORY LEGITIMATE — Mixed: Hospital policy update notice from administration. Official @hospital.org sender. No links, no requests.",
                "MANDATORY LEGITIMATE — Mixed/Payroll: Monthly payslip notification from official HR system. Official @hospital.org sender. Standard notification.",
                "MANDATORY LEGITIMATE — Mixed: Team meeting or briefing invitation from manager. Official @hospital.org sender. Normal workplace communication.",
            ],
        },
    }
    role_assess = assess_scenarios.get(role_type, assess_scenarios["admin"])
    phish_list  = role_assess[True]
    legit_list  = role_assess[False]

    # FIX 8: حساب الـ rank الصحيح بدلاً من index الكلي
    # نحتاج نعرف "هذا الـ phishing/legit الثاني أم الثالث؟"
    # نستخدم assess_pattern من session_state إذا متوفر
    pattern = st.session_state.get("assess_pattern", [])
    if pattern and index < len(pattern):
        if is_phishing:
            # عدد الأسئلة الـ phishing قبل هذا السؤال
            rank = sum(1 for i in range(index) if pattern[i] == True)
        else:
            # عدد الأسئلة الـ legit قبل هذا السؤال
            rank = sum(1 for i in range(index) if pattern[i] == False)
    else:
        rank = index

    if is_phishing:
        forced_task_raw = phish_list[rank % len(phish_list)]
    else:
        forced_task_raw = legit_list[rank % len(legit_list)]

    # FIX 9: ترجمة forced_task حسب اللغة
    ASSESS_TRANSLATIONS = {
        # ── Admin phishing ──────────────────────────────────────
        "MANDATORY PHISHING — Admin/Billing: Fake supplier invoice for medical equipment. Vary: supplier name (MedSupply Co./Gulf Medical/Al-Rashid Medical), equipment type (surgical instruments/lab supplies/radiology equipment/ICU monitors), invoice amount (SAR 75,000–200,000), PDF filename. Target: billing or procurement admin.":
            "إجباري تصيد — إداري/فواتير: فاتورة مورد معدات طبية مزيفة. غيّر: اسم المورد (MedSupply/الخليج الطبي/الرشيد الطبي)، نوع المعدات (أجهزة جراحية/مستلزمات مختبر/أجهزة تصوير/أجهزة ICU)، المبلغ (75,000–200,000 ريال)، اسم ملف PDF. الهدف: موظف فوترة أو مشتريات.",
        "MANDATORY PHISHING — Admin/Insurance: Fake health insurance portal — re-verify staff coverage. Vary: provider name (Tawuniya/Bupa Arabia/MedGulf/AXA), claim type (annual renewal/coverage update/reimbursement), suspicious link URL. Target: insurance coordinator.":
            "إجباري تصيد — إداري/تأمين: بوابة تأمين صحي مزيفة لإعادة التحقق من التغطية. غيّر: اسم شركة التأمين (التعاونية/بوبا/ميدغلف/AXA)، نوع الطلب (تجديد/تحديث تغطية/استرداد)، رابط مشبوه. الهدف: منسق تأمين.",
        "MANDATORY PHISHING — Admin/HR: Fake payroll system — salary on hold until bank details updated. Vary: bank detail type (IBAN/account number/branch code), deadline (end of month/within 48h/before 15th), sender name. Target: HR or billing staff.":
            "إجباري تصيد — إداري/رواتب: نظام رواتب مزيف — الراتب موقوف حتى تحديث البيانات البنكية. غيّر: نوع البيانات (IBAN/رقم الحساب/رمز الفرع)، الموعد النهائي (نهاية الشهر/48 ساعة/قبل الـ15)، اسم المرسل. الهدف: موظف HR أو فوترة.",
        "MANDATORY PHISHING — Admin/Executive: Hospital CEO or Director impersonation — urgent financial transfer or sensitive payroll data request. Vary: director name, amount, urgency reason. Pure social engineering, no link needed. Target: admin manager.":
            "إجباري تصيد — إداري/مدير: انتحال هوية المدير التنفيذي — طلب تحويل مالي عاجل أو بيانات رواتب حساسة. غيّر: اسم المدير، المبلغ، سبب الاستعجال. هندسة اجتماعية بحتة. الهدف: مدير إداري.",
        "MANDATORY PHISHING — Admin/Procurement: Fake medical procurement portal — supplier contract must be renewed via suspicious link. Vary: supplier type, contract value, deadline, suspicious URL. Target: procurement officer.":
            "إجباري تصيد — إداري/مشتريات: بوابة مشتريات طبية مزيفة — تجديد عقد مورد عبر رابط مشبوه. غيّر: نوع المورد، قيمة العقد، الموعد النهائي، الرابط المشبوه. الهدف: مسؤول مشتريات.",
        # ── Admin legit ─────────────────────────────────────────
        "MANDATORY LEGITIMATE — Admin: Routine weekly patient appointment schedule reminder from department head. Official @hospital.org sender. No links, no requests, no urgency.":
            "إجباري شرعي — إداري: تذكير روتيني أسبوعي بجدول مواعيد المرضى من رئيس القسم. مرسل رسمي @hospital.org. بدون روابط أو طلبات.",
        "MANDATORY LEGITIMATE — Admin/HR: Upcoming mandatory staff training notice (fire safety/CPR/MOH compliance). Official @hospital.org sender. Informational only.":
            "إجباري شرعي — إداري/موارد بشرية: إشعار تدريب إلزامي قادم (سلامة/إسعافات/امتثال). مرسل رسمي @hospital.org. للإعلام فقط.",
        "MANDATORY LEGITIMATE — Admin/Procurement: Approved medical supply order confirmed and dispatched. Official @hospital.org sender. No suspicious links.":
            "إجباري شرعي — إداري/مشتريات: تأكيد اعتماد وشحن طلب توريد طبي. مرسل رسمي @hospital.org. بدون روابط مشبوهة.",
        "MANDATORY LEGITIMATE — Admin/Payroll: Monthly payslip notification from official HR system. Official @hospital.org sender. Standard routine notification.":
            "إجباري شرعي — إداري/رواتب: إشعار راتب شهري من نظام الموارد البشرية الرسمي. مرسل رسمي @hospital.org. إشعار روتيني عادي.",
        "MANDATORY LEGITIMATE — Admin: Departmental meeting invitation from manager about next week. Official @hospital.org sender. Normal business communication.":
            "إجباري شرعي — إداري: دعوة اجتماع قسم من المدير للأسبوع القادم. مرسل رسمي @hospital.org. تواصل عمل عادي.",
        # ── Clinical phishing ───────────────────────────────────
        "MANDATORY PHISHING — Clinical/EMR: Fake EMR credential harvest. Vary: system name (EMR/Patient Portal/Clinical System/HealthRecord), suspicious link URL, ONE spelling mistake (credintials OR urgant OR acces OR imediatly — never use recived). Address to Dr. or Nurse.":
            "إجباري تصيد — سريري/EMR: سرقة بيانات نظام السجلات الطبية. غيّر: اسم النظام (EMR/بوابة المريض/النظام السريري)، الرابط المشبوه، خطأ إملائي واحد (اختر: تسجيـل/عاجلة/وصلت). خاطب الدكتور أو الممرض.",
        "MANDATORY PHISHING — Clinical/PDF: Malicious patient lab results PDF. Vary: patient department (ICU/oncology/cardiology/radiology/pediatrics), patient case reference, PDF filename (patient_results_XXXX.pdf), doctor name. Include one spelling mistake.":
            "إجباري تصيد — سريري/PDF: مرفق PDF خبيث لنتائج مختبر مريض. غيّر: القسم (ICU/أورام/قلب/أشعة/أطفال)، رقم الحالة، اسم ملف PDF، اسم الطبيب. ضمّن خطأً إملائياً واحداً.",
        "MANDATORY PHISHING — Clinical/MOH: Fake MOH clinical protocol requiring immediate link click. Vary: protocol topic (infection control/COVID update/MRSA alert/vaccination/antimicrobial resistance), MOH official name, suspicious link URL.":
            "إجباري تصيد — سريري/وزارة: بروتوكول سريري مزيف من وزارة الصحة يستلزم نقر رابط فوري. غيّر: موضوع البروتوكول (مكافحة عدوى/كوفيد/MRSA/تطعيمات)، اسم المسؤول، الرابط المشبوه.",
        "MANDATORY PHISHING — Clinical/Impersonation: Medical director or chief of staff impersonation — urgent patient data or system credentials request. Vary: director name, specialty (Surgery/Internal Medicine/Emergency/Oncology), specific request. Pure social engineering.":
            "إجباري تصيد — سريري/انتحال: انتحال هوية المدير الطبي — طلب عاجل لبيانات مرضى أو بيانات دخول الأنظمة. غيّر: اسم المدير، التخصص (جراحة/باطنية/طوارئ/أورام)، الطلب المحدد.",
        "MANDATORY PHISHING — Clinical/Excel: Malicious clinical duty roster Excel. Vary: schedule period (next month/Ramadan/Q2/holiday coverage), head nurse name, Excel filename. Request to enable macros.":
            "إجباري تصيد — سريري/Excel: جدول مناوبات سريري مزيف كملف Excel خبيث. غيّر: الفترة (رمضان/الربع الثاني/الإجازات)، اسم رئيسة التمريض، اسم الملف. اطلب تفعيل الماكرو.",
        # ── Clinical legit ──────────────────────────────────────
        "MANDATORY LEGITIMATE — Clinical: Next week shift schedule update from head nurse. Official @hospital.org sender. No links, no requests. Normal clinical communication.":
            "إجباري شرعي — سريري: تحديث جدول المناوبة للأسبوع القادم من رئيسة التمريض. مرسل رسمي @hospital.org. بدون روابط أو طلبات.",
        "MANDATORY LEGITIMATE — Clinical: Patient case review reminder for ward round or MDT meeting. Official @hospital.org sender. Routine clinical workflow, no suspicious elements.":
            "إجباري شرعي — سريري: تذكير بمراجعة حالة مريض لجولة الزيارة أو اجتماع الفريق. مرسل رسمي @hospital.org. روتين سريري عادي.",
        "MANDATORY LEGITIMATE — Clinical/MOH: MOH mandatory training reminder (CPD/BLS/infection control). Official @moh.gov.sa or @hospital.org sender. Informational only.":
            "إجباري شرعي — سريري/وزارة: تذكير التدريب الإلزامي من وزارة الصحة (CPD/BLS/مكافحة عدوى). مرسل رسمي @moh.gov.sa. للإعلام فقط.",
        "MANDATORY LEGITIMATE — Clinical: Updated infection control guidelines from infection control team. Official @hospital.org sender. Policy update only, no links.":
            "إجباري شرعي — سريري: تحديث إرشادات مكافحة العدوى من الفريق المختص. مرسل رسمي @hospital.org. تحديث سياسة فقط.",
        "MANDATORY LEGITIMATE — Clinical: Department meeting invitation from medical director about clinical protocols. Official @hospital.org sender. Normal professional communication.":
            "إجباري شرعي — سريري: دعوة اجتماع قسم من المدير الطبي لمناقشة البروتوكولات. مرسل رسمي @hospital.org. تواصل مهني عادي.",
        # ── IT phishing ─────────────────────────────────────────
        "MANDATORY PHISHING — IT/VPN: Fake VPN re-authentication alert. Vary: VPN name (Cisco AnyConnect/FortiClient/Pulse Secure/GlobalProtect), suspicious portal URL, urgency reason (security update/certificate renewal/mandatory re-auth). Target: IT specialist.":
            "إجباري تصيد — تقني/VPN: تنبيه إعادة مصادقة VPN مزيف. غيّر: اسم الـ VPN (Cisco AnyConnect/FortiClient/Pulse Secure)، الرابط المشبوه، سبب الاستعجال. الهدف: متخصص تقنية معلومات.",
        "MANDATORY PHISHING — IT/SSL: Fake SSL certificate expiry for hospital system. Vary: affected system (hospital website/patient portal/EMR login/staff intranet/lab system), renewal deadline, suspicious link URL. Target: system administrator.":
            "إجباري تصيد — تقني/SSL: انتهاء شهادة SSL مزيف لنظام المستشفى. غيّر: النظام المتأثر (الموقع/بوابة المريض/EMR/الإنترانت)، الموعد النهائي، الرابط المشبوه. الهدف: مدير النظام.",
        "MANDATORY PHISHING — IT/Helpdesk: Fake helpdesk ticket requesting remote access or credentials. Vary: ticket reference number, reported issue (server outage/network fault/EMR performance), requester name. Target: IT helpdesk staff.":
            "إجباري تصيد — تقني/مكتب المساعدة: تذكرة مكتب مساعدة مزيفة تطلب وصولاً عن بُعد أو بيانات دخول. غيّر: رقم التذكرة، المشكلة المُبلَّغة (انقطاع الخادم/عطل الشبكة/أداء EMR)، اسم مقدم الطلب.",
        "MANDATORY PHISHING — IT/CIO: CIO or CISO impersonation — urgent server credentials or disable security settings. Vary: executive name, specific system (firewall/server/database), urgency reason. Pure social engineering. Target: IT specialist.":
            "إجباري تصيد — تقني/مدير: انتحال هوية مدير تقنية المعلومات — بيانات خادم عاجلة أو تعطيل إعدادات أمان. غيّر: اسم المدير، النظام المحدد (جدار ناري/خادم/قاعدة بيانات)، سبب الاستعجال.",
        "MANDATORY PHISHING — IT/License: Fake software license renewal portal. Vary: software name (antivirus/EMR/Windows Server/database license), expiry urgency (24h/end of day/this week), suspicious renewal URL. Target: IT admin.":
            "إجباري تصيد — تقني/ترخيص: بوابة تجديد ترخيص برنامج مزيفة. غيّر: اسم البرنامج (مضاد الفيروسات/EMR/Windows Server)، مدى الإلحاح (24 ساعة/نهاية اليوم/هذا الأسبوع)، الرابط المشبوه.",
        # ── IT legit ────────────────────────────────────────────
        "MANDATORY LEGITIMATE — IT: Scheduled server maintenance notice for next weekend. Official @hospital.org sender. Informational only, no credentials needed.":
            "إجباري شرعي — تقني: إشعار صيانة خادم مجدولة للعطلة القادمة. مرسل رسمي @hospital.org. للإعلام فقط، لا حاجة لبيانات دخول.",
        "MANDATORY LEGITIMATE — IT: Software update announcement for hospital systems (antivirus/Windows/EMR patch). Official @hospital.org sender. Standard IT notification.":
            "إجباري شرعي — تقني: إعلان تحديث برنامج لأنظمة المستشفى (مضاد الفيروسات/ويندوز/تحديث EMR). مرسل رسمي @hospital.org. إشعار تقني عادي.",
        "MANDATORY LEGITIMATE — IT: Network upgrade scheduled notification from IT department. Official @hospital.org sender. Informational, no action required.":
            "إجباري شرعي — تقني: إشعار ترقية شبكة مجدولة من قسم تقنية المعلومات. مرسل رسمي @hospital.org. للإعلام فقط.",
        "MANDATORY LEGITIMATE — IT/Helpdesk: IT helpdesk ticket resolution confirmation — issue resolved. Official @hospital.org sender. Closing notification only.":
            "إجباري شرعي — تقني/مكتب المساعدة: تأكيد حل تذكرة مكتب المساعدة. مرسل رسمي @hospital.org. إشعار إغلاق فقط.",
        "MANDATORY LEGITIMATE — IT: Cybersecurity awareness training reminder for IT staff. Official @hospital.org or @moh.gov.sa sender. Training schedule only.":
            "إجباري شرعي — تقني: تذكير تدريب الوعي الأمني لموظفي تقنية المعلومات. مرسل رسمي @hospital.org. جدول تدريب فقط.",
        # ── Other phishing ──────────────────────────────────────
        "MANDATORY PHISHING — Mixed/Admin: Fake supplier invoice for medical equipment — urgent payment request. Vary: supplier name, equipment type, invoice amount (SAR 50,000–150,000), PDF filename. Target: general hospital employee.":
            "إجباري تصيد — مختلط/إداري: فاتورة مورد طبية مزيفة — طلب دفع عاجل. غيّر: اسم المورد، نوع المعدات، المبلغ (50,000–150,000 ريال)، اسم PDF. الهدف: موظف عام.",
        "MANDATORY PHISHING — Mixed/Clinical: Fake hospital system login credential harvest. Vary: system name (EMR/staff portal/scheduling system), suspicious URL, ONE spelling mistake. Target: general hospital employee.":
            "إجباري تصيد — مختلط/سريري: سرقة بيانات دخول نظام المستشفى. غيّر: اسم النظام (EMR/بوابة الموظف/جدول المناوبات)، الرابط المشبوه، خطأ إملائي واحد. الهدف: موظف عام.",
        "MANDATORY PHISHING — Mixed/IT: Fake hospital network or cybersecurity alert — urgent credential update. Vary: alert type (security breach/VPN expiry/account lockout), suspicious portal URL. Target: general hospital employee.":
            "إجباري تصيد — مختلط/تقني: تنبيه أمني مزيف للشبكة — تحديث بيانات دخول عاجل. غيّر: نوع التنبيه (اختراق/انتهاء VPN/قفل الحساب)، الرابط المشبوه. الهدف: موظف عام.",
        "MANDATORY PHISHING — Mixed/Admin: Fake payroll notification — salary on hold until bank details updated. Vary: bank detail type, urgency deadline, sender name. Target: general hospital employee.":
            "إجباري تصيد — مختلط/إداري: إشعار راتب مزيف — موقوف حتى تحديث البيانات البنكية. غيّر: نوع البيانات البنكية، الموعد النهائي، اسم المرسل. الهدف: موظف عام.",
        "MANDATORY PHISHING — Mixed/Clinical: Fake MOH health directive — immediate acknowledgment via link. Vary: directive topic (vaccination/safety/compliance), MOH official name, suspicious URL. Target: general hospital employee.":
            "إجباري تصيد — مختلط/سريري: توجيه صحي مزيف من وزارة الصحة — تأكيد فوري عبر رابط. غيّر: موضوع التوجيه (تطعيمات/سلامة/امتثال)، اسم المسؤول، الرابط. الهدف: موظف عام.",
        # ── Other legit ─────────────────────────────────────────
        "MANDATORY LEGITIMATE — Mixed: Routine weekly work schedule update from department head. Official @hospital.org sender. No suspicious elements.":
            "إجباري شرعي — مختلط: تحديث جدول عمل أسبوعي روتيني من رئيس القسم. مرسل رسمي @hospital.org. بدون عناصر مشبوهة.",
        "MANDATORY LEGITIMATE — Mixed/HR: Staff training reminder from HR (safety/compliance/professional development). Official @hospital.org sender. Informational only.":
            "إجباري شرعي — مختلط/HR: تذكير تدريب موظفين من الموارد البشرية (سلامة/امتثال/تطوير). مرسل رسمي @hospital.org. للإعلام فقط.",
        "MANDATORY LEGITIMATE — Mixed: Hospital policy update notice from administration. Official @hospital.org sender. No links, no requests.":
            "إجباري شرعي — مختلط: إشعار تحديث سياسة المستشفى من الإدارة. مرسل رسمي @hospital.org. بدون روابط أو طلبات.",
        "MANDATORY LEGITIMATE — Mixed/Payroll: Monthly payslip notification from official HR system. Official @hospital.org sender. Standard notification.":
            "إجباري شرعي — مختلط/رواتب: إشعار راتب شهري من نظام الموارد البشرية الرسمي. مرسل رسمي @hospital.org. إشعار عادي.",
        "MANDATORY LEGITIMATE — Mixed: Team meeting or briefing invitation from manager. Official @hospital.org sender. Normal workplace communication.":
            "إجباري شرعي — مختلط: دعوة اجتماع فريق أو إحاطة من المدير. مرسل رسمي @hospital.org. تواصل عمل عادي.",
    }
    if is_ar:
        forced_task = ASSESS_TRANSLATIONS.get(forced_task_raw, forced_task_raw)
    else:
        forced_task = forced_task_raw

    # FIX 10: lang_rule مطابق للتعلم
    if is_ar:
        lang_rule = (
            "اللغة: عربية فصحى فقط في كل النصوص (subject/body/explanation).\n"
            "استثناء: عناوين البريد الإلكتروني والروابط (http://...) تبقى لاتينية.\n"
            "ممنوع: أي حرف لاتيني داخل النصوص العربية.\n"
            "حقل 'to': البريد الإلكتروني فقط بدون أي نص."
        )
    else:
        lang_rule = "Language: English only throughout. No Arabic or foreign characters in text fields. Email addresses and URLs stay Latin."

    # تعريف المتغيرات المستخدمة في JSON template
    if is_ar:
        from_ex = "اسم المرسل <email@domain.com>"
        subj_ex = "موضوع الرسالة"
        body_ex = "نص الرسالة بالعربية الفصحى"
        expl    = "اشرح بوضوح لماذا هذا البريد " + ("تصيد إلكتروني وما هي علاماته التحذيرية" if is_phishing else "شرعي وآمن وما الذي يجعله موثوقاً")
    else:
        from_ex = "Sender Name <email@domain.com>"
        subj_ex = "subject line"
        body_ex = "email body in English"
        expl    = f"Clearly explain why this email is {'phishing and identify the red flags' if is_phishing else 'legitimate and safe'}"

    return f"""Phishing awareness assessment email for Saudi healthcare. Seed:{session_seed}

TARGET: {r_desc}
CONTEXT: {r_ctx}

TASK — MANDATORY SCENARIO (do NOT change this):
{forced_task}
Additional context: {task}

DIFFICULTY: {diff_rule}

LANGUAGE: {lang_rule}

FORMAT: body=plain text only, \\n for line breaks, no HTML. "to"=email address only. Keep body concise: 100-180 words maximum. Keep "explanation" to 2-3 short sentences maximum.
IMPORTANT: never use the double-quote character (") anywhere inside any text value (subject/body/explanation/from). If you need to quote a word, use single quotes (') instead — a stray " inside a value breaks the JSON output.
{"If phishing uses a link: put URL in suspicious_link AND in body. If attachment: filename in attachment field." if is_phishing else 'suspicious_link:"", attachment:""'}
{"If legitimate: use real official domain (@hospital.org or @moh.gov.sa), no suspicious links, no urgent credential requests.' " if not is_phishing else ""}

RETURN ONLY VALID JSON:
{{"is_phishing":{"true" if is_phishing else "false"},"from":"{from_ex}","to":"employee@hospital.org","subject":"{subj_ex}","attachment":"","body":"{body_ex}","suspicious_link":"","explanation":"{expl}"}}"""

def get_system_prompt():
    """
    FIX 4+5: System prompt يُقيّد النموذج بصرامة حسب الصعوبة والـ role.
    - FIX 4: قواعد الصعوبة (Easy/Medium/Hard)
    - FIX 5: تعليمات الـ role لضمان أن التحية والمحتوى مناسبان للدور
    """
    difficulty = st.session_state.get("difficulty", "medium")
    role       = st.session_state.get("role", "Clinical")
    role_info  = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info

    # تعليمات الـ role للتحية والمحتوى
    role_greetings = {
        "clinical": (
            "TARGET ROLE: Clinical staff (nurses, doctors, pharmacists, lab technicians).\n"
            "GREETING: Use 'Dear Dr. [Name]' or 'Dear Nurse [Name]' — medical titles only.\n"
            "CONTENT: Must relate to EMR systems, patient records, clinical schedules, lab results, pharmacy, MOH medical alerts, medical device updates, or clinical protocols.\n"
            "DO NOT use administrative, billing, or IT content."
        ),
        "admin": (
            "TARGET ROLE: Administrative/management staff — pick ONE specific sub-role each time (rotate between them):\n"
            "  - Medical Secretary: manages doctor schedules, correspondence, referral letters\n"
            "  - Receptionist: patient check-in, phone calls, appointment booking\n"
            "  - Patient Records Clerk: patient files, medical history, document archiving\n"
            "  - Insurance Coordinator: health insurance claims, pre-authorizations, coverage updates\n"
            "  - Billing Specialist: invoices, payments, accounts receivable, supplier contracts\n"
            "  - Procurement Officer: medical equipment orders, supplier relationships, purchase orders\n"
            "  - Hospital Administrator: staff HR policies, MOH accreditation, budget approvals\n\n"
            "GREETING: Match the sub-role — e.g. 'Dear Ms. Reem,' / 'Dear Medical Secretary,' / 'Dear Ms. Al-Zahrani,' — NEVER use 'Dr.' or medical titles.\n\n"
            "CONTENT: Choose a DIFFERENT scenario each time — rotate through these varied attack types:\n"
            "  1. Fake health insurance portal — update employee coverage or re-submit denied claims\n"
            "  2. Fake supplier invoice — urgent payment for medical equipment delivery\n"
            "  3. Fake payroll/HR system — update bank account or salary information\n"
            "  4. Fake patient appointment system — verify login after system migration\n"
            "  5. Fake MOH accreditation request — upload required compliance documents\n"
            "  6. Fake HR policy acknowledgment — click link to confirm new leave/overtime policy\n"
            "  7. Fake medical procurement portal — renew supplier contract before expiry\n"
            "  8. CEO/director impersonation — urgent financial transfer or sensitive data request\n\n"
            "DO NOT repeat the same scenario. DO NOT use clinical (lab/pharmacy/EMR) or IT infrastructure content."
        ),
        "it": (
            "TARGET ROLE: IT/Informatics staff (IT specialist, system administrator, cybersecurity officer).\n"
            "GREETING: Use 'Dear [Name],' or 'Dear IT Team,' or 'Dear Mr./Ms. [Name]' — NOT 'Dr.'.\n"
            "CONTENT: Must relate to VPN access, network infrastructure, server maintenance, EMR system updates, SSL certificates, firewall rules, software licenses, IT helpdesk, or endpoint security.\n"
            "DO NOT use clinical or administrative content."
        ),
        "other": (
            "TARGET ROLE: General hospital employee — could be from any department.\n"
            "GREETING: Use 'Dear [Name],' or 'Dear Colleague,' — avoid specific titles like 'Dr.' unless the scenario requires it.\n"
            "CONTENT: Follow the MANDATORY SCENARIO exactly — it already specifies the department context (admin/clinical/IT). Generate content that any hospital employee could plausibly receive.\n"
            "The scenario rotates across all three role types to ensure maximum variety."
        ),
    }
    role_instruction = role_greetings.get(role_type, role_greetings["admin"])

    sys_prompts = {
        "easy": (
            "You are a cybersecurity trainer generating phishing email examples.\n\n"
            f"{role_instruction}\n\n"
            "EASY level RULES — ALL mandatory:\n"
            f"1. Use a CLEARLY FAKE domain suited to the role (admin: hosp1tal-hr.xyz / moh-pay.net | clinical: emr-secure.xyz / medrecords.net | it: vpn-update.xyz / sysadmin-alert.net)\n"
            "2. Include EXACTLY 2 obvious spelling mistakes in the body\n"
            "3. Use ALL-CAPS for at least 2 sentences — aggressive urgency\n"
            "4. Generic greeting ONLY: \'Dear Staff\' or \'Dear Team\' — NO personal name\n"
            "5. Blatant suspicious request matching the scenario (urgent payment / share password / enter credentials)\n"
            "These rules are NON-NEGOTIABLE."
        ),
        "medium": (
            "You are a cybersecurity trainer generating phishing email examples.\n\n"
            f"{role_instruction}\n\n"
            "MEDIUM level RULES — ALL mandatory:\n"
            f"1. Use a slightly suspicious domain (admin: hospital-hr-portal.net / moh-billing.com | clinical: emr-health-sa.net / moh-clinic.com | it: vpn-hospital-sa.net / itsupport-moh.com)\n"
            "2. Include EXACTLY 1 minor spelling mistake — subtle, one word only\n"
            "3. ZERO ALL-CAPS — use normal sentence case throughout\n"
            "4. Moderate urgency only: \'Please respond by end of week\' — no threatening language\n"
            "5. Semi-personal greeting matching sub-role (e.g. \'Dear Ms. Al-Zahrani,\')\n"
            "6. Unusual but plausible request for the workplace context\n"
            "These rules are NON-NEGOTIABLE. NO ALL-CAPS under any circumstances."
        ),
        "hard": (
            "You are a cybersecurity trainer generating phishing email examples.\n\n"
            f"{role_instruction}\n\n"
            "HARD level RULES — ALL mandatory:\n"
            f"1. Domain with ONE tiny change only (admin: hosp1tal.org / moh.gov-sa.com | clinical: moh.gov.sa-health.com / hosp1tal-clinic.org | it: hosp1tal-it.org / moh-itsupport.sa.com)\n"
            "2. ZERO spelling or grammar mistakes — flawless professional language\n"
            "3. ZERO ALL-CAPS — completely professional tone throughout\n"
            "4. Polite subtle urgency ONLY: \'Kindly review before end of business day\'\n"
            "5. Full name + exact job title in greeting matching the sub-role\n"
            "6. ONLY ONE subtle red flag (the domain) — everything else perfectly legitimate\n"
            "These rules are NON-NEGOTIABLE. The email must look almost completely real."
        ),
    }
    return sys_prompts.get(difficulty, sys_prompts["medium"])


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
    """Record a single API call metric"""
    _init_provider_metrics(provider)
    m = st.session_state["metrics"][provider]
    m["calls"] += 1
    if is_error:
        m["errors"] += 1
        return
    m["speed"].append(round(speed_sec, 2))
    if json_success:
        m["json_ok"] += 1
    else:
        m["json_fail"] += 1
    if content_hash and content_hash not in m["hashes"]:
        m["hashes"].append(content_hash)

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
            anthropic_max_tokens = max(max_tokens, 2400)
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
                    "messages":   [{"role": "user", "content": prompt}]
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
                    block.get("text", "")
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
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": system_prompt + "\n\n" + prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature":     0.85
                    }
                },
                timeout=60
            )
            raw = resp.json()
            elapsed = time.time() - start_time
            try:
                text = raw["candidates"][0]["content"]["parts"][0]["text"]
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
    result["from"] = clean_email_field(result.get("from",""))
    result["to"] = extract_to_email(result.get("to",""))
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
    """يولّد إيميل Other من الـ static template ويضيف AI Analysis"""
    import copy
    template = copy.deepcopy(OTHER_TEMPLATES[index % len(OTHER_TEMPLATES)])
    email_data = template["email"]
    
    # نطلب الـ AI Analysis من اللـ LLM
    try:
        prompt = build_other_analysis_prompt(email_data, language, difficulty)
        analysis = call_groq(prompt)
        
        # ندمج الإيميل الجاهز مع الـ analysis
        email_data["indicators"] = analysis.get("indicators", [
            {"number": 1, "title": "Suspicious Domain", "description": "The sender's domain is clearly fake and not affiliated with the hospital."},
            {"number": 2, "title": "Aggressive Urgency", "description": "The email uses ALL-CAPS and threatening language to pressure the recipient."},
            {"number": 3, "title": "Request for Credentials", "description": "Legitimate systems never ask for passwords or credentials via email."}
        ])
        email_data["why_risky"] = analysis.get("why_risky", "This phishing email targets hospital staff to steal sensitive credentials.")
        email_data["learning_tip"] = analysis.get("learning_tip", "Always verify the sender's domain and never click suspicious links in emails.")
        email_data["email_type"] = email_data.get("email_type", "Phishing")
        
    except Exception:
        # Fallback analysis
        email_data["indicators"] = [
            {"number": 1, "title": "Suspicious Domain", "description": "The sender domain is not the official hospital domain."},
            {"number": 2, "title": "Spelling Mistakes", "description": "The email contains obvious spelling mistakes unusual for official communications."},
            {"number": 3, "title": "Aggressive Urgency", "description": "The email uses ALL-CAPS and threats to pressure immediate action."}
        ]
        email_data["why_risky"] = "This phishing email attempts to steal hospital staff credentials."
        email_data["learning_tip"] = "Always verify the sender domain before clicking any link or sharing credentials."
    
    return email_data


def generate_other_assess_email(index, is_phishing, language, difficulty):
    """يولّد إيميل Other للاختبار من static template"""
    import copy
    if is_phishing:
        email_data = copy.deepcopy(OTHER_ASSESS_PHISHING[index % len(OTHER_ASSESS_PHISHING)])
        explanation = email_data.pop("explanation", "")
        email_data["is_phishing"] = True
    else:
        email_data = copy.deepcopy(OTHER_ASSESS_LEGIT[index % len(OTHER_ASSESS_LEGIT)])
        explanation = email_data.pop("explanation", "")
        email_data["is_phishing"] = False
    
    email_data["explanation"] = explanation
    return email_data


def generate_email(role, index, language, difficulty="medium"):
    # للـ Other: استخدم static templates مباشرة
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        return generate_other_email(index, language, difficulty)
    try:
        data = call_groq(build_prompt(role, index, language))
        if "error" in data:
            return {"error": data['error'].get('message', str(data['error']))}
        if "choices" not in data:
            return {"error": f"Unexpected API response: {str(data)[:200]}"}
        raw    = data["choices"][0]["message"]["content"].strip()
        result = parse_json_response(raw)
        result = clean_result(result, language=="Arabic")
        result["to"] = get_recipient(role, index, language)
        if result.get("suspicious_link","").strip():
            if result["suspicious_link"] not in result.get("body",""):
                result["body"] = result.get("body","") + f'\n{result["suspicious_link"]}'
        return result
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}

def generate_assess_email(role, index, is_phishing, language, difficulty="medium"):
    # للـ Other: استخدم static templates مباشرة
    role_info = ROLE_MAP.get(role, ROLE_MAP.get("Clinical"))
    _, _, role_type = role_info
    if role_type == "other":
        return generate_other_assess_email(index, is_phishing, language, difficulty)
    # FIX: max_tokens raised again (1200 -> 2200). 1200 was too tight,
    # especially for Arabic, which needs more tokens than English for the
    # same content. Hitting the limit truncated the JSON mid-response,
    # which is why every retry kept failing with the same parse error.
    for attempt in range(3):
        try:
            data = call_groq(build_assess_prompt(role, index, is_phishing, language), max_tokens=2200)
            if "error" in data:
                return {"error": data["error"].get("message", str(data["error"]))}
            result = parse_json_response(data["choices"][0]["message"]["content"].strip())
            result = clean_result(result, language=="Arabic")
            result["to"] = get_recipient(st.session_state.get("role","Clinical"), index, language)
            if result.get("suspicious_link","").strip():
                if result["suspicious_link"] not in result.get("body",""):
                    result["body"] = result.get("body","") + f'\n{result["suspicious_link"]}'
            return result
        except json.JSONDecodeError:
            if attempt == 2:
                return {"error": "Failed to parse. Please try again."}
        except Exception as e:
            return {"error": str(e)}

def render_email_window(email, is_arabic, show_badges=False):
    bd = 'rtl' if is_arabic else 'ltr'
    ta = 'right' if is_arabic else 'left'
    email_font = 'Tahoma,Arial,sans-serif' if is_arabic else "'Courier New',monospace"

    body_raw        = re.sub(r'<[^>]+>','', email.get("body",""))
    suspicious_text = re.sub(r'<[^>]+>','', email.get("suspicious_text",""))
    suspicious_link = re.sub(r'<[^>]+>','', email.get("suspicious_link","")).strip()

    body_raw = re.sub(r'suspicious_link\s*:\s*', '', body_raw, flags=re.IGNORECASE)
    body_raw = re.sub(r'suspicious_text\s*:\s*', '', body_raw, flags=re.IGNORECASE)

    if suspicious_link and suspicious_link not in body_raw:
        link_bare = re.sub(r'^https?://', '', suspicious_link)
        if link_bare not in body_raw:
            body_raw = body_raw.rstrip() + f'\n\n{suspicious_link}'

    has_attachment  = bool(email.get("attachment","").strip())

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
                    f'padding:.2rem .5rem;background:rgba(239,68,68,.08);color:#FCA5A5;">'
                    f'{make_badge(b)}{safe_s}</span>', 1)

        if suspicious_link:
            safe_l = html_lib.escape(suspicious_link)
            if safe_l in body_html:
                b = next_badge()
                body_html = body_html.replace(safe_l,
                    f'<span style="border:2px solid rgba(239,68,68,.6);border-radius:6px;'
                    f'padding:.2rem .5rem;background:rgba(239,68,68,.08);color:#60A5FA;'
                    f'text-decoration:underline;">{make_badge(b)}{safe_l}</span>', 1)
            else:
                b = next_badge()
                body_html += (f'<br><br><span style="border:2px solid rgba(239,68,68,.6);'
                              f'border-radius:6px;padding:.2rem .5rem;background:rgba(239,68,68,.08);'
                              f'color:#60A5FA;text-decoration:underline;">'
                              f'{make_badge(b)}{html_lib.escape(suspicious_link)}</span>')

    body_html = body_html.replace("\n","<br>")

    from_val = html_lib.escape(email.get("from",""))
    to_val   = html_lib.escape(email.get("to","employee@hospital.org"))
    subj_val = html_lib.escape(email.get("subject",""))
    att_val  = html_lib.escape(email.get("attachment",""))

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
    user_name    = st.session_state.get("user_name","")
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
        cur_lang  = st.session_state.get("language","")
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
            user_logged = st.session_state.get("user_name","").strip() != ""

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
        st.error(f"**Error:** {email['error']}")
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
    <span style="display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;background:#DC2626;color:white;font-size:.75rem;font-weight:800;flex-shrink:0;">{ind.get('number','')}</span>
    <span style="font-weight:700;color:#E2E8F0;font-size:.95rem;">{ind.get('title','')}</span>
  </div>
  <div style="color:#94A3B8;font-size:.9rem;line-height:1.65;{pad};direction:{row_dir};text-align:{ta2};">{ind.get('description','')}</div>
</div>"""

        st.markdown(f"""
<div class="tutor-panel">
  <div style="font-size:1.3rem;font-weight:900;color:#F8FAFC;margin-bottom:.2rem;">🎯 {t("AI Tutor Analysis","تحليل المعلم الذكي")}</div>
  <div style="color:#64748B;font-size:.85rem;margin-bottom:1.2rem;">{t("AI-guided phishing awareness","شرح توعوي بالتصيد")}</div>
  <div class="tutor-section">{t("What is suspicious?","ما هو المشبوه؟")}</div>
  {indicators_html}
  <div class="tutor-section">{t("Why is it risky?","لماذا هو خطير؟")}</div>
  <div class="tutor-text">{email.get("why_risky","")}</div>
  <div class="tutor-section">💡 {t("Learning Tip","نصيحة تعليمية")}</div>
  <div class="tip-box">{email.get("learning_tip","")}</div>
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
        st.error(f"Error: {email['error']}")
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
        exp=re.sub(r'<[^>]+>','',em.get("explanation",""))
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
        st.markdown(f'<div style="border:1px solid {bc2};border-radius:14px;padding:1.2rem 1.5rem;background:{bg2};margin-bottom:1rem;direction:{da};"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;flex-wrap:wrap;gap:.5rem;"><span style="font-weight:800;color:#E2E8F0;">{ri} {tr(f"Q{i+1}",f"س{i+1}")} — {html_lib.escape(em.get("subject",""))}</span><span style="background:{"rgba(239,68,68,.2)" if pattern[i] else "rgba(16,185,129,.2)"};color:{"#FCA5A5" if pattern[i] else "#6EE7B7"};padding:.2rem .8rem;border-radius:99px;font-size:.85rem;font-weight:700;">{ic} {tl}</span></div><div dir="{da}" style="color:#94A3B8;font-size:.9rem;line-height:1.6;direction:{da};text-align:{"right" if is_arabic else "left"};unicode-bidi:embed;">{exp}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    if st.button(tr("Go to Report →","← الانتقال للتقرير"),key="go_report"):
        st.session_state["page"]="report"; st.rerun()


def page_report():
    is_arabic=st.session_state["language"]=="Arabic"; da='rtl' if is_arabic else 'ltr'; TOTAL=10
    def tp(e,a): return a if is_arabic else e
    st.markdown(f"""<style>#MainMenu,header,footer{{visibility:hidden;}}.stApp{{background:radial-gradient(circle at top left,#0B2E68 0%,#020617 35%,#020617 100%);color:white;}}.block-container{{max-width:900px;padding-top:2rem;}}.stButton>button{{min-height:52px !important;font-weight:800 !important;border-radius:12px !important;background:rgba(15,23,42,.88) !important;color:white !important;border:1px solid rgba(37,99,235,.55) !important;width:100% !important;}}.stButton>button:hover{{background:linear-gradient(90deg,#0B4FA8,#0284C7) !important;border-color:#1EA7FF !important;}}div[style*="direction:rtl"]{{text-align:right;}}</style>""",unsafe_allow_html=True)
    answers=st.session_state.get("assess_answers",{}); pattern=st.session_state.get("assess_pattern",[True]*5+[False]*5)
    role=st.session_state.get("role",""); lang=st.session_state.get("language","English")
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
    user_name  = st.session_state.get("user_name","")
    user_email = st.session_state.get("user_email","")
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
        st.markdown(f'<div style="border:1px solid rgba(16,185,129,.35);border-radius:14px;padding:1.2rem;background:rgba(16,185,129,.05);direction:{da};text-align:{"right" if is_arabic else "left"};"><div style="font-weight:800;color:#F1F5F9;margin-bottom:.8rem;text-align:{"right" if is_arabic else "left"};">💪 {tp("Strengths","نقاط القوة")}</div>{si}</div>',unsafe_allow_html=True)
    with s2:
        ai2="".join([f'<div style="color:#FCA5A5;margin-bottom:.4rem;text-align:{"right" if is_arabic else "left"};">⚠️ {a}</div>' for a in areas]) or f'<div style="color:#94A3B8;">{tp("Great work!","عمل رائع!")}</div>'
        st.markdown(f'<div style="border:1px solid rgba(239,68,68,.35);border-radius:14px;padding:1.2rem;background:rgba(239,68,68,.05);direction:{da};text-align:{"right" if is_arabic else "left"};"><div style="font-weight:800;color:#F1F5F9;margin-bottom:.8rem;text-align:{"right" if is_arabic else "left"};">📈 {tp("Areas to Improve","مجالات التحسين")}</div>{ai2}</div>',unsafe_allow_html=True)
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

    user_name  = st.text_input(tl("Full name","الاسم الكامل"), value=st.session_state.get("user_name",""), placeholder=tl("e.g. Dr. Sarah Al-Mutairi","مثال: د. سارة المطيري"))
    user_email = st.text_input(tl("Email address","البريد الإلكتروني"), value=st.session_state.get("user_email",""), placeholder="name@hospital.org")

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
        "good_btn":         {"en": "👍 Good (4/5 all)",                   "ar": "👍 جيد (4/5 للكل)"},
        "avg_btn":          {"en": "😐 Average (3/5)",                   "ar": "😐 متوسط (3/5)"},
        "poor_btn":         {"en": "👎 Poor (2/5 all)",                   "ar": "👎 ضعيف (2/5 للكل)"},
        "saved_45":         {"en": "✅ Saved 4/5 for all metrics",         "ar": "✅ تم حفظ 4/5 لكل المعايير"},
        "saved_35":         {"en": "✅ Saved 3/5 for all metrics",         "ar": "✅ تم حفظ 3/5 لكل المعايير"},
        "saved_25":         {"en": "✅ Saved 2/5 for all metrics",         "ar": "✅ تم حفظ 2/5 لكل المعايير"},
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

    tab1, tab2, tab3 = st.tabs([T('tab_provider'), T('tab_score'), T('tab_manual')])

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
            "gemini":    {"label": "🔵 Gemini — 2.0 Flash",          "secret": "GEMINI_API_KEY",    "color": "#3B82F6"},
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
    # TAB 2 — Score Card (8 metrics)
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        metrics = st.session_state.get("metrics", {})
        manual  = st.session_state.get("manual_ratings", {})

        provider_labels = {
            "groq":      "🟠 Groq",
            "anthropic": "🟣 Claude",
            "openai":    "🟢 OpenAI",
            "gemini":    "🔵 Gemini",
        }

        def stars(val, max_val=5):
            filled = round((val / max_val) * 5) if max_val > 0 else 0
            return "⭐" * filled + "☆" * (5 - filled)

        def avg(lst):
            return round(sum(lst)/len(lst), 2) if lst else None

        rows = []

        def get_m(p): return metrics.get(p, {})
        def get_man(p): return manual.get(p, {})

        # 1. Speed
        speed_row = [T('speed_metric')]
        for p in ["groq","anthropic","openai","gemini"]:
            speeds = get_m(p).get("speed", [])
            speed_row.append(f"{avg(speeds):.1f}s" if speeds else "—")
        rows.append(speed_row)

        # 2. JSON Success Rate
        json_row = [T('json_metric')]
        for p in ["groq","anthropic","openai","gemini"]:
            m = get_m(p)
            total_j = m.get("json_ok",0) + m.get("json_fail",0)
            rate = int(m.get("json_ok",0)/total_j*100) if total_j > 0 else None
            json_row.append(f"{rate}%" if rate is not None else "—")
        rows.append(json_row)

        # 3. Error Rate
        err_row = [T('error_metric')]
        for p in ["groq","anthropic","openai","gemini"]:
            m = get_m(p)
            calls = m.get("calls",0)
            errs  = m.get("errors",0)
            rate  = int(errs/calls*100) if calls > 0 else None
            err_row.append(f"{rate}%" if rate is not None else "—")
        rows.append(err_row)

        # 4. Diversity (unique hashes)
        div_row = [T('diversity_metric')]
        for p in ["groq","anthropic","openai","gemini"]:
            hashes = get_m(p).get("hashes",[])
            calls  = get_m(p).get("calls",0)
            div_row.append(f"{len(hashes)}/{calls}" if calls > 0 else "—")
        rows.append(div_row)

        # 5-8 Manual ratings
        for metric_key, metric_label in [
            ("quality",    T('quality_metric')),
            ("difficulty", T('difficulty_metric')),
            ("arabic",     T('arabic_metric')),
            ("medical",    T('medical_metric')),
        ]:
            row = [metric_label]
            for p in ["groq","anthropic","openai","gemini"]:
                ratings = get_man(p).get(metric_key, [])
                a = avg(ratings)
                row.append(f"{a:.1f}/5 {stars(a) if a else ''}" if a is not None else "—")
            rows.append(row)

        # Render table
        st.markdown(f'<div dir="{_dir}" style="font-weight:900;color:#D1FAE5;font-size:1.1rem;margin-bottom:1rem;">{T("score_title")}</div>', unsafe_allow_html=True)

        # Header
        hcols = st.columns([2,1,1,1,1])
        headers_list = [T('metric_col'), "🟠 Groq", "🟣 Claude", "🟢 OpenAI", "🔵 Gemini"]
        for ci, hdr in enumerate(headers_list):
            with hcols[ci]:
                st.markdown(f'<div dir="{_dir}" style="font-weight:800;color:#9CA3AF;font-size:.82rem;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.1);">{hdr}</div>', unsafe_allow_html=True)

        for row in rows:
            rcols = st.columns([2,1,1,1,1])
            for ci, cell in enumerate(row):
                with rcols[ci]:
                    clr = "#D1FAE5" if ci == 0 else "#F0FDF4"
                    st.markdown(f'<div dir="{_dir}" style="color:{clr};font-size:.85rem;padding:.4rem 0;border-bottom:1px solid rgba(255,255,255,.05);">{cell}</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(f'<div dir="{_dir}" style="font-size:.75rem;color:#6B7280;">{T("auto_manual_note")}</div>', unsafe_allow_html=True)

        if st.button(T('reset_metrics'), use_container_width=True):
            st.session_state["metrics"] = {}
            st.session_state["manual_ratings"] = {}
            st.success(T('metrics_reset'))
            st.rerun()

    # ──────────────────────────────────────────────────────────
    # TAB 3 — Manual Ratings (👍 اليدوية)
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        cur_prov = st.session_state.get("ai_provider", "groq")
        prov_label = provider_info.get(cur_prov, {}).get("label", cur_prov)

        st.markdown(f'<div style="font-weight:900;color:#D1FAE5;margin-bottom:.3rem;">{T("rate_title")}</div>'
                    f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:1rem;">{T("active_provider")}: {prov_label}</div>',
                    unsafe_allow_html=True)

        if "manual_ratings" not in st.session_state:
            st.session_state["manual_ratings"] = {}
        if cur_prov not in st.session_state["manual_ratings"]:
            st.session_state["manual_ratings"][cur_prov] = {
                "quality": [], "difficulty": [], "arabic": [], "medical": []
            }

        manual_metrics = [
            ("quality",    T('quality_label'),     T('quality_desc')),
            ("difficulty", T('diff_acc_label'),    T('diff_acc_desc')),
            ("arabic",     T('arabic_label'),      T('arabic_desc')),
            ("medical",    T('medical_label'),     T('medical_desc')),
        ]

        ratings_to_save = {}
        all_rated = True

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

        for mk, ml, mdesc in manual_metrics:
            st.markdown(f'<div style="margin-bottom:.2rem;"><span style="font-weight:700;color:#E2E8F0;">{ml}</span>'
                        f'<span style="color:#6B7280;font-size:.8rem;margin-right:.5rem;"> — {mdesc}</span></div>',
                        unsafe_allow_html=True)
            rating = st.select_slider(
                label=ml,
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{'⭐'*x}{'☆'*(5-x)} ({x}/5)",
                key=f"rating_{mk}_{cur_prov}",
                label_visibility="collapsed"
            )
            ratings_to_save[mk] = rating
            st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)

        col_note, _ = st.columns([2,1])
        with col_note:
            note = st.text_input(T('note_label'), placeholder=T('note_placeholder'),
                                 key=f"note_{cur_prov}")

        if st.button(T('save_btn'), use_container_width=True):
            for mk, val in ratings_to_save.items():
                st.session_state["manual_ratings"][cur_prov][mk].append(val)
            st.success(f"{T('ratings_saved')} {prov_label}!")
            st.rerun()

        # Quick thumbs up/down shortcut
        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#9CA3AF;font-size:.85rem;margin-bottom:.4rem;">{T("quick_rating")}</div>', unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button(T('good_btn'), use_container_width=True, key="quick_good"):
                for mk in ["quality","difficulty","arabic","medical"]:
                    st.session_state["manual_ratings"][cur_prov][mk].append(4)
                st.success(T('saved_45'))
                st.rerun()
        with qc2:
            if st.button(T('avg_btn'), use_container_width=True, key="quick_avg"):
                for mk in ["quality","difficulty","arabic","medical"]:
                    st.session_state["manual_ratings"][cur_prov][mk].append(3)
                st.success(T('saved_35'))
                st.rerun()
        with qc3:
            if st.button(T('poor_btn'), use_container_width=True, key="quick_bad"):
                for mk in ["quality","difficulty","arabic","medical"]:
                    st.session_state["manual_ratings"][cur_prov][mk].append(2)
                st.success(T('saved_25'))
                st.rerun()

        # History summary
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        man = st.session_state["manual_ratings"].get(cur_prov, {})
        if any(man.values()):
            st.markdown(f'<div style="font-weight:700;color:#D1FAE5;margin-bottom:.4rem;">{T("rating_history")} {prov_label}</div>', unsafe_allow_html=True)
            for mk, ml, _ in manual_metrics:
                lst = man.get(mk, [])
                if lst:
                    a = round(sum(lst)/len(lst),1)
                    st.markdown(f'<div style="color:#9CA3AF;font-size:.82rem;">{ml}: {T("avg_label")} {a}/5 ({len(lst)} {T("ratings_label")}) {"⭐"*round(a)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
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
