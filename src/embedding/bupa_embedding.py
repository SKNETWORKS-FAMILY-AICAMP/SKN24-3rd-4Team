from __future__ import annotations

import os
import re
import shutil
import argparse
from math import ceil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
import pdfplumber
import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "raw" / "bupa"

DB_LATEST_DIR = BASE_DIR / "vectordb" / "bupa" / "bupa_latest"
DB_ALL_DIR    = BASE_DIR / "vectordb" / "bupa" / "bupa_all"

COLLECTION_LATEST = "bupa_latest"
COLLECTION_ALL    = "bupa_all"

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ─────────────────────────────────────────────────────────────────
# 실행 옵션
# ─────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE      = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "8"))
INDEX_BATCH_SIZE  = int(os.getenv("INDEX_BATCH_SIZE", "100"))
RESET_VECTORDB    = os.getenv("RESET_VECTORDB", "true").lower() == "true"

TORCH_NUM_THREADS = int(
    os.getenv("TORCH_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
)

torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ─────────────────────────────────────────────────────────────────
# 파일 목록
# 같은 plan_tier 안에서는 최신 버전 1개만 is_latest=True 가 되도록 관리
# ─────────────────────────────────────────────────────────────────
PDF_CONFIGS: List[Dict[str, Any]] = [
    {
        "path":         DATA_DIR / "HKX_IHHP_Membership_Guide_EN_NOV24_0053016.pdf",
        "plan_type":    "IHHP",
        "plan_tier":    "IHHP",
        "plan_display": "Bupa IHHP",
        "region":       "Global",
        "is_modular":   True,
        "source_id":    "bupa_ihhp",
        "is_spread":    False,
        "doc_year":     2024,
        "is_latest":    True,
    },
    {
        "path":         DATA_DIR / "Bupa-Global-DAC-Select-Global-Health-Plan-MEMBERSHIP-GUIDE-2024.05.pdf",
        "plan_type":    "GHP",
        "plan_tier":    "Select",
        "plan_display": "Bupa GHP Select",
        "region":       "Global",
        "is_modular":   False,
        "source_id":    "bupa_ghp_select",
        "is_spread":    True,
        "doc_year":     2024,
        "is_latest":    True,
    },
    {
        "path":         DATA_DIR / "Bupa-Global-DAC-Major-Medical-Health-Plan-MEMBERSHIP-GUIDE-2024.05.pdf",
        "plan_type":    "GHP",
        "plan_tier":    "MajorMedical",
        "plan_display": "Bupa GHP Major Medical",
        "region":       "Global",
        "is_modular":   False,
        "source_id":    "bupa_ghp_major_medical",
        "is_spread":    True,
        "doc_year":     2024,
        "is_latest":    True,
    },
    {
        "path":         DATA_DIR / "Bupa-Global-DAC-Premier-Global-Health-Plan-MEMBERSHIP-GUIDE-2024.05.pdf",
        "plan_type":    "GHP",
        "plan_tier":    "Premier",
        "plan_display": "Bupa GHP Premier",
        "region":       "Global",
        "is_modular":   False,
        "source_id":    "bupa_ghp_premier",
        "is_spread":    True,
        "doc_year":     2024,
        "is_latest":    True,
    },
    {
        "path":         DATA_DIR / "UAE-Elite-Global-Health-Plan-Membership-Guide-EN-NOV24-0051587.pdf",
        "plan_type":    "GHP",
        "plan_tier":    "Elite",
        "plan_display": "Bupa UAE Elite",
        "region":       "UAE",
        "is_modular":   False,
        "source_id":    "bupa_ghp_elite_uae",
        "is_spread":    True,
        "doc_year":     2024,
        "is_latest":    True,
    },
    {
        "path":         DATA_DIR / "DAC_Ultimate_Global_Health_Plan_Membership_Guide_EN_DEC24_0054831.pdf",
        "plan_type":    "GHP",
        "plan_tier":    "Ultimate",
        "plan_display": "Bupa Ultimate",
        "region":       "Global",
        "is_modular":   False,
        "source_id":    "bupa_ghp_ultimate",
        "is_spread":    True,
        "doc_year":     2024,
        "is_latest":    True,
    },
]

# ─────────────────────────────────────────────────────────────────
# 섹션 감지 — 제외 패턴 (페이지 상단에서 발견되면 통째로 버림)
# ─────────────────────────────────────────────────────────────────
EXCLUDE_PATTERNS: List[str] = [
    r"welcome to membersworld",
    r"how to access membersworld",
    r"wellbeing services",
    r"global virtual care",
    r"second medical opinion",
    r"privacy notice",
    r"data protection",
    r"round the clock reassurance",
    r"a guide to your .* health plan",   # 표지
    r"^hello\b",                          # Hello 인트로
    r"bupa global offers you",            # 마지막 연락처 페이지
    r"general services:.*\+44",           # 연락처 페이지
]

# ─────────────────────────────────────────────────────────────────
# 섹션 감지 — 섹션 타입 패턴 (우선순위 순서)
# ─────────────────────────────────────────────────────────────────
SECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("benefit_table", [
        r"table of benefits",
        r"benefit and explanation",
        r"overall annual (policy )?maximum",
        r"paid in full",
        r"hospital plan",     # 'Hospital Plan (continued)' 포함
        r"module \d",         # 'Module 1', 'Module 2' 등
        r"modules 4",         # 'Modules 4A and 4B'
    ]),
    ("exclusion", [
        r"general exclusions",
        r"what is not covered",
        r"your exclusions",
        r"exceptions to cover",
    ]),
    ("claim_process", [
        r"the claiming process",
        r"how to (make|submit) a claim",
        r"pay and claim",
        r"direct (payment|settlement)",
        r"claiming process",
    ]),
    ("pre_auth", [
        r"pre.?authoris",
        r"need treatment",
        r"mandatory pre.?authorisation",
        r"how to pre.?authorise",
        r"our approach to costs",
    ]),
    ("glossary", [
        r"^glossary\b",
        r"defined terms?\s+description",
        r"defined term\b",
    ]),
    ("membership_admin", [
        r"want to add more people",
        r"adding your (newborn|dependant)",
        r"children covered at no additional cost",
        r"dependants",
        r"newborn (application|care|child)",
    ]),
    ("terms_conditions", [
        r"terms and conditions",
        r"your policy\b",
        r"^\s*\d+\.\s+(your policy|your cover|premium and payment|renewal|changes to your policy|ending this policy)",
    ]),
]


def detect_section_type(page_text: str) -> Optional[str]:
    """
    페이지 텍스트에서 section_type을 감지.
    - None 반환 시 해당 페이지는 제외 대상
    - 제외 패턴은 페이지 첫 200자(헤더 영역)에서만 검사
    - 섹션 패턴은 전체 텍스트 대상
    """
    text_lower  = page_text.lower().strip()
    header_zone = text_lower[:200]

    # 1. 제외 패턴 우선 체크
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, header_zone, re.IGNORECASE | re.MULTILINE):
            return None

    # 2. 섹션 타입 감지
    for section_type, patterns in SECTION_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return section_type

    # 3. 어떤 섹션에도 해당하지 않으면 제외
    return None


# ─────────────────────────────────────────────────────────────────
# 텍스트 정제
# ─────────────────────────────────────────────────────────────────
def clean_page_text(text: str) -> str:
    """
    페이지 텍스트 노이즈 제거:
    - URL (https://, www.)
    - 독립 줄의 페이지 번호
    - 저작권·면책 문구
    - 3개 이상 연속 빈 줄 → 2개로
    """
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"Bupa Global is a trade name.*?Blue Cross and Blue Shield Association\.",
        "", text, flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# 다국어 검색 태그 (메타데이터 보강)
# ─────────────────────────────────────────────────────────────────
PLAN_TIER_ALIASES: Dict[str, List[str]] = {
    "IHHP":        ["ihhp", "individual health plan", "hong kong", "홍콩", "개인 건강보험"],
    "Select":      ["select", "셀렉트", "select plan"],
    "MajorMedical":["major medical", "메이저 메디컬", "major medical plan"],
    "Premier":     ["premier", "프리미어", "premier plan"],
    "Elite":       ["elite", "엘리트", "uae elite", "elite plan"],
    "Ultimate":    ["ultimate", "얼티밋", "ultimate plan"],
}

SECTION_TYPE_ALIASES: Dict[str, List[str]] = {
    "benefit_table":    ["table of benefits", "보장 표", "보장 금액", "한도", "limit", "coverage table"],
    "exclusion":        ["general exclusions", "not covered", "미보장", "제외사항", "exclusion"],
    "claim_process":    ["claiming process", "how to claim", "청구 방법", "청구 절차", "reimbursement"],
    "pre_auth":         ["pre-authorisation", "preauth", "사전승인", "need treatment", "입원 전 승인"],
    "glossary":         ["glossary", "defined terms", "용어 정의", "약관 용어"],
    "membership_admin": ["add dependant", "newborn", "피부양자 추가", "신생아 등록", "membership admin"],
    "terms_conditions": ["terms and conditions", "your policy", "약관", "정책 조건", "보험 조항"],
}

INSURANCE_SEARCH_TAGS: List[str] = [
    "coverage", "covered", "benefit", "limit", "co-payment", "copay",
    "deductible", "waiting period", "exclusion", "outpatient", "inpatient",
    "maternity", "cancer", "chronic condition", "pre-existing condition",
    "pre-authorisation", "preauthorization", "planned hospitalisation",
    "direct billing", "claim", "reimbursement", "invoice", "receipt",
    "서류", "청구", "환급", "직접청구", "사전승인", "보장", "혜택",
    "한도", "면책", "제외사항", "외래", "입원", "출산", "기왕증",
]


def build_search_tags(cfg: Dict[str, Any], section_type: str) -> str:
    tier_aliases    = PLAN_TIER_ALIASES.get(cfg["plan_tier"], [])
    section_aliases = SECTION_TYPE_ALIASES.get(section_type, [])

    return "\n".join([
        "[search_tags]",
        f"plan_tier: {' | '.join(tier_aliases)}",
        f"section_type: {' | '.join(section_aliases)}",
        f"plan_type: {cfg['plan_type']}",
        f"region: {cfg['region']}",
        f"doc_year: {cfg['doc_year']}",
        f"is_latest: {cfg.get('is_latest', False)}",
        "insurer: Bupa 부파",
        "keywords: " + ", ".join(INSURANCE_SEARCH_TAGS),
    ])


def enrich_text_for_multilingual_search(
    text: str,
    cfg: Dict[str, Any],
    section_type: str,
) -> str:
    return f"{text}\n\n{build_search_tags(cfg, section_type)}"


# ─────────────────────────────────────────────────────────────────
# 공통 메타데이터 빌더
# ─────────────────────────────────────────────────────────────────
def build_common_metadata(
    cfg: Dict[str, Any],
    section_type: str,
    chunk_type: str,
    page_label: str,
    physical_page: int,
    chunk_idx: Optional[int] = None,
    table_idx: Optional[int] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source":        os.path.basename(cfg["path"]),
        "source_id":     cfg["source_id"],
        "plan_type":     cfg["plan_type"],
        "plan_tier":     cfg["plan_tier"],
        "plan_display":  cfg["plan_display"],
        "region":        cfg["region"],
        "is_modular":    str(cfg["is_modular"]),
        "doc_year":      cfg["doc_year"],
        "is_latest":     cfg.get("is_latest", False),
        "section_type":  section_type,
        "chunk_type":    chunk_type,
        "page":          page_label,
        "physical_page": physical_page,
        "insurer":       "Bupa",
    }

    if chunk_idx is not None:
        metadata["chunk_idx"] = chunk_idx

    if table_idx is not None:
        metadata["table_idx"] = table_idx

    return metadata


# ─────────────────────────────────────────────────────────────────
# 스프레드 페이지 유틸
# ─────────────────────────────────────────────────────────────────
def _get_fitz_page_parts(
    page: Any,
    page_num: int,
    is_spread: bool,
) -> List[Tuple[Any, str, int]]:
    """
    Returns list of (fitz_page_or_clip_result, page_label, physical_page_num).
    스프레드(2페이지 1장) 문서의 경우 좌/우로 분할.
    """
    if not is_spread:
        return [(page, str(page_num), page_num)]

    if page_num == 1:
        return [(page, "1", 1)]

    rect  = page.rect
    mid_x = rect.width / 2
    left_idx  = (page_num - 1) * 2
    right_idx = (page_num - 1) * 2 + 1

    left_clip  = page.get_text(clip=fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1))
    right_clip = page.get_text(clip=fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1))

    # 클립은 텍스트 문자열로 반환되므로 별도 처리
    return [
        (left_clip,  str(left_idx),  page_num),
        (right_clip, str(right_idx), page_num),
    ]


def _get_pdfplumber_page_parts(
    page: Any,
    page_num: int,
    is_spread: bool,
) -> List[Tuple[Any, str, int]]:
    """
    Returns list of (pdfplumber_page_or_crop, page_label, physical_page_num).
    """
    if not is_spread:
        return [(page, str(page_num), page_num)]

    if page_num == 1:
        return [(page, "1", 1)]

    w, h = page.width, page.height
    left_idx  = (page_num - 1) * 2
    right_idx = (page_num - 1) * 2 + 1

    return [
        (page.crop((0,     0, w / 2, h)), str(left_idx),  page_num),
        (page.crop((w / 2, 0, w,     h)), str(right_idx), page_num),
    ]


# ─────────────────────────────────────────────────────────────────
# 텍스트 청크 — PDF 로드 + 섹션 분류
# ─────────────────────────────────────────────────────────────────
def load_text_docs(cfg: Dict[str, Any]) -> List[Document]:
    """
    PyMuPDF로 페이지 단위 로딩 → 섹션 감지 → 정제 → Document 생성.
    스프레드 문서는 좌/우 분리 처리.
    """
    docs: List[Document] = []
    is_spread = cfg.get("is_spread", False)
    path: Path = cfg["path"]

    if not path.exists():
        print(f"[WARN] 파일이 없습니다: {path}")
        return []

    fitz_doc = fitz.open(str(path))

    try:
        for page_num, page in enumerate(fitz_doc, start=1):
            if is_spread:
                if page_num == 1:
                    parts: List[Tuple[str, str, int]] = [
                        (page.get_text("text"), "1", 1)
                    ]
                else:
                    rect  = page.rect
                    mid_x = rect.width / 2
                    left_idx  = (page_num - 1) * 2
                    right_idx = (page_num - 1) * 2 + 1

                    parts = [
                        (
                            page.get_text("text", clip=fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1)),
                            str(left_idx),
                            page_num,
                        ),
                        (
                            page.get_text("text", clip=fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1)),
                            str(right_idx),
                            page_num,
                        ),
                    ]
            else:
                parts = [(page.get_text("text"), str(page_num), page_num)]

            for raw_text, page_label, phys_page in parts:
                if not raw_text.strip():
                    continue

                section_type = detect_section_type(raw_text)
                if section_type is None:
                    continue

                cleaned = clean_page_text(raw_text)
                if len(cleaned) < 80:
                    continue

                content = enrich_text_for_multilingual_search(cleaned, cfg, section_type)

                docs.append(
                    Document(
                        page_content=content,
                        metadata=build_common_metadata(
                            cfg=cfg,
                            section_type=section_type,
                            chunk_type="text",
                            page_label=page_label,
                            physical_page=phys_page,
                        ),
                    )
                )
    finally:
        fitz_doc.close()

    return docs


# ─────────────────────────────────────────────────────────────────
# 테이블 청크 — pdfplumber 표 추출
# ─────────────────────────────────────────────────────────────────
SYMBOL_MAP: Dict[str, str] = {
    "✔": "Covered", "✓": "Covered", "●": "Covered", "O": "Covered",
    "\uf0fc": "Covered", "\u2713": "Covered",
    "✘": "Not Covered", "X": "Not Covered",
    "N/A": "Not Covered",
}

TABLE_EXTRACT_SETTINGS: Dict[str, Any] = {
    "vertical_strategy":   "text",
    "horizontal_strategy": "text",
    "snap_tolerance":      3,
}

# 테이블 추출 대상 섹션
TABLE_TARGET_SECTIONS = {"benefit_table", "exclusion", "pre_auth"}


def table_to_text(table: List[List[Any]], is_modular: bool = False) -> str:
    rows = []
    header_row: Optional[List[str]] = None

    for row_idx, row in enumerate(table):
        cleaned_row: List[str] = []

        for cell in row:
            if cell is not None and str(cell).strip():
                text = str(cell).strip().replace("\n", " ")
                text = SYMBOL_MAP.get(text, text)
                cleaned_row.append(text)
            else:
                cleaned_row.append("")  # 빈 셀은 Not Covered 남발 방지

        if row_idx == 0:
            header_row = cleaned_row

        # IHHP 모듈형: Covered + Not Covered 혼합 행에 태그 추가
        if is_modular and header_row:
            has_covered     = any("Covered" in c and c != "Not Covered" for c in cleaned_row)
            has_not_covered = any(c == "Not Covered" for c in cleaned_row)
            if has_covered and has_not_covered:
                cleaned_row.append("[일부 모듈 가입 시 보장]")

        rows.append(" | ".join(c for c in cleaned_row if c))

    return "\n".join(rows)


def is_meaningful_table(table: List[List[Any]]) -> bool:
    if not table or len(table) < 2:
        return False
    non_empty = sum(1 for row in table for cell in row if cell and str(cell).strip())
    return non_empty >= 4


def load_table_docs(cfg: Dict[str, Any]) -> List[Document]:
    """
    pdfplumber로 표를 추출하여 Document 리스트로 반환.
    benefit_table / exclusion / pre_auth 섹션의 표만 처리.
    """
    docs: List[Document] = []
    path: Path = cfg["path"]
    is_spread  = cfg.get("is_spread", False)
    is_modular = cfg["is_modular"]

    if not path.exists():
        print(f"[WARN] 파일이 없습니다: {path}")
        return []

    current_section: Optional[str] = None

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if is_spread:
                if page_num == 1:
                    sides: List[Tuple[Any, str, int]] = [(page, "1", 1)]
                else:
                    w, h = page.width, page.height
                    left_idx  = (page_num - 1) * 2
                    right_idx = (page_num - 1) * 2 + 1
                    sides = [
                        (page.crop((0,     0, w / 2, h)), str(left_idx),  page_num),
                        (page.crop((w / 2, 0, w,     h)), str(right_idx), page_num),
                    ]
            else:
                sides = [(page, str(page_num), page_num)]

            for side_page, page_label, phys_page in sides:
                page_text = side_page.extract_text() or ""

                detected = detect_section_type(page_text)
                if detected is not None:
                    current_section = detected

                if current_section not in TABLE_TARGET_SECTIONS:
                    continue

                extracted_tables = side_page.extract_tables(TABLE_EXTRACT_SETTINGS)

                for tbl_idx, table in enumerate(extracted_tables):
                    if not is_meaningful_table(table):
                        continue

                    table_text = table_to_text(table, is_modular=is_modular)
                    if len(table_text) < 50:
                        continue

                    modular_note = (
                        "\n※ 모듈형 표: Not Covered는 해당 모듈 미가입 상태."
                        " 모듈 가입 시 보장 항목은 [일부 모듈 가입 시 보장] 태그로 표시됩니다.\n"
                        if is_modular else "\n"
                    )

                    header = f"[{cfg['plan_display']} | {current_section} Table | p.{page_label}]"
                    content = enrich_text_for_multilingual_search(
                        f"{header}{modular_note}{table_text}",
                        cfg,
                        current_section,
                    )

                    docs.append(
                        Document(
                            page_content=content,
                            metadata=build_common_metadata(
                                cfg=cfg,
                                section_type=current_section,
                                chunk_type="table",
                                page_label=page_label,
                                physical_page=phys_page,
                                table_idx=tbl_idx,
                            ),
                        )
                    )

    return docs


# ─────────────────────────────────────────────────────────────────
# 섹션별 차등 청크 분할
# ─────────────────────────────────────────────────────────────────
CHUNK_CONFIGS: Dict[str, Dict[str, int]] = {
    "benefit_table":    {"chunk_size": 1000, "chunk_overlap": 150},
    "exclusion":        {"chunk_size": 1200, "chunk_overlap": 200},
    "glossary":         {"chunk_size": 800,  "chunk_overlap": 100},
    "claim_process":    {"chunk_size": 1000, "chunk_overlap": 150},
    "pre_auth":         {"chunk_size": 1000, "chunk_overlap": 150},
    "waiting_period":   {"chunk_size": 800,  "chunk_overlap": 100},
    "membership_admin": {"chunk_size": 1000, "chunk_overlap": 150},
    "terms_conditions": {"chunk_size": 1200, "chunk_overlap": 200},
}

DEFAULT_CHUNK_CONFIG: Dict[str, int] = {"chunk_size": 1000, "chunk_overlap": 150}

COMMON_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


def split_docs_by_section(docs: List[Document]) -> List[Document]:
    """섹션 타입별로 최적화된 청크 크기를 적용해 분할."""
    groups: Dict[str, List[Document]] = {}
    for doc in docs:
        stype = doc.metadata.get("section_type", "unknown")
        groups.setdefault(stype, []).append(doc)

    all_chunks: List[Document] = []

    for stype, group_docs in groups.items():
        cfg = CHUNK_CONFIGS.get(stype, DEFAULT_CHUNK_CONFIG)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=COMMON_SEPARATORS,
            length_function=len,
        )

        chunks = splitter.split_documents(group_docs)
        all_chunks.extend(chunks)

    return all_chunks


# ─────────────────────────────────────────────────────────────────
# 도큐먼트 빌드 (전체 파이프라인)
# ─────────────────────────────────────────────────────────────────
def build_documents(
    config_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Document]:
    """
    PDF_CONFIGS를 순회하며:
    1) 텍스트 로드 + 섹션 분류 (PyMuPDF)
    2) 표 추출 (pdfplumber)
    3) 텍스트 청크 분할
    4) 텍스트 청크 + 테이블 청크 합산 반환
    """
    target_configs = config_list or PDF_CONFIGS
    all_text_docs:  List[Document] = []
    all_table_docs: List[Document] = []

    for cfg in target_configs:
        plan_tier = cfg["plan_tier"]
        source    = os.path.basename(cfg["path"])

        print(
            f"[INFO] 처리 중: {source} | tier={plan_tier} | "
            f"region={cfg['region']} | year={cfg['doc_year']} | "
            f"is_latest={cfg.get('is_latest', False)}"
        )

        if not Path(cfg["path"]).exists():
            print(f"[WARN] 파일이 없습니다: {cfg['path']}")
            continue

        # 텍스트 페이지 로드
        raw_text_docs = load_text_docs(cfg)
        print(f"  └─ 텍스트 페이지: {len(raw_text_docs)}면")

        # 섹션별 청크 분할
        text_chunks = split_docs_by_section(raw_text_docs)
        all_text_docs.extend(text_chunks)
        print(f"  └─ 텍스트 청크: {len(text_chunks)}개")

        # 테이블 추출
        table_docs = load_table_docs(cfg)
        all_table_docs.extend(table_docs)
        print(f"  └─ 테이블 청크: {len(table_docs)}개")

    all_docs = all_text_docs + all_table_docs
    print(
        f"[INFO] 전체 청크 — 텍스트: {len(all_text_docs)}, "
        f"테이블: {len(all_table_docs)}, 합계: {len(all_docs)}"
    )
    return all_docs


# ─────────────────────────────────────────────────────────────────
# 임베딩 / 벡터스토어
# ─────────────────────────────────────────────────────────────────
def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE,
        },
    )


def reset_vectordbs_if_needed() -> None:
    if RESET_VECTORDB:
        for db_dir in [DB_LATEST_DIR, DB_ALL_DIR]:
            if db_dir.exists():
                print(f"[INFO] 기존 벡터DB 삭제: {db_dir}")
                shutil.rmtree(db_dir, ignore_errors=True)


def _index_to_single_store(
    documents: List[Document],
    persist_directory: Path,
    collection_name: str,
    embeddings: HuggingFaceEmbeddings,
    batch_size: int,
) -> Optional[Chroma]:
    if not documents:
        print(f"[WARN] 인덱싱할 문서가 없습니다: {collection_name}")
        return None

    vectordb = None
    total      = len(documents)
    num_batches = ceil(total / batch_size)

    for i in range(num_batches):
        start      = i * batch_size
        end        = min(start + batch_size, total)
        batch_docs = documents[start:end]

        print(f"[INFO] [{collection_name}] Embedding batch {i + 1}/{num_batches} ({start}~{end})")

        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name,
            )
        else:
            vectordb.add_documents(batch_docs)

    if vectordb is not None:
        try:
            vectordb.persist()
        except Exception:
            pass

    return vectordb


def build_vectorstores(
    reload: bool = False,
    batch_size: int = INDEX_BATCH_SIZE,
) -> Tuple[Optional[Chroma], Optional[Chroma], List[Document], List[Document]]:
    if reload:
        reset_vectordbs_if_needed()

    documents        = build_documents()
    latest_documents = [d for d in documents if d.metadata.get("is_latest", False)]

    print(f"[INFO] Total chunks(all):    {len(documents)}")
    print(f"[INFO] Total chunks(latest): {len(latest_documents)}")
    print(
        f"[INFO] EMBED_MODEL_NAME={EMBED_MODEL_NAME}, "
        f"EMBED_DEVICE={EMBED_DEVICE}, "
        f"EMBED_BATCH_SIZE={EMBED_BATCH_SIZE}, "
        f"INDEX_BATCH_SIZE={batch_size}, "
        f"TORCH_NUM_THREADS={TORCH_NUM_THREADS}"
    )

    embeddings = build_embeddings()

    vectorstore_latest = _index_to_single_store(
        documents=latest_documents,
        persist_directory=DB_LATEST_DIR,
        collection_name=COLLECTION_LATEST,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    vectorstore_all = _index_to_single_store(
        documents=documents,
        persist_directory=DB_ALL_DIR,
        collection_name=COLLECTION_ALL,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    print("[DONE] latest/all 벡터스토어 구축 완료")
    return vectorstore_latest, vectorstore_all, documents, latest_documents


def load_vectorstores() -> Tuple[Chroma, Chroma]:
    embeddings = build_embeddings()

    vectorstore_latest = Chroma(
        persist_directory=str(DB_LATEST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_LATEST,
    )
    vectorstore_all = Chroma(
        persist_directory=str(DB_ALL_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_ALL,
    )
    return vectorstore_latest, vectorstore_all


# ─────────────────────────────────────────────────────────────────
# 버전 갱신 유틸 (신규 플랜 문서 추가 시)
# ─────────────────────────────────────────────────────────────────
def update_vectorstore_latest(
    vectorstore_latest: Chroma,
    vectorstore_all: Chroma,
    new_cfg: Dict[str, Any],
) -> Tuple[List[Document], List[Document]]:
    """
    신규 문서 버전이 들어왔을 때:
    - 같은 source_id 의 기존 latest 문서는 latest DB 에서 제거
    - PDF_CONFIGS 내부의 기존 latest 는 False 로 변경
    - 신규 문서는 is_latest=True 로 추가
    - latest DB / all DB 모두에 신규 chunk 추가
    """
    new_cfg = dict(new_cfg)
    source_id = new_cfg.get("source_id", "")
    new_cfg["is_latest"] = True

    # 1) 기존 latest 삭제
    existing = vectorstore_latest.get(
        where={
            "$and": [
                {"source_id": {"$eq": source_id}},
                {"is_latest": {"$eq": True}},
            ]
        }
    )
    old_ids = existing.get("ids", [])
    if old_ids:
        vectorstore_latest.delete(ids=old_ids)
        print(f"[INFO] 기존 latest 청크 삭제: {len(old_ids)}개 | source_id={source_id}")

    # 2) PDF_CONFIGS 내 기존 latest 해제
    for item in PDF_CONFIGS:
        if item.get("source_id") == source_id and item.get("is_latest", False):
            item["is_latest"] = False

    PDF_CONFIGS.append(new_cfg)

    # 3) 신규 문서 청킹
    new_docs = build_documents([new_cfg])
    if not new_docs:
        print("[WARN] 신규 문서에서 생성된 청크가 없습니다.")
        return [], []

    # 4) latest / all 에 추가
    vectorstore_latest.add_documents(new_docs)
    vectorstore_all.add_documents(new_docs)

    try:
        vectorstore_latest.persist()
    except Exception:
        pass

    try:
        vectorstore_all.persist()
    except Exception:
        pass

    print(
        f"[DONE] 신규 최신 버전 반영 완료 | "
        f"source_id={source_id} | year={new_cfg['doc_year']} | chunks={len(new_docs)}"
    )
    return new_docs, new_docs


def get_documents_from_store(vectorstore: Chroma) -> List[Document]:
    raw  = vectorstore.get(include=["documents", "metadatas"])
    docs = []
    for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


# ─────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Bupa RAG 벡터스토어 구축")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="기존 latest/all DB 삭제 후 재구축",
    )
    args = parser.parse_args()

    vs_latest, vs_all, all_docs, latest_docs = build_vectorstores(
        reload=args.reload,
        batch_size=INDEX_BATCH_SIZE,
    )

    print(f"[INFO] 전체 청크 수(all):    {len(all_docs)}")
    print(f"[INFO] 최신 청크 수(latest): {len(latest_docs)}")

    if vs_latest is None or vs_all is None:
        print("[WARN] 벡터스토어 생성이 완전하지 않을 수 있습니다.")


if __name__ == "__main__":
    main()
