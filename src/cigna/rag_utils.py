"""
rag_utils.py — Cigna RAG 파이프라인 · 리트리버 + LangGraph 파이프라인
Section 3 이후 (리트리버 구성 ~ LangGraph 그래프 조립 + 실행 헬퍼)

사용 예시:
    from rag_utils import ask_cigna
    print(ask_cigna("Silver 플랜 공제액은 얼마인가요?"))
"""

import json
from typing import List, Dict, Literal, Optional

from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from ingest import (
    vectorstore_latest,
    vectorstore_all,
    latest_chunks,
    all_chunks,
    cosine_sim,
    print_sim,
)


# ══════════════════════════════════════════════════════════════════
# Section 3 · 리트리버 구성
# ══════════════════════════════════════════════════════════════════

retriever_dense = vectorstore_latest.as_retriever(
    search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 20},
)

# BM25 인덱스
bm25_corpus = [c.page_content.lower().split() for c in latest_chunks]
bm25        = BM25Okapi(bm25_corpus)


def bm25_search(query: str, k: int = 5) -> List[Document]:
    scores  = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [latest_chunks[i] for i in top_idx]


def rrf_rank(bm25_list: List[Document], dense_list: List[Document], k: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion."""
    scores:  Dict[str, float]    = {}
    doc_map: Dict[str, Document] = {}
    for rank, doc in enumerate(bm25_list, 1):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + 1 / (k + rank)
        doc_map[key] = doc
    for rank, doc in enumerate(dense_list, 1):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + 1 / (k + rank)
        doc_map[key] = doc
    return [doc_map[k_] for k_ in sorted(scores, key=scores.get, reverse=True)]


def hybrid_retriever(query: str, k: int = 5) -> List[Document]:
    """BM25 + Dense MMR → RRF 혼합 검색."""
    bm25_res  = bm25_search(query, k=k * 2)
    dense_res = retriever_dense.invoke(query)
    return rrf_rank(bm25_res, dense_res)[:k]


# ══════════════════════════════════════════════════════════════════
# Section 4 · RAG 기본 체인 (Term Locking + LCEL)
# ══════════════════════════════════════════════════════════════════

TERM_LOCK_TABLE = """
=== 보험 용어 번역 고정 테이블 (Term Locking) ===
- Deductible → 공제액(Deductible)
- Co-insurance / Cost Share → 공동부담률(Co-insurance)
- Copay → 정액 본인부담(Copay)
- Out-of-Pocket Maximum → 최대 본인부담금(OOP Max)
- Prior Approval → 사전 승인(Prior Approval)
- In-network → 네트워크 내(In-network)
- Out-of-network → 네트워크 외(Out-of-network)
- Mental Health Care → 정신건강 케어(Mental Health Care)
- Free Look Period → 청약 철회 기간(Free Look Period)
"""

RAG_TEMPLATE = """
당신은 Cigna Global 국제 건강보험 전문 안내 어시스턴트입니다.
[참고 문서]를 근거로 질문에 답하세요.
{term_lock}
규칙: 문서 근거만 사용 / 출처 인용 [source_type version p.page] / 없으면 '확인 불가' / 보험 추천 금지
[참고 문서]\n{context}\n\n[질문]\n{question}
"""

rag_prompt = PromptTemplate(
    input_variables=['context', 'question', 'term_lock'],
    template=RAG_TEMPLATE,
)

llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)


def format_docs(docs: List[Document]) -> str:
    """검색된 Document 리스트 → 출처 포함 컨텍스트 문자열."""
    return '\n\n'.join(
        f"[{i}] {d.metadata.get('source_type', '?')} {d.metadata.get('doc_version', '?')}"
        f" p.{d.metadata.get('page', '?')}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


# ══════════════════════════════════════════════════════════════════
# Section 6 · LangGraph State 정의
# ══════════════════════════════════════════════════════════════════

class CignaRAGState(TypedDict):
    """LangGraph 전체 파이프라인 상태."""
    question:            str
    plan_info:           dict
    difficulty:          str            # low / medium / high
    needs_clarification: bool
    missing_info:        List[str]
    retrieved_docs:      List[Document]
    hyde_query:          str
    rewrite_count:       int
    answer:              str


# 기본 사용자 플랜 (질문에 명시 없을 때 사용)
DEFAULT_PLAN: dict = {
    'plan_tier':       'Gold',
    'deductible':      750,
    'cost_share_pct':  20,
    'oop_max':         2000,
    'deductible_used': 0,
}


# ══════════════════════════════════════════════════════════════════
# Section 7 · LangGraph 노드 정의
# ══════════════════════════════════════════════════════════════════

# ── Node 1 · classify_question ──────────────────────────────────

class QuestionClassification(BaseModel):
    difficulty: Literal['low', 'medium', 'high'] = Field(
        description='low=수치조회, medium=절차/비교, high=계산/다중조건/복합추론'
    )
    needs_clarification: bool = Field(
        description='계산 질문인데 deductible/cost_share 등 수치가 없으면 True'
    )
    missing_info: List[str] = Field(
        description="부족한 정보 목록. 예: ['deductible 금액', 'cost share %']"
    )
    reasoning: str = Field(description='분류 근거 한 줄')


CLASSIFY_PROMPT = PromptTemplate.from_template("""
당신은 Cigna 보험 질문을 분류하는 전문가입니다.

질문: {question}
사용자 플랜 정보: {plan_info}

분류 기준:
- low: 문서에서 수치/정의 하나를 찾는 질문 (예: 'Silver deductible 얼마?')
- medium: 두 개념 비교, 절차 확인 (예: 'prior approval 필요한가?')
- high: 계산, 여러 조건 조합, 다단계 추론 필요

needs_clarification=True 조건:
- 계산 질문(환급액, 본인부담금)인데 deductible 금액/cost share % 가 질문에 없고
  plan_info도 비어 있을 때
""")

classifier_llm = llm.with_structured_output(QuestionClassification)


def classify_question(state: CignaRAGState) -> CignaRAGState:
    """Node 1: 난이도 + 정보 충족도 판단."""
    plan_info = state.get('plan_info') or DEFAULT_PLAN
    result: QuestionClassification = (CLASSIFY_PROMPT | classifier_llm).invoke({
        'question': state['question'],
        'plan_info': json.dumps(plan_info, ensure_ascii=False),
    })
    print(f'  [분류] 난이도={result.difficulty} | 정보부족={result.needs_clarification}')
    if result.missing_info:
        print(f'  [부족 정보] {result.missing_info}')
    return {
        **state,
        'difficulty':          result.difficulty,
        'needs_clarification': result.needs_clarification,
        'missing_info':        result.missing_info,
        'plan_info':           plan_info,
        'rewrite_count':       state.get('rewrite_count', 0),
    }


# ── Node 2 · hyde_retrieve ──────────────────────────────────────

HYDE_PROMPT = PromptTemplate.from_template("""
다음 Cigna 보험 질문에 대해 실제 문서에 있을 법한 가상의 답변을 영어로 작성하세요.
정확하지 않아도 됩니다. 검색 품질 향상이 목적입니다.

질문: {question}
부족한 정보 (가정해서 작성): {missing_info}
""")

hyde_chain = HYDE_PROMPT | llm | StrOutputParser()


def hyde_retrieve(state: CignaRAGState) -> CignaRAGState:
    """Node 2: HyDE — 가상 답변을 검색어로 사용해 문서 검색."""
    pseudo_answer = hyde_chain.invoke({
        'question':     state['question'],
        'missing_info': ', '.join(state.get('missing_info', [])),
    })
    print(f'  [HyDE 가상 답변] {pseudo_answer[:80]}...')
    docs = hybrid_retriever(pseudo_answer, k=5)
    return {**state, 'hyde_query': pseudo_answer, 'retrieved_docs': docs}


# ── Node 3 · retrieve_by_difficulty ────────────────────────────

def multihop_search(question: str, max_hop: int = 3) -> List[Document]:
    """Multi-hop 검색 — 첫 검색 결과를 보고 다음 검색어를 결정."""
    accumulated, current_query = [], question
    for hop in range(max_hop):
        new_docs = hybrid_retriever(current_query, k=3)
        accumulated.extend(new_docs)
        if hop < max_hop - 1:
            ctx = format_docs(accumulated[:6])
            nq  = (llm | StrOutputParser()).invoke(
                f'원래질문: {question}\n현재컨텍스트:\n{ctx[:600]}\n'
                f'추가 검색이 필요하면 검색어를 한 줄로, 충분하면 DONE:'
            )
            if nq.strip().upper() == 'DONE':
                break
            current_query = nq.strip()
    seen, unique = set(), []
    for d in accumulated:
        k = d.page_content[:80]
        if k not in seen:
            seen.add(k)
            unique.append(d)
    return unique


def retrieve_by_difficulty(state: CignaRAGState) -> CignaRAGState:
    """Node 3: 난이도에 따라 Dense / Hybrid / Multi-hop 선택."""
    q    = state.get('rewritten_question') or state['question']
    diff = state.get('difficulty', 'medium')

    if diff == 'low':
        docs = retriever_dense.invoke(q)
        print(f'  [Dense 검색] {len(docs)}개')
    elif diff == 'high':
        docs = multihop_search(q, max_hop=3)
        print(f'  [Multi-hop 검색] {len(docs)}개')
    else:
        docs = hybrid_retriever(q, k=5)
        print(f'  [Hybrid 검색] {len(docs)}개')

    return {**state, 'retrieved_docs': docs}


# ── Node 4 · grade_documents ────────────────────────────────────

GRADE_PROMPT = (
    '귀하는 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다.\n'
    '검색된 문서: {context}\n'
    '사용자 질문: {question}\n'
    '문서에 질문과 관련된 키워드 또는 의미가 포함되어 있으면 yes, 아니면 no.'
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(description='관련 있으면 "yes", 없으면 "no"')


grader_llm = llm.with_structured_output(GradeDocuments)


def grade_documents_node(state: CignaRAGState) -> Literal['generate_answer', 'rewrite_query', 'hyde_fallback']:
    """Node 4 (조건부 엣지): 검색 결과 관련성 평가 → 다음 노드 결정."""
    question      = state['question']
    docs          = state.get('retrieved_docs', [])
    context       = format_docs(docs[:3])
    rewrite_count = state.get('rewrite_count', 0)

    result: GradeDocuments = (PromptTemplate.from_template(GRADE_PROMPT) | grader_llm).invoke({
        'question': question,
        'context':  context[:800],
    })
    print(f'  [Grade] {result.binary_score} | 재작성 횟수: {rewrite_count}')

    if result.binary_score == 'yes':
        return 'generate_answer'
    elif rewrite_count >= 2:
        return 'hyde_fallback'
    else:
        return 'rewrite_query'


# ── Node 5 · rewrite_query ──────────────────────────────────────

REWRITE_PROMPT = PromptTemplate.from_template(
    '입력된 내용을 바탕으로 사용자의 근본적인 의도와 의미를 분석한다.\n'
    '다음은 초기 질문이다:\n ------ \n{question}\n ------ \n'
    '분석한 의도를 반영하여 더 명확하고 구체적인 Cigna 보험 검색어로 재구성한다.'
    '(영어 키워드 포함)'
)


def rewrite_query(state: CignaRAGState) -> CignaRAGState:
    """Node 5: 관련성 낮을 때 쿼리 의도를 재해석해 재작성."""
    rewritten = (REWRITE_PROMPT | llm | StrOutputParser()).invoke({
        'question': state['question']
    })
    print(f'  [재작성] {rewritten[:80]}')
    return {
        **state,
        'rewritten_question': rewritten,
        'rewrite_count':      state.get('rewrite_count', 0) + 1,
    }


# ── Node 6a · generate_answer ───────────────────────────────────

def generate_answer(state: CignaRAGState) -> CignaRAGState:
    """Node 6a: 정상 경로 — Term Locking + 출처 인용 답변."""
    docs     = state.get('retrieved_docs', [])
    context  = format_docs(docs)
    plan_ctx = ''
    if state.get('plan_info'):
        p = state['plan_info']
        plan_ctx = (
            f"\n사용자 플랜: {p.get('plan_tier', '?')} | "
            f"공제액 ${p.get('deductible', '?')} (사용 ${p.get('deductible_used', 0)}) | "
            f"Co-insurance {p.get('cost_share_pct', '?')}%"
        )
    answer = (rag_prompt | llm | StrOutputParser()).invoke({
        'context':   context,
        'question':  state['question'],
        'term_lock': TERM_LOCK_TABLE + plan_ctx,
    })
    return {**state, 'answer': answer}


# ── Node 6b · hyde_fallback ─────────────────────────────────────

FALLBACK_TEMPLATE = """
당신은 Cigna 보험 안내 어시스턴트입니다.
아래 정보가 부족하여 정확한 답변이 어렵습니다: {missing_info}

그러나 현재 검색된 문서를 바탕으로 가능한 범위에서 답변하겠습니다.
{term_lock}

⚠️ 부족한 정보: {missing_info}
   → 정확한 답변을 위해 위 정보를 질문에 포함해 다시 물어보세요.

[참고 문서]\n{context}\n\n[질문]\n{question}
"""


def hyde_fallback(state: CignaRAGState) -> CignaRAGState:
    """Node 6b: HyDE 폴백 — 부족 정보 안내 + 가상 답변."""
    missing = state.get('missing_info', [])
    context = format_docs(state.get('retrieved_docs', []))

    answer = (PromptTemplate.from_template(FALLBACK_TEMPLATE) | llm | StrOutputParser()).invoke({
        'missing_info': ', '.join(missing) if missing else '없음',
        'context':      context,
        'question':     state['question'],
        'term_lock':    TERM_LOCK_TABLE,
    })
    note = (
        f'\n\n---\n⚠️ 다음 정보를 제공하면 더 정확한 답변이 가능합니다: {", ".join(missing)}'
        if missing else ''
    )
    return {**state, 'answer': answer + note}


# ══════════════════════════════════════════════════════════════════
# Section 8 · LangGraph 그래프 조립
# ══════════════════════════════════════════════════════════════════

def route_after_classify(state: CignaRAGState) -> str:
    """classify_question 후 분기: HyDE or 난이도별 검색."""
    return 'hyde_retrieve' if state.get('needs_clarification') else 'retrieve_by_difficulty'


builder = StateGraph(CignaRAGState)

builder.add_node('classify_question',      classify_question)
builder.add_node('hyde_retrieve',          hyde_retrieve)
builder.add_node('retrieve_by_difficulty', retrieve_by_difficulty)
builder.add_node('rewrite_query',          rewrite_query)
builder.add_node('generate_answer',        generate_answer)
builder.add_node('hyde_fallback',          hyde_fallback)

builder.add_edge(START, 'classify_question')

builder.add_conditional_edges(
    'classify_question',
    route_after_classify,
    {'hyde_retrieve': 'hyde_retrieve', 'retrieve_by_difficulty': 'retrieve_by_difficulty'},
)

for retrieve_node in ['hyde_retrieve', 'retrieve_by_difficulty']:
    builder.add_conditional_edges(
        retrieve_node,
        grade_documents_node,
        {'generate_answer': 'generate_answer',
         'rewrite_query':   'rewrite_query',
         'hyde_fallback':   'hyde_fallback'},
    )

builder.add_edge('rewrite_query',   'retrieve_by_difficulty')
builder.add_edge('generate_answer', END)
builder.add_edge('hyde_fallback',   END)

cigna_graph = builder.compile()
print('✅ LangGraph 그래프 컴파일 완료')


# ══════════════════════════════════════════════════════════════════
# Section 9 · 파이프라인 실행 헬퍼
# ══════════════════════════════════════════════════════════════════

def ask_cigna(question: str, plan_info: Optional[dict] = None) -> str:
    """LangGraph 파이프라인 실행 헬퍼.

    Args:
        question:  사용자 질문
        plan_info: 사용자 플랜 정보 (없으면 DEFAULT_PLAN 사용)

    Returns:
        최종 답변 문자열
    """
    init_state: CignaRAGState = {
        'question':            question,
        'plan_info':           plan_info or {},
        'difficulty':          '',
        'needs_clarification': False,
        'missing_info':        [],
        'retrieved_docs':      [],
        'hyde_query':          '',
        'rewrite_count':       0,
        'answer':              '',
    }
    result = cigna_graph.invoke(init_state)
    return result['answer']
