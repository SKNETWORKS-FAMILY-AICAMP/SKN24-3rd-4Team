"""
insurance_rag.py 에서 analyze_node 부분만 교체
"""

def analyze_node(self, state: InsuranceState) -> dict:
    messages = state['messages']
    question = messages[-1].content

    history = []
    for m in messages[:-1]:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        history.append(f"{role}: {m.content}")
    context_str = "\n".join(history[-5:])

    # ── 각 플러그인의 analyze() 호출 ──────────────────────────
    analysis = self.plugin.analyze(
        question=question,
        context_str=context_str,
        state={
            "plan_or_intent":    state.get("plan_or_intent"),
            "known_treatment":   state.get("known_treatment"),
            "slots":             state.get("slots", {}),
            "needs_clarification": False,
            "clarification_message": state.get("clarification_msg", ""),
            "followup_count":    state.get("followup_count", 0),
            "max_followups":     state.get("max_followups", 2),
            "extra":             state.get("extra", {}),
        }
    )

    # plan / treatment 누적 (플러그인이 이미 처리했지만 state 반영)
    final_plan      = analysis.get("plan_or_intent")
    final_treatment = analysis.get("known_treatment")

    return {
        "normalized_query":  analysis,
        "plan_or_intent":    final_plan,
        "known_treatment":   final_treatment,
        "current_question":  question,
        "clarification_msg": analysis.get("clarification_message", ""),
        "slots":             analysis.get("extra", {}).get("slots", state.get("slots", {})),
        "followup_count":    analysis.get("extra", {}).get("followup_count", state.get("followup_count", 0)),
        "extra":             analysis.get("extra", {}),
    }


# ── InsuranceState에 추가할 필드 ──────────────────────────────────
class InsuranceState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    plan_or_intent:    Optional[str]
    known_treatment:   Optional[str]
    normalized_query:  Dict[str, Any]
    retrieved_docs:    List[str]
    current_question:  str
    clarification_msg: str
    slots:             Dict[str, Any]   # Allianz slot 시스템용
    followup_count:    int              # followup 횟수 추적
    max_followups:     int              # 최대 followup 허용 수
    extra:             Dict[str, Any]   # 보험사별 추가 데이터


# ── retrieve_node도 extra 전달 ────────────────────────────────────
def retrieve_node(self, state: InsuranceState) -> dict:
    query = state['normalized_query'].get('english_query') or state['current_question']
    plan  = state.get('plan_or_intent')

    docs = self.plugin.retrieve(
        query=query,
        normalized=state['normalized_query'],
        plan_or_intent=plan,
        extra=state.get('extra', {}),       # ← 보험사별 추가 데이터 전달
    )

    formatted = []
    for d in docs:
        source_path = d.metadata.get('source', 'document_unknown.pdf')
        file_name   = Path(source_path).name
        page_num    = d.metadata.get('page', '?')
        formatted.append(f"[Source: {file_name} / Page: {page_num}] {d.page_content}")

    return {"retrieved_docs": formatted}
