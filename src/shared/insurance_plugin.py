# insurance_plugin.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document


class InsurancePlugin(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    def plans(self) -> List[str]:
        return []

    @abstractmethod
    def retrieve(
        self,
        query: str,
        normalized: dict,
        plan_or_intent: Optional[str],
        **kwargs
    ) -> List[Document]:
        pass

    @abstractmethod
    def analyze(
        self,
        question: str,
        context_str: str,
        state: Dict[str, Any],
    ) -> dict:
        """보험사별 분석. 표준 dict 반환."""
        pass

    # insurance_plugin.py에 추가
    @property
    def clarification_style(self) -> str:
        """공통 clarification 스타일 가이드 - analyze 프롬프트에 주입"""
        return """
    [CLARIFICATION MESSAGE STYLE - CRITICAL]
    - If user says they don't know or can't remember:
      * Acknowledge it first with empathy
      * Explain WHY the info is needed
      * Offer an alternative path
    - NEVER use the exact same phrasing as the previous clarification message
    - Each clarification must feel like a natural conversation, not a form
    - clarification_message MUST be in the SAME language as the user message
    """