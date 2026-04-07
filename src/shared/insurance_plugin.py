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