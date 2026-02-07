from __future__ import annotations
from pydantic import BaseModel


class DocumentInput(BaseModel):
    title: str
    content: str


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    priority: str  # high / medium / low
    dependencies: list[str] = []


class RiskItem(BaseModel):
    description: str
    category: str  # risk | open_question | assumption | missing_data
    severity: str  # high / medium / low


class AnalysisResult(BaseModel):
    summary: str
    key_decisions: list[str]
    action_items: list[ActionItem]
    risks: list[RiskItem]


class DocumentResponse(BaseModel):
    id: int
    title: str
    word_count: int
    created_at: str


class DocumentDetail(DocumentResponse):
    content: str
    analysis: AnalysisResult | None = None
    analysis_status: str | None = None
