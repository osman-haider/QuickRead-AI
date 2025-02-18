from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    text: str
    question: str