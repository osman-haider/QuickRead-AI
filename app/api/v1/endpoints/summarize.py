from fastapi import APIRouter, HTTPException
from app.models.schemas import SummarizeRequest
from app.services.llm_service import summarize_text

router = APIRouter()

@router.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        summary = summarize_text(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))