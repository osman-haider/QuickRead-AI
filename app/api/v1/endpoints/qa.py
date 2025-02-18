from fastapi import APIRouter, HTTPException
from app.models.schemas import QARequest
from app.services.llm_service import answer_question

router = APIRouter()

@router.post("/qa")
async def qa(request: QARequest):
    try:
        answer = answer_question(request.text, request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))