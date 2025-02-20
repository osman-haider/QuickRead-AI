from fastapi import APIRouter, HTTPException
from app.models.schemas import QARequest
from app.services.qa_service import get_answer

router = APIRouter()

@router.post("/qa")
async def qa(request: QARequest):
    try:
        answer, context = get_answer(request.question)
        return {"answer": answer, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))