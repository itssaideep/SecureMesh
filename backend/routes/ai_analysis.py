from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class AnalysisRequest(BaseModel):
    ip: str
    commands: List[str]

@router.post("/summarize")
async def summarize_attack(request: AnalysisRequest):
    # TODO: Connect to OpenAI API or local model
    # For now, return a mock response matching the PDF example
    return {
        "summary": "Possible malware deployment attempt detected. Attacker downloaded executable shell script.",
        "risk_level": "High"
    }
