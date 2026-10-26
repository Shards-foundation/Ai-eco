from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    prompt: str
    context_id: Optional[str] = None

class QueryResponse(BaseModel):
    id: str
    answer: str
    sources: List[str]
