import logging
from typing import Dict, Any

class ModelGateway:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            "fast": "llama-3-8b",
            "reasoning": "deepseek-r1",
            "heavy": "llama-3-70b",
            "coding": "deepseek-coder"
        }

    def route_request(self, payload: Dict[str, Any]) -> str:
        prompt = payload.get("prompt", "").lower()
        
        # Simple heuristic routing
        if any(kw in prompt for kw in ["code", "python", "javascript", "refactor"]):
            target = self.models["coding"]
        elif any(kw in prompt for kw in ["analyze", "reason", "math", "logic"]):
            target = self.models["reasoning"]
        elif len(prompt) > 2000:
            target = self.models["heavy"]
        else:
            target = self.models["fast"]
            
        self.logger.info(f"Routing prompt to model: {target}")
        return target

    async def forward_request(self, target_model: str, data: Dict[str, Any]):
        # Implementation for proxying to Triton or vLLM
        pass
