"""Lightweight reranker server — BGE-reranker-base via transformers.

Replaces vLLM for dev. Saves ~1.4GB RAM vs vLLM.
Usage: python scripts/reranker_server.py
Env vars: RERANK_MODEL_PATH, RERANK_PORT (default 8002), RERANK_DEVICE (default cuda)
"""

from __future__ import annotations

import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.environ.get(
    "RERANK_MODEL_PATH", "E:/BaiduNetdiskDownload/nlp/models/BAAI/bge-reranker-base"
)
PORT = int(os.environ.get("RERANK_PORT", "8002"))
DEVICE = os.environ.get("RERANK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Reranker Server")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to(DEVICE).eval()


def _score(query: str, documents: list[str]) -> list[float]:
    pairs = [[query, doc] for doc in documents]
    with torch.no_grad():
        encoded = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**encoded).logits
        scores = logits.squeeze(-1).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores


class RerankRequest(BaseModel):
    model: str = "/model"
    query: str
    documents: list[str]


@app.post("/v1/rerank")
async def rerank(req: RerankRequest) -> dict:
    scores = _score(req.query, req.documents)
    results = [
        {"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"results": results}


@app.get("/models")
async def models():
    return {"data": [{"id": "/model", "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_PATH}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
