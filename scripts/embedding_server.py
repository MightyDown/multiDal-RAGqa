"""Lightweight embedding server — Qwen3-Embedding-0.6B via transformers.

Replaces vLLM for dev. Saves ~500MB RAM vs vLLM.
Usage: python scripts/embedding_server.py
Env vars: EMBED_MODEL_PATH, EMBED_PORT (default 8000), EMBED_DEVICE (default cuda)
"""

from __future__ import annotations

import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "E:/BaiduNetdiskDownload/nlp/models/Qwen/Qwen3-Embedding-0.6B")
PORT = int(os.environ.get("EMBED_PORT", "8000"))
DEVICE = os.environ.get("EMBED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Embedding Server")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE).eval()


def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def _embed(texts: list[str]) -> list[list[float]]:
    with torch.no_grad():
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)
        outputs = model(**encoded)
        pooled = _mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()


class EmbedRequest(BaseModel):
    model: str = "/model"
    input: str | list[str]

class EmbedResponse(BaseModel):
    data: list[dict]


@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest) -> EmbedResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = _embed(texts)
    return EmbedResponse(
        data=[{"embedding": v, "index": i} for i, v in enumerate(vecs)]
    )


@app.get("/models")
async def models():
    return {"data": [{"id": "/model", "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_PATH}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
