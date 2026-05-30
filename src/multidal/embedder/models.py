from dataclasses import dataclass

import numpy as np


@dataclass
class Embedding:
    vector: np.ndarray
    model_name: str
    dim: int


@dataclass
class EmbeddedChunk:
    chunk_id: str
    content: str
    embedding: Embedding
    modality: str  # "text" | "image"
    metadata: dict
