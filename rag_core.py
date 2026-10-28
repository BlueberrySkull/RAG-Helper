"""Ядро RAG-пайплайна

Модели (эмбеддер, LLM) принимаются функциями как параметры - это позволяет
тестам подставлять лёгкие фейки вместо реальных моделей (см. test_rag_core.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RELEVANCE_THRESHOLD = 0.5
NOT_FOUND_ANSWER = "В инструкциях этого нет, обратитесь в техническую поддержку"

SYSTEM_PROMPT = (
    "Ты - помощник по внутренней информационной системе ГБУ БДД. "
    "Отвечай на вопрос сотрудника подробно и по шагам, используя только "
    "приведённый ниже контекст из инструкций - объясни весь процесс, а не "
    "только один короткий факт. Ответ должен быть минимум 2-3 предложения. "
    "Если в контексте нет ответа на вопрос, честно скажи, что в инструкциях "
    "этого нет, и посоветуй обратиться в техническую поддержку. "
    "Не придумывай факты, которых нет в контексте."
)


@dataclass
class Chunk:
    text: str
    source: str  # имя файла в knowledge_base, откуда взят фрагмент


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # (n_chunks, dim), нормализованные эмбеддинги


def load_documents(kb_path: str | Path) -> list[Chunk]:
    """Читает knowledge_base/*.txt и нарезает каждый файл на фрагменты по заголовкам '## '."""
    chunks: list[Chunk] = []
    for file in sorted(Path(kb_path).glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        for section in re.split(r"\n(?=## )", text):
            section = section.strip()
            if section.startswith("##"):
                chunks.append(Chunk(text=section, source=file.name))
    return chunks


def embed_texts(texts: list[str], embedder) -> np.ndarray:
    """Обёртка над sentence-transformers: принимает уже загруженную модель."""
    return np.asarray(embedder.encode(list(texts), normalize_embeddings=True))


def build_index(kb_path: str | Path, embedder) -> Index:
    chunks = load_documents(kb_path)
    vectors = embed_texts([c.text for c in chunks], embedder)
    return Index(chunks=chunks, vectors=vectors)


def search(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int = 3,
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[int]:
    """Косинусное сходство (векторы уже нормализованы), топ-k, отсечение по порогу."""
    if len(doc_vecs) == 0:
        return []
    scores = doc_vecs @ np.asarray(query_vec)
    top_k = np.argsort(-scores)[:k]
    return [int(i) for i in top_k if scores[i] >= threshold]


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    user = f"Контекст:\n{context}\n\nВопрос: {question}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def answer(question: str, index: Index, embedder, llm) -> dict:
    """Связывает пайплайн: поиск -> (если нашлось) промпт -> генерация -> ответ + источники."""
    query_vec = embed_texts([question], embedder)[0]
    hit_indices = search(query_vec, index.vectors)
    if not hit_indices:
        return {"answer": NOT_FOUND_ANSWER, "sources": []}

    chunks = [index.chunks[i] for i in hit_indices]
    prompt = build_prompt(question, chunks)
    completion = llm(prompt, max_tokens=512, stop=["<|im_end|>"])
    text = completion["choices"][0]["text"].strip()
    sources = sorted({c.source for c in chunks})
    return {"answer": text, "sources": sources}


def load_embedder(model_path: str | Path = "models/embedder"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path))


def load_llm(model_path: str | Path = "models/llm/meno-tiny-0.1-fp16.gguf"):
    from llama_cpp import Llama

    return Llama(model_path=str(model_path), n_ctx=4096, verbose=False)
