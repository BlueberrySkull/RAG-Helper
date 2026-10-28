"""Тесты rag_core.py на фейковых эмбеддере и LLM - без скачивания весов."""
from __future__ import annotations

import numpy as np

import rag_core
from rag_core import NOT_FOUND_ANSWER, Chunk, Index


class FakeEmbedder:
    """Возвращает заранее заданный вектор для каждого известного текста."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 3):
        self._vectors = vectors
        self._dim = dim

    def encode(self, texts, normalize_embeddings=True):
        out = np.array(
            [self._vectors.get(t, [0.0] * self._dim) for t in texts], dtype=float
        )
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt: str | None = None

    def __call__(self, prompt, max_tokens=512, stop=None):
        self.last_prompt = prompt
        return {"choices": [{"text": self.response_text}]}


def write_kb_file(tmp_path, name: str, content: str):
    (tmp_path / name).write_text(content, encoding="utf-8")


def test_load_documents_splits_into_chunks_by_section(tmp_path):
    write_kb_file(
        tmp_path,
        "otpuska.txt",
        "# Заявки на отпуск\n\n"
        "## Как создать заявку\n\nСоздать новую заявку.\n\n"
        "## Как отправить на согласование\n\nОтправить из статуса «Новая».\n",
    )

    chunks = rag_core.load_documents(tmp_path)

    assert len(chunks) == 2
    assert all(c.source == "otpuska.txt" for c in chunks)
    assert chunks[0].text.startswith("## Как создать заявку")
    assert chunks[1].text.startswith("## Как отправить на согласование")


def test_load_documents_skips_title_only_preamble(tmp_path):
    write_kb_file(tmp_path, "smeny.txt", "# Просто заголовок без разделов\n")

    chunks = rag_core.load_documents(tmp_path)

    assert chunks == []


def test_search_returns_topk_above_threshold():
    doc_vecs = np.array(
        [
            [1.0, 0.0],  # совпадает с запросом
            [0.9, 0.1],  # похож
            [0.0, 1.0],  # ортогонален - должен быть отфильтрован
        ]
    )
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    query_vec = np.array([1.0, 0.0])

    result = rag_core.search(query_vec, doc_vecs, k=3, threshold=0.5)

    assert result == [0, 1]


def test_search_returns_empty_list_when_nothing_relevant():
    doc_vecs = np.array([[0.0, 1.0], [0.0, -1.0]])
    query_vec = np.array([1.0, 0.0])

    result = rag_core.search(query_vec, doc_vecs, k=3, threshold=0.5)

    assert result == []


def test_build_prompt_contains_context_and_question():
    chunks = [Chunk(text="## Раздел\n\nТекст раздела.", source="tabeli.txt")]

    prompt = rag_core.build_prompt("Как отправить табель?", chunks)

    assert "Как отправить табель?" in prompt
    assert "Текст раздела." in prompt
    assert "[tabeli.txt]" in prompt


def test_answer_returns_not_found_when_no_relevant_chunks():
    chunk = Chunk(text="## Раздел\n\nТекст.", source="tabeli.txt")
    index = Index(chunks=[chunk], vectors=np.array([[0.0, 1.0]]))
    embedder = FakeEmbedder({"Как приготовить борщ?": [1.0, 0.0]}, dim=2)
    llm = FakeLLM("не должен вызываться")

    result = rag_core.answer("Как приготовить борщ?", index, embedder, llm)

    assert result == {"answer": NOT_FOUND_ANSWER, "sources": []}
    assert llm.last_prompt is None


def test_answer_returns_llm_text_and_sources_when_relevant():
    chunk = Chunk(text="## Как отправить табель\n\nОписание.", source="tabeli.txt")
    index = Index(chunks=[chunk], vectors=np.array([[1.0, 0.0]]))
    embedder = FakeEmbedder({"Как отправить табель?": [1.0, 0.0]}, dim=2)
    llm = FakeLLM("Табель отправляется из карточки отдела.")

    result = rag_core.answer("Как отправить табель?", index, embedder, llm)

    assert result["answer"] == "Табель отправляется из карточки отдела."
    assert result["sources"] == ["tabeli.txt"]
    assert "Как отправить табель?" in llm.last_prompt


def test_build_index_embeds_all_chunk_texts(tmp_path):
    write_kb_file(
        tmp_path, "smeny.txt", "# Смены\n\n## Как создать смену\n\nОписание.\n"
    )
    embedder = FakeEmbedder({"## Как создать смену\n\nОписание.": [1.0, 0.0]}, dim=2)

    index = rag_core.build_index(tmp_path, embedder)

    assert len(index.chunks) == 1
    assert index.vectors.shape == (1, 2)
