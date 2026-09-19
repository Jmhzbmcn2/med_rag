from medical_rag.llm import chat
from medical_rag.retrieval import Hit

SYSTEM_PROMPT = (
    "Bạn là trợ lý y khoa nói tiếng Việt. Chỉ trả lời dựa trên các đoạn tài liệu được đánh số trong tin nhắn của người dùng. "
    "Sau mỗi ý lấy từ tài liệu, ghi số nguồn dạng [n]. "
    "Nếu tài liệu không đủ thông tin, hãy nói rõ là chưa có đủ thông tin, không tự suy đoán. "
    "Không chẩn đoán bệnh cho người dùng; khuyên đi khám bác sĩ khi triệu chứng nặng hoặc kéo dài."
)
NO_RESULT = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(
        f"[{n}] {hit.article_title} — {hit.section_path}\n{hit.text}" for n, hit in enumerate(hits, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"},
    ]


def answer(retriever, question: str, k: int = 5) -> dict:
    hits = retriever.search(question, k=k)
    if not hits:
        return {"answer": NO_RESULT, "sources": []}
    sources = [
        {
            "n": n,
            "title": hit.article_title,
            "url": hit.article_url,
            "section_path": hit.section_path,
            "type": hit.type,
            "text": hit.text,
            "score": hit.score,
        }
        for n, hit in enumerate(hits, 1)
    ]
    return {"answer": chat(build_messages(question, hits)), "sources": sources}
