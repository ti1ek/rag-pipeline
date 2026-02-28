from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import OPENAI_API_KEY

SYSTEM_PROMPT = """Ты — эксперт-аналитик, отвечающий на вопросы по годовым отчётам казахстанских компаний.
Отвечай строго на основе предоставленного контекста. Если информации недостаточно, скажи об этом.
Отвечай на русском языке, кратко и точно. Указывай конкретные цифры и факты из контекста."""


def generate_answer(
    question: str,
    context_docs: list[Document],
    model: str = "gpt-4o-mini",
) -> str:
    """Generate an answer using GPT-4o-mini given retrieved context documents."""
    llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=0)

    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Контекст:\n{context}\n\nВопрос: {question}"),
    ]

    response = llm.invoke(messages)
    return response.content
