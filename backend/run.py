from langchain_groq import ChatGroq

from backend.config import settings

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.model_name,
    temperature=0.7,
)

response = llm.invoke("Write a 3-sentence intro about what LangGraph is.")
print(response.content)
