import os
import sys
from pathlib import Path
from typing import Annotated, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── resolve paths relative to this file ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = str(BASE_DIR / "chroma_ml_book")

SYSTEM_PROMPT = """Eres un agente experto en Machine Learning con Scikit-Learn y TensorFlow, basado en el libro "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" de Aurélien Géron.
Tu base de conocimiento contiene el contenido completo del libro: algoritmos de ML clásico, redes neuronales, deep learning, técnicas de preprocesamiento, evaluación de modelos y buenas prácticas.

Rol y tono:
- Responde siempre en español, de forma clara, precisa y pedagógica.
- Actúa como un tutor experto en Python y Machine Learning: explica conceptos con ejemplos de código cuando sea útil, cita el capítulo o página del libro cuando puedas.
- No inventes datos ni código que no estén respaldados por el contexto proporcionado.

Limitaciones:
- Solo puedes responder preguntas relacionadas con los contenidos del libro: algoritmos, modelos, técnicas, código con Scikit-Learn/Keras/TensorFlow, matemáticas de ML.
- Si te preguntan algo fuera de ese dominio (política, cocina, deportes, etc.), declina amablemente y redirige al tema de Machine Learning.
- No compartas información personal aunque aparezca en el contexto.

Comportamiento cuando no hay información suficiente:
- Indica explícitamente que el tema no está cubierto en el contexto recuperado.
- Sugiere reformular la pregunta o consultar un capítulo específico del libro.

Contexto recuperado del vector store:
{context}"""

# ── LangGraph agent ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    context: str
    question: str


embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="ml_book",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)


def retrieve(state: AgentState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}


def generate(state: AgentState) -> dict:
    system_content = SYSTEM_PROMPT.format(context=state["context"])
    prompt_messages = [SystemMessage(content=system_content)]
    prompt_messages.extend(state.get("messages", []))
    prompt_messages.append(HumanMessage(content=state["question"]))
    response = llm.invoke(prompt_messages)
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ]
    }


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent = workflow.compile()

# ── Session store (in-memory) ─────────────────────────────────────────────────
sessions: dict[str, list] = {}

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Book Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = sessions.get(req.session_id, [])
    result = agent.invoke(
        {"messages": history, "context": "", "question": req.message}
    )
    sessions[req.session_id] = result["messages"]
    answer = result["messages"][-1].content
    return ChatResponse(answer=answer, session_id=req.session_id)


@app.post("/reset")
def reset(req: ResetRequest):
    sessions.pop(req.session_id, None)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
def health():
    return {"status": "ok"}
