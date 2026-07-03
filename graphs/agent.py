from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class AgentState(TypedDict):
    ticket: str
    is_clear: bool
    chunks: List[Document]
    response: str
    requires_escalation: bool

def check_clarity(state: AgentState) -> AgentState:
    prompt = f"""Is this support ticket clear enough to give response on?
    Ticket: {state["ticket"]}
    Reply with a YES or NO only.
    """
    result = llm.invoke(prompt)

    if result.content.strip().upper == "YES":
        state["is_clear"] = True

    else:
        state["is_clear"] = False

    return state