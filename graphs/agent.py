from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from chains.retriever import get_retriever
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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

    if result.content.strip().upper() == "YES":
        state["is_clear"] = True

    else:
        state["is_clear"] = False

    return state

def clarify(state: AgentState) -> AgentState:
    prompt=f"""Please ask the user to clarify the message they are trying to ask in this ticket.
    Ticket:{state["ticket"]}
    Reply with a reply asking the user , return a string only with the reply contained in it.
    """

    result = llm.invoke(prompt)
    state["response"] = result.content

    return state

def escalate(state: AgentState) -> AgentState:
    state["response"] = "Your issue ticket is being handed over to our support staff."
    return state

def retrieve_and_resolve(state: AgentState) -> AgentState:
    retriever = get_retriever()
    state["chunks"] = retriever.invoke(state["ticket"])
    prompt=f"""Resolve this ticket's issue using the given chunks of data
    Ticket:{state["ticket"]}
    Chunks:{state["chunks"]}
    Reply only if you find the answer or have knowledge about it, or else just say "NO"
    """
    result = llm.invoke(prompt)
    if(result.content.strip().upper() == "NO"):
        state["requires_escalation"] = True
    else:
        state["response"] = result.content
        state["requires_escalation"] = False

    return state

# these two are routing functions required for conditional edges, i read docs
def route_check_clarity(state: AgentState) -> str:
    if(state["is_clear"] == False):
        return "clarify"
    else:
        return "retrieve_and_resolve"
    
def route_check_escalation(state: AgentState) -> str:
    if(state["requires_escalation"] == False):
        return END
    else:
        return "escalate"

graph = StateGraph(AgentState)

graph.add_node("check_clarity", check_clarity)
graph.add_node("clarify", clarify)
graph.add_node("retrieve_and_resolve", retrieve_and_resolve)
graph.add_node("escalate", escalate)

graph.add_edge(START, "check_clarity")
graph.add_conditional_edges("check_clarity", route_check_clarity)
graph.add_edge("clarify", END)
graph.add_conditional_edges("retrieve_and_resolve", route_check_escalation)
graph.add_edge("escalate", END)

app = graph.compile()