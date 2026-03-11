"""
07_langgraph_agent.py — Multi-node LangGraph agent powered by Gemini
Covers: StateGraph, nodes, conditional edges, tool integration, human-in-the-loop

Graph topology:
  START → planner → executor → evaluator → (loop or END)
"""
from typing import TypedDict, Annotated, Literal
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os

# ── LLM Setup ─────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)


# ── Tools ──────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns mock results."""
    # Replace with SerpAPI / Tavily in production
    return f"[Mock search results for '{query}']: Found 5 relevant articles about {query}."


@tool
def code_executor(code: str) -> str:
    """Execute Python code safely and return output."""
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len}})
        return output.getvalue() or "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"


@tool
def summarize_text(text: str, max_words: int = 100) -> str:
    """Summarize a long text into a concise version."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


TOOLS = [web_search, code_executor, summarize_text]
llm_with_tools = llm.bind_tools(TOOLS)


# ── State Definition ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:      Annotated[list[BaseMessage], operator.add]
    plan:          str
    current_step:  int
    max_steps:     int
    final_answer:  str
    task:          str


# ── Nodes ──────────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    """Break the task into steps and store the plan."""
    plan_prompt = f"""
    You are a task planner. Given the task below, create a concise step-by-step plan.
    Respond with ONLY a numbered list. Max 5 steps.
    
    Task: {state['task']}
    """
    response = llm.invoke([HumanMessage(content=plan_prompt)])
    print(f"📋 Plan:\n{response.content}\n")
    return {
        "plan":         response.content,
        "current_step": 0,
        "messages":     [AIMessage(content=f"Plan created:\n{response.content}")],
    }


def executor_node(state: AgentState) -> dict:
    """Execute the next step using tools if needed."""
    step = state["current_step"] + 1
    messages = state["messages"] + [
        SystemMessage(content=f"Execute step {step} of the plan. Use tools as needed."),
        HumanMessage(content=f"Task: {state['task']}\nPlan:\n{state['plan']}\nNow execute step {step}."),
    ]
    response = llm_with_tools.invoke(messages)
    print(f"⚙️  Step {step} executor response: {response.content[:200]}...")
    return {
        "messages":     [response],
        "current_step": step,
    }


def evaluator_node(state: AgentState) -> dict:
    """Decide: continue with more steps or produce final answer."""
    eval_prompt = f"""
    Task: {state['task']}
    Steps completed: {state['current_step']} / {state['max_steps']}
    
    Conversation so far:
    {chr(10).join([f"{m.type}: {m.content[:200]}" for m in state['messages'][-4:]])}
    
    Has the task been completed? Reply with exactly one word: YES or NO.
    If YES, also provide the final answer on the next line.
    """
    response = llm.invoke([HumanMessage(content=eval_prompt)])
    lines     = response.content.strip().split("\n", 1)
    completed = lines[0].strip().upper().startswith("YES")
    answer    = lines[1].strip() if len(lines) > 1 else state["messages"][-1].content

    print(f"🔍 Evaluator: Completed={completed}")
    return {"final_answer": answer if completed else ""}


# ── Routing Logic ──────────────────────────────────────────────────────────
def route_after_evaluator(state: AgentState) -> Literal["executor_node", "__end__"]:
    """Continue looping or end based on evaluator result."""
    if state.get("final_answer") or state["current_step"] >= state["max_steps"]:
        return END
    return "executor_node"


# ── Graph Assembly ─────────────────────────────────────────────────────────
def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner_node",  planner_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("tool_node",     ToolNode(TOOLS))
    graph.add_node("evaluator_node", evaluator_node)

    # Edges
    graph.add_edge(START,              "planner_node")
    graph.add_edge("planner_node",     "executor_node")
    graph.add_edge("executor_node",    "evaluator_node")
    graph.add_conditional_edges(
        "evaluator_node",
        route_after_evaluator,
        {"executor_node": "executor_node", END: END},
    )

    return graph.compile()


# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = build_agent_graph()

    # Example task
    task = "Research the top 3 vector databases for production RAG systems and compare them."

    initial_state: AgentState = {
        "messages":     [HumanMessage(content=task)],
        "plan":         "",
        "current_step": 0,
        "max_steps":    4,
        "final_answer": "",
        "task":         task,
    }

    print(f"🚀 Running agent for task: {task}\n{'='*60}")
    final_state = agent.invoke(initial_state)

    print(f"\n{'='*60}")
    print("✅ FINAL ANSWER:")
    print(final_state.get("final_answer", "No final answer produced."))
    print(f"\nTotal steps: {final_state['current_step']}")
