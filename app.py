import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Tools ---
@tool
def lookup_documentation(library: str) -> str:
    """Look up quick reference syntax and overview documentation for popular programming libraries."""
    docs = {
        "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python based on standard type hints.",
        "pandas": "Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.",
        "requests": "Requests allows you to send organic, highly ergonomic HTTP/1.1 requests in Python easily."
    }
    return docs.get(library.lower(), "Documentation not found for that library.")

@tool
def count_lines_of_code(code: str) -> int:
    """Counts the total number of lines in a given code string."""
    if not code:
        return 0
    return len(code.splitlines())

@tool
def analyze_code_snippet(code: str) -> str:
    """Analyze a given Python code snippet for basic formatting and syntax check suggestions."""
    issues = []
    if "print " in code:
        issues.append("Python 3 uses print() as a function, avoid print statement spacing.")
    if "eval(" in code:
        issues.append("Security Warning: Use of eval() is dangerous and discouraged.")

    result = {
        "length_chars": len(code),
        "detected_issues": issues if issues else ["No critical anti-patterns detected in quick scan."]
    }
    return json.dumps(result)

tools = [lookup_documentation, count_lines_of_code, analyze_code_snippet]

# --- 2. Initialize Model & Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized real-time AI coding assistant agent restricted ONLY to programming, code debugging, and library documentation. "
        "For any other roles, topics, questions, or general knowledge outside of coding and software development, "
        "you must say exactly: 'I am not authorized to answer questions outside of real-time AI coding assistance.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(
    title="Real-Time AI Coding Assistant Server",
    version="1.0",
    description="An API server providing real-time AI coding assistance, debugging, and docs using LangChain and Gemini."
)

add_routes(
    app,
    formatted_agent_chain,
    path="/coding-assistant"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
