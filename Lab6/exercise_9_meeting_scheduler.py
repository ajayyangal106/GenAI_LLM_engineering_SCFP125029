from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

load_dotenv()

@tool
def check_calendar_availability(person: str, date: str) -> str:
    """Check a person's calendar availability on a date."""
    return f"{person} is available on {date}."

@tool
def get_timezone_offset(city: str) -> str:
    """Return a simple timezone offset for a city."""
    zones = {"Mumbai": "UTC+5:30", "Delhi": "UTC+5:30", "London": "UTC+0", "Dubai": "UTC+4"}
    return zones.get(city.title(), "Timezone not found")

@tool
def book_meeting(person: str, date: str, time: str) -> str:
    """Book a meeting with a person."""
    return f"Meeting booked with {person} on {date} at {time}."

tools = [check_calendar_availability, get_timezone_offset, book_meeting]
model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(tools)

def run_query(query):
    messages = [{"role": "user", "content": query}]
    response = model_with_tools.invoke(messages)
    print("
Query:", query)
    print("Suggested tool calls:", response.tool_calls)
    messages.append(response)
    tool_map = {t.name: t for t in tools}
    for call in response.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        print("Tool result:", result)
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    final = model.invoke(messages)
    print("Final answer:", final.content)

run_query(input("Availability query: "))
run_query(input("Full booking query: "))
