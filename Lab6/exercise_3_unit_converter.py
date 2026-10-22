from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

load_dotenv()

@tool
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert Celsius to Fahrenheit."""
    return f"{celsius}°C = {(celsius * 9 / 5) + 32:.2f}°F"

@tool
def fahrenheit_to_celsius(fahrenheit: float) -> str:
    """Convert Fahrenheit to Celsius."""
    return f"{fahrenheit}°F = {((fahrenheit - 32) * 5 / 9):.2f}°C"

@tool
def kilometers_to_miles(kilometers: float) -> str:
    """Convert kilometers to miles."""
    return f"{kilometers} km = {kilometers * 0.621371:.2f} miles"

@tool
def miles_to_kilometers(miles: float) -> str:
    """Convert miles to kilometers."""
    return f"{miles} miles = {miles / 0.621371:.2f} km"

tools = [celsius_to_fahrenheit, fahrenheit_to_celsius, kilometers_to_miles, miles_to_kilometers]
model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(tools)

user_query = input("What conversion do you need? ")
messages = [{"role": "user", "content": user_query}]
response = model_with_tools.invoke(messages)

print("
Suggested tool calls:")
print(response.tool_calls)

for call in response.tool_calls:
    tool_map = {t.name: t for t in tools}
    result = tool_map[call["name"]].invoke(call["args"])
    print("Tool result:", result)
    messages.append(response)
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final answer:")
print(final_response.content)
