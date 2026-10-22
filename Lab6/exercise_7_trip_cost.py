from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

load_dotenv()

@tool
def get_flight_cost(origin: str, destination: str) -> str:
    """Give an approximate return flight cost between two cities."""
    routes = {("Delhi", "Goa"): 7000, ("Mumbai", "Goa"): 4500, ("Pune", "Goa"): 5000}
    price = routes.get((origin.title(), destination.title()), 8000)
    return f"Approximate flight cost: ₹{price}"

@tool
def get_hotel_cost(city: str, nights: int) -> str:
    """Give an approximate hotel cost for the given number of nights."""
    rates = {"Goa": 2500, "Mumbai": 3000, "Pune": 2200, "Delhi": 2800}
    price = rates.get(city.title(), 2500) * nights
    return f"Approximate hotel cost for {nights} nights: ₹{price}"

@tool
def get_food_budget(city: str, days: int) -> str:
    """Give an approximate food budget for a trip."""
    rates = {"Goa": 800, "Mumbai": 900, "Pune": 700, "Delhi": 850}
    price = rates.get(city.title(), 800) * days
    return f"Approximate food budget for {days} days: ₹{price}"

tools = [get_flight_cost, get_hotel_cost, get_food_budget]
model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(tools)

user_query = input("Ask for a trip cost estimate: ")
messages = [{"role": "user", "content": user_query}]
response = model_with_tools.invoke(messages)
print("
Suggested tool calls:")
print(response.tool_calls)
messages.append(response)

tool_map = {t.name: t for t in tools}
for call in response.tool_calls:
    result = tool_map[call["name"]].invoke(call["args"])
    print(f"{call['name']} result:", result)
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final answer:")
print(final_response.content)
