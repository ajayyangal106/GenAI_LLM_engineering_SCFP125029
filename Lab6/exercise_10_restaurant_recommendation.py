from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

load_dotenv()

@tool
def find_restaurants(cuisine: str, city: str) -> str:
    """Find restaurants matching a cuisine and city."""
    data = {
        ("italian", "pune"): ["Pizza Corner", "Olive Table"],
        ("indian", "pune"): ["Spice Garden", "Royal Thali"],
        ("italian", "mumbai"): ["Pasta House", "Pizza Corner"],
        ("indian", "mumbai"): ["Coastal Kitchen", "Spice Garden"]
    }
    return ", ".join(data.get((cuisine.lower(), city.lower()), ["No matching restaurants found"]))

@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Return a restaurant rating out of five."""
    ratings = {"Pizza Corner": 4.4, "Olive Table": 4.6, "Spice Garden": 4.3, "Royal Thali": 4.2, "Pasta House": 4.5, "Coastal Kitchen": 4.4}
    return f"{restaurant_name}: {ratings.get(restaurant_name, 4.0)}/5"

@tool
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    """Check whether a restaurant is open on a day."""
    return f"{restaurant_name} is open on {day}."

@tool
def estimate_cost_for_two(restaurant_name: str) -> str:
    """Estimate the cost for two people."""
    prices = {"Pizza Corner": 900, "Olive Table": 1400, "Spice Garden": 800, "Royal Thali": 700, "Pasta House": 1200, "Coastal Kitchen": 1300}
    return f"Estimated cost for two at {restaurant_name}: ₹{prices.get(restaurant_name, 1000)}"

tools = [find_restaurants, get_restaurant_rating, check_restaurant_open, estimate_cost_for_two]
model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(tools)

query = input("Ask for a restaurant recommendation: ")
messages = [{"role": "user", "content": query}]
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
