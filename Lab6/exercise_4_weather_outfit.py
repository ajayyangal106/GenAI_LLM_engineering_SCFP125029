from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get sample current weather for a city."""
    weather = {
        "Mumbai": {"temperature": 32, "conditions": "hot and humid"},
        "Pune": {"temperature": 25, "conditions": "cloudy"},
        "Delhi": {"temperature": 30, "conditions": "sunny"},
        "Bengaluru": {"temperature": 22, "conditions": "rainy"}
    }
    data = weather.get(city.title())
    if not data:
        return f"Weather data unavailable for {city}."
    return f"Temperature: {data['temperature']}°C, Conditions: {data['conditions']}"

@tool
def recommend_outfit(temperature: float, conditions: str) -> str:
    """Recommend an outfit using temperature and weather conditions."""
    conditions = conditions.lower()
    if "rain" in conditions:
        return "Carry an umbrella and wear a light rain jacket."
    if temperature >= 30 or "humid" in conditions:
        return "Wear light cotton clothes and carry water."
    if temperature <= 15:
        return "Wear warm clothes and a jacket."
    return "Wear comfortable clothes and carry a light jacket."

tools = [get_weather, recommend_outfit]
model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(tools)

user_query = input("Ask about the weather and outfit: ")
messages = [{"role": "user", "content": user_query}]
first = model_with_tools.invoke(messages)
print("
First tool call:")
print(first.tool_calls)
messages.append(first)

for call in first.tool_calls:
    if call["name"] == "get_weather":
        result = get_weather.invoke(call["args"])
        print("Weather result:", result)
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

second = model_with_tools.invoke(messages)
print("
Second tool call:")
print(second.tool_calls)
messages.append(second)

for call in second.tool_calls:
    if call["name"] == "recommend_outfit":
        result = recommend_outfit.invoke(call["args"])
        print("Outfit result:", result)
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final recommendation:")
print(final_response.content)
