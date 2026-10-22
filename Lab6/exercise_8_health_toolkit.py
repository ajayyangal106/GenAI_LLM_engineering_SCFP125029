from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
import os

load_dotenv()

class BMIInput(BaseModel):
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)

class CalorieInput(BaseModel):
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    age: int = Field(gt=0)
    activity_level: str = Field(description="sedentary, light, moderate, or active")

def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return f"BMI: {bmi:.2f}"

def calculate_daily_calories(weight_kg: float, height_cm: float, age: int, activity_level: str) -> str:
    # This is a simple classroom estimate based on weight and activity level.
    factors = {"sedentary": 24, "light": 27, "moderate": 30, "active": 33}
    factor = factors.get(activity_level.lower(), 30)
    return f"Estimated daily calories: {weight_kg * factor:.0f} kcal"

bmi_tool = StructuredTool.from_function(calculate_bmi, name="calculate_bmi", description="Calculate BMI from weight and height.", args_schema=BMIInput)
calorie_tool = StructuredTool.from_function(calculate_daily_calories, name="calculate_daily_calories", description="Estimate daily calorie needs.", args_schema=CalorieInput)

class HealthToolkit:
    def get_tools(self):
        return [bmi_tool, calorie_tool]

health_tools = HealthToolkit().get_tools()
print("Toolkit tools:", [tool.name for tool in health_tools])

model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools(health_tools)
user_query = input("Ask for BMI and calorie needs: ")
messages = [{"role": "user", "content": user_query}]
response = model_with_tools.invoke(messages)
print("
Suggested tool calls:")
print(response.tool_calls)
messages.append(response)

tool_map = {t.name: t for t in health_tools}
for call in response.tool_calls:
    result = tool_map[call["name"]].invoke(call["args"])
    print(f"{call['name']} result:", result)
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final answer:")
print(final_response.content)
