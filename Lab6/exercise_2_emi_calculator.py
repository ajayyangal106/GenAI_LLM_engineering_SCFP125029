from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
import os

load_dotenv()

class EMIInput(BaseModel):
    principal: float = Field(gt=0, description="Loan principal")
    annual_rate: float = Field(ge=0, le=30, description="Annual interest rate")
    tenure_months: int = Field(gt=0, description="Loan tenure in months")

def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    rate = annual_rate / 12 / 100
    if rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * rate * (1 + rate) ** tenure_months
        emi = emi / ((1 + rate) ** tenure_months - 1)
    return f"Monthly EMI: ₹{emi:.2f}"

emi_tool = StructuredTool.from_function(
    func=calculate_emi,
    name="calculate_emi",
    description="Calculate the monthly EMI for a loan.",
    args_schema=EMIInput
)

model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools([emi_tool])

user_query = input("Ask about the loan EMI: ")
messages = [{"role": "user", "content": user_query}]
response = model_with_tools.invoke(messages)

print("
Suggested tool calls:")
print(response.tool_calls)

for call in response.tool_calls:
    result = emi_tool.invoke(call["args"])
    print("
Tool result:", result)
    messages.append(response)
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final answer:")
print(final_response.content)
