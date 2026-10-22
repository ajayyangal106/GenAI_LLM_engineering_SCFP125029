from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
import os

load_dotenv()

class ExpenseInput(BaseModel):
    total_amount: float = Field(gt=0, description="Total bill amount")
    num_people: int = Field(gt=0, description="Number of people")

def split_bill(total_amount: float, num_people: int) -> str:
    share = total_amount / num_people
    return f"₹{total_amount:.2f} split between {num_people} people = ₹{share:.2f} each"

expense_tool = StructuredTool.from_function(
    func=split_bill,
    name="split_expense",
    description="Split a bill equally among people.",
    args_schema=ExpenseInput
)

manual = expense_tool.invoke({"total_amount": 2400, "num_people": 4})
print("Manual result:", manual)

model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
model_with_tools = model.bind_tools([expense_tool])
user_query = input("Ask to split a bill: ")
messages = [{"role": "user", "content": user_query}]
response = model_with_tools.invoke(messages)
print("
Suggested tool calls:")
print(response.tool_calls)
messages.append(response)

for call in response.tool_calls:
    result = expense_tool.invoke(call["args"])
    print("Bound-flow tool result:", result)
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final_response = model.invoke(messages)
print("
Final answer:")
print(final_response.content)
