import asyncio
from typing import Type
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
import os

load_dotenv()

class StockInput(BaseModel):
    product_id: str = Field(description="Product ID")

class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = "Check whether a product is in stock using its product ID."
    args_schema: Type[BaseModel] = StockInput

    def _run(self, product_id: str) -> str:
        stock = {"P1023": 10, "P1024": 25, "P1025": 0, "P1026": 7}
        quantity = stock.get(product_id.upper())
        if quantity is None:
            return f"{product_id} was not found."
        return f"{product_id}: {quantity} units available." if quantity else f"{product_id} is out of stock."

    async def _arun(self, product_id: str) -> str:
        await asyncio.sleep(0.2)
        return self._run(product_id)

async def main():
    stock_tool = ProductStockTool()

    print("Sync result:", stock_tool.invoke({"product_id": "P1023"}))
    print("Async result:", await stock_tool.ainvoke({"product_id": "P1023"}))

    model = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    model_with_tools = model.bind_tools([stock_tool])
    user_query = input("Ask about product stock: ")
    messages = [{"role": "user", "content": user_query}]
    response = model_with_tools.invoke(messages)
    print("
Suggested tool calls:")
    print(response.tool_calls)
    messages.append(response)

    for call in response.tool_calls:
        result = stock_tool.invoke(call["args"])
        print("Tool result:", result)
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final_response = model.invoke(messages)
    print("
Final answer:")
    print(final_response.content)

if __name__ == "__main__":
    asyncio.run(main())
