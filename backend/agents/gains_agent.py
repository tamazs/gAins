import asyncio
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


_MCP_SERVER = Path(__file__).parent.parent / "mcp_server.py"


class _SourceTracker(BaseCallbackHandler):
    def __init__(self):
        self.sources: list[str] = []

    def on_tool_end(self, output, **kwargs):
        for line in str(output).splitlines():
            if line.startswith("[Source:"):
                source = line.split("|")[0].replace("[Source:", "").strip().rstrip("]")
                if source not in self.sources:
                    self.sources.append(source)


class GainsAgent:
    def __init__(self, model: str = "gAinsModel"):
        self._model = model

    def run(self, user_input: str) -> tuple[str, list[str]]:
        return asyncio.run(self._run_async(user_input))

    async def _run_async(self, user_input: str) -> tuple[str, list[str]]:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(_MCP_SERVER)],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                llm = ChatOllama(model=self._model)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are a strength training advisor. The prompt you receive will tell you exactly what task to perform. "
                        "Follow the instructions in the prompt precisely — it will specify which tools to call and what to return. "
                        "Never give generic advice. Always ground your response in the data you retrieve from tools. "
                        "Respond only with the JSON structure specified in the prompt, no markdown, no extra text."
                    )),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ])

                agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
                executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

                tracker = _SourceTracker()
                result = await executor.ainvoke(
                    {"input": user_input},
                    config={"callbacks": [tracker]},
                )
                return result["output"], tracker.sources
