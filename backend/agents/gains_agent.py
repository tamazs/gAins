from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

from agents.tools.rag_tool import rag_tool
from agents.tools.session_history_tool import session_history_tool


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
        llm = ChatOllama(model=model)
        tools = [rag_tool, session_history_tool]

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "When you receive a workout session to analyse, always follow this order: "
                "1. Call session_history_tool to retrieve the user's recent sessions and identify trends in weight, volume, and RPE across them. "
                "2. Call rag_tool with a query informed by what you found in the history. "
                "3. Respond with advice that explicitly references the historical trend and is grounded in the retrieved evidence. "
                "Never give generic advice — always reflect the user's actual history in your reasoning and summary. "
                "Respond only with the JSON structure you were given, no markdown, no extra text."
            )),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        self._executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def run(self, user_input: str) -> tuple[str, list[str]]:
        tracker = _SourceTracker()
        result = self._executor.invoke({"input": user_input}, config={"callbacks": [tracker]})
        return result["output"], tracker.sources
