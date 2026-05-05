from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler

from agents.tools.rag_tool import rag_tool
from agents.tools.session_history_tool import session_history_tool
from agents.tools.goal_entries_tool import goal_entries_tool


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
        tools = [session_history_tool, goal_entries_tool, rag_tool]

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
        self._executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def run(self, user_input: str) -> tuple[str, list[str]]:
        tracker = _SourceTracker()
        result = self._executor.invoke({"input": user_input}, config={"callbacks": [tracker]})
        return result["output"], tracker.sources
