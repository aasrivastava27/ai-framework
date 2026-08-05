from crewai import Agent
from crewai import LLM

from config.settings import settings

from crew.tools.gmail_tools import ( 
    SearchGmailTool, 
    GetEmailContentTool,
    SendGmailTool,
)


llm = LLM(
    model=f"groq/{settings.MODEL_NAME}",
    api_key=settings.GROQ_API_KEY,
)


class AgentFactory:

    def create_email_agent(self, google_email):
        return Agent (

            role="Email Assistant",

            goal=(
                "Help users search, read, and send Gmail messages accurately."
            ),

            backstory=(
                "You are an expert Gmail assistant. "
                "Use Gmail tools whenever needed. "
                "Never invent email data. "
                "Never claim an action succeeded unless the tool confirms it."
            ),

            llm=llm,
            tools = [
                SearchGmailTool(),
                GetEmailContentTool(),
                SendGmailTool()
            ],

            verbose=True,
        )
    