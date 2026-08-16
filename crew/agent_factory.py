from crewai import Agent
from crewai import LLM

from config.settings import settings

from crew.tools.gmail_tools import (
    SearchGmailTool,
    GetEmailContentTool,
    SendGmailTool,
)

from crew.tools.calendar_tools import (
    ListCalendarsTool,
    GetCalendarEventsTool,
    CreateCalendarEventTool,
    UpdateCalendarEventTool,
)

from crew.tools.time_tools import CurrentDateTimeTool


llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=settings.GEMINI_API_KEY,
    max_tokens=1000,
)


class AgentFactory:

    def create_calendar_agent(self):

        return Agent(
            role="Google Calendar Assistant",

            goal=(
                "Help users manage Google Calendar accurately. "
                "Use only the calendar tools necessary for the request."
            ),

            backstory=(
                "You are a Google Calendar assistant. "
                "You must carefully follow the user's requested date, time, "
                "event title, and changes. "
                "Never invent event IDs or event information. "
                "Never claim an action succeeded unless the tool confirms it."
            ),

            llm=llm,

            tools=[
                ListCalendarsTool(),
                GetCalendarEventsTool(),
                CreateCalendarEventTool(),
                UpdateCalendarEventTool(),
                CurrentDateTimeTool(),
            ],

            verbose=True,
        )


    def create_gmail_agent(self):

        return Agent(
            role="Gmail Assistant",

            goal=(
                "Help users interact with Gmail accurately and safely."
            ),

            backstory=(
                "You are a Gmail assistant with access to the user's Gmail "
                "through the provided Gmail tools. "

                "When the user asks about their emails, inbox, messages, "
                "senders, subjects, email contents, or sending an email, "
                "you MUST use the appropriate Gmail tool. "

                "Do NOT tell the user that you lack access to Gmail. "
                "The Gmail tools provide the required access. "

                "For reading an email, first use Search Gmail to find the "
                "relevant email, then use Get Email Content when the actual "
                "email contents are required. "

                "For sending an email, use Send Gmail Message. "

                "Never invent information or claim an action succeeded "
                "without tool confirmation."
            ),

            llm=llm,

            tools=[
                SearchGmailTool(),
                GetEmailContentTool(),
                SendGmailTool(),
            ],

            verbose=True,
        )