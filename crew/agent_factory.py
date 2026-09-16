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

from crew.tools.maps_tools import (
    GeocodeLocationTool,
    GetRouteTool,
    GetDistanceTool,
    FindNearbyPlacesTool,
    GetPlaceDetailsTool,
)
from crew.tools.time_tools import CurrentDateTimeTool
from crew.tools.csv_tools import CSVAnalyzerTool


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
    
    
    def create_maps_agent(self):

        return Agent(
            role="Maps Assistant",

            goal=(
                "Help users with location searches, routes, "
                "distances, and nearby places accurately."
            ),

            backstory=(
                "You are a maps and navigation assistant. "
                "You have access to OpenStreetMap through the "
                "provided Maps tools. "

                "Use the Maps tools whenever the user asks about "
                "locations, routes, distances, or nearby places. "

                "Never invent locations, distances, routes, "
                "places, or travel information. "

                "Use the tool results as the source of truth. "
                "If the user specifies a travel mode, use it. "
                "Otherwise, use car routing by default."
            ),

            llm=llm,

            tools=[
                GeocodeLocationTool(),
                GetRouteTool(),
                GetDistanceTool(),
                FindNearbyPlacesTool(),
                GetPlaceDetailsTool(),
            ],

            verbose=True,
        )
    
    def create_data_agent(self):

        return Agent(
            role="CSV Data Analysis Assistant",

            goal=(
                "Analyze CSV files accurately using the "
                "native CSV Analyzer tool."
            ),

            backstory=(
                "You are a data analysis assistant with access "
                "to a native Python CSV analysis tool powered "
                "by pandas. "

                "Whenever the user asks to analyze a CSV file, "
                "you MUST use the CSV Analyzer tool. "

                "Never invent data, calculations, rows, "
                "statistics, or file contents. "

                "Use the authenticated user's email when "
                "calling the CSV Analyzer tool. "

                "Use the actual tool result as the source "
                "of truth. "

                "If a previous pipeline result is provided, "
                "use it as input for the current task. "

                "Perform ONLY the current task."
            ),

            llm=llm,

            tools=[
                CSVAnalyzerTool(),
            ],

            verbose=True,
        )

