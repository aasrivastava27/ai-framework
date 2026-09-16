from crewai import Crew, Process
from enum import Enum

from crew.agent_factory import AgentFactory
from crew.task_factory import TaskFactory


class Route(Enum):
    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    DOCS = "docs"
    MAPS = "maps"
    DATA = "data"



class CrewManager:
    """
    Creates and executes the appropriate CrewAI workflow
    depending on the requested Google Workspace service.
    """

    def __init__(self):
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()

    def run(self, route: str, question: str, google_email: str):

        route = Route(route.lower())

        if route == Route.CALENDAR:
            return self._run_calendar(question, google_email)

        elif route == Route.GMAIL:
            return self._run_gmail(question, google_email)

        elif route == Route.MAPS:
            return self._run_maps(question, google_email)
        elif route == Route.DATA:
            return self._run_data(question, google_email)
        else:
            raise ValueError(f"Unsupported crew route: {route}")

    # ---------------------------------------------------------
    # Calendar
    # ---------------------------------------------------------

    def _run_calendar(self, question: str, google_email: str):

        agent = self.agent_factory.create_calendar_agent()

        task = self.task_factory.create_workspace_task(
            agent=agent,
            user_prompt=question,
            google_email=google_email,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return str(crew.kickoff())

    # ---------------------------------------------------------
    # Gmail
    # ---------------------------------------------------------

    def _run_gmail(self, question: str, google_email: str):

        agent = self.agent_factory.create_gmail_agent()

        task = self.task_factory.create_workspace_task(
            agent=agent,
            user_prompt=question,
            google_email=google_email,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return str(crew.kickoff())
    
    # ---------------------------------------------------------
    # Maps
    # ---------------------------------------------------------

    def _run_maps(self, question: str, google_email: str):

        agent = self.agent_factory.create_maps_agent()

        task = self.task_factory.create_workspace_task(
            agent=agent,
            user_prompt=question,
            google_email=google_email,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return str(crew.kickoff())
    
    # ---------------------------------------------------------
    # Data / CSV
    # ---------------------------------------------------------

    def _run_data(self, question: str, google_email: str):

        agent = self.agent_factory.create_data_agent()

        task = self.task_factory.create_data_task(
            agent=agent,
            user_prompt=question,
            google_email=google_email,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return str(crew.kickoff())


