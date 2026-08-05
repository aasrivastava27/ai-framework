from crewai import Crew, Process
from enum import Enum
from crew.agent_factory import AgentFactory
from crew.task_factory import TaskFactory

class Route(Enum):
    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    DOCS = "docs"

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

        if route == Route.GMAIL:
            return self._run_gmail(question, google_email)

        elif route == Route.CALENDAR:
            return self._run_calendar(question)

        else:
            raise ValueError(f"Unsupported crew route: {route}")

    # ---------------------------------------------------------
    # Gmail
    # ---------------------------------------------------------

    def _run_gmail(self, question: str, google_email):

        agent = self.agent_factory.create_email_agent(google_email)

        task = self.task_factory.create_email_search_task(
            agent=agent,
            user_prompt=question,
            google_email=google_email
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        return str(crew.kickoff())

    # ---------------------------------------------------------
    # Calendar
    # ---------------------------------------------------------

    def _run_calendar(self, question: str, google_email: str):

        agent = self.agent_factory.create_calendar_agent()

        task = self.task_factory.create_calendar_task(
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