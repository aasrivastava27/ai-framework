from crewai import Crew, Process

from crew.agents import AgentFactory
from crew.tasks import TaskFactory


class CrewManager:
    """
    Handles creation and execution of CrewAI workflows.
    """

    def __init__(self):

        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()

    def run(self, question: str) -> str:

        research_agent = self.agent_factory.create_research_agent()

        research_task = self.task_factory.create_research_task(
            question=question,
            agent=research_agent
        )

        crew = Crew(

            agents=[
                research_agent
            ],

            tasks=[
                research_task
            ],

            process=Process.sequential,

            verbose=True
        )

        result = crew.kickoff()

        return str(result)