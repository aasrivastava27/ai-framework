from crewai import Agent, LLM

from config.settings import settings


class AgentFactory:
    """
    Creates all CrewAI agents used by the framework.
    """

    def __init__(self):

        self.llm = LLM(
            model=f"groq/{settings.MODEL_NAME}",
            api_key=settings.GROQ_API_KEY,
        )

    def create_research_agent(self):

        return Agent(

            role="Research Specialist",

            goal="Answer research questions accurately and clearly.",

            backstory=(
                "You are an expert researcher who explains "
                "technical and non-technical topics in a simple way."
            ),

            llm=self.llm,

            verbose=True
        )