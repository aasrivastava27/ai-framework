from crewai import Agent, Task, Crew
from crewai import LLM
from config.settings import settings
from crew.tools.gmail_tools import SearchGmailTool

llm = LLM(
    model=f"groq/{settings.MODEL_NAME}",
    api_key=settings.GROQ_API_KEY,
)

agent = Agent(
    role="Email Assistant",
    goal="Help users search Gmail",
    backstory="You are an expert Gmail assistant.",
    llm=llm,
    tools=[SearchGmailTool()],
    verbose=True,
)

task = Task(
    description="""
Search my Gmail for emails received in the last 30 days.
My email address is aashutoshsr27@gmail.com.
""",
    expected_output="A list of matching emails.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

result = crew.kickoff()

print(result)