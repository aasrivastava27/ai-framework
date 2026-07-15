from crewai import Task


class TaskFactory:
    """
    Creates all CrewAI tasks used by the framework.
    """

    def create_research_task(self, question: str, agent):

        return Task(

            description=(
                f"""
                Answer the following user question.

                Question:
                {question}
                """
            ),

            expected_output=(
                "A clear, complete and accurate answer "
                "to the user's question."
            ),

            agent=agent
        )