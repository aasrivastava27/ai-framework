from llm.groq_client import groq_client


class AIFramework:
    """
    Main orchestrator of the AI system.

    Every user request enters here first.
    """

    def __init__(self):
        self.llm = groq_client

    def process_request(self, question: str) -> str:
        """
        Process a user request.

        For now, every request is sent directly to Groq.
        Later this method will decide whether to use:
        - RAG
        - CrewAI
        - MCP
        - Memory
        """

        response = self.llm.ask(question)

        return response