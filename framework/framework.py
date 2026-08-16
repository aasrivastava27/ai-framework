from llm.gemini_client import gemini_client
from orchestrator.orchestrator import Orchestrator


class AIFramework:
    """
    Main entry point of the AI system.

    Every user request enters here first and is
    passed to the central orchestrator.
    """

    def __init__(self):

        self.llm = gemini_client
        self.orchestrator = Orchestrator()

        self.google_email = None

    def process_request(self, question: str, google_email: str) -> str:

        self.google_email = google_email

        return self.orchestrator.execute(
            question=question,
            google_email=self.google_email,
        )