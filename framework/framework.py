from llm.groq_client import groq_client
from router.router import RequestRouter, Route


class AIFramework:
    """
    Main orchestrator of the AI system.

    Every user request enters here first.
    """

    def __init__(self):
        self.llm = groq_client
        self.router = RequestRouter()

    def process_request(self, question: str) -> str:

        route = self.router.determine_route(question)

        if route == Route.LLM:
            return self.llm.ask(question)
        
        elif route == Route.RAG:
            pass

        elif route == Route.CREW:
            pass

        raise NotImplementedError(f"Route {route} is not implemented yet.")