from llm.groq_client import groq_client
from router.router import RequestRouter, Route
from rag.retrieval_pipeline import RAG

class AIFramework:
    """
    Main orchestrator of the AI system.

    Every user request enters here first.
    """

    def __init__(self):
        self.llm = groq_client
        self.router = RequestRouter()
        self.rag = RAG()

    def process_request(self, question: str) -> str:

        route = self.router.determine_route(question)

        if route == Route.LLM:
            return self.llm.ask(question)

        elif route == Route.RAG:
            return self.rag.answer(
                question=question,
                chat_history=""
            )
        elif route == Route.CREW:
            raise NotImplementedError("CrewAI workflow not implemented yet.")

        raise RuntimeError("Unknown route.")