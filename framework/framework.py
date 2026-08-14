from llm.groq_client import groq_client
from router.router import RequestRouter, Route
from rag.retrieval_pipeline import RAG
from crew.crew_manager import CrewManager

class AIFramework:
    """
    Main orchestrator of the AI system.

    Every user request enters here first.
    """

    def __init__(self):
        self.llm = groq_client
        self.router = RequestRouter()
        self.rag = RAG()
        self.crew = CrewManager()
        self.google_email = None

    def process_request(self, question: str) -> str:

        route = self.router.determine_route(question)
        print(f"\n[DEBUG] Selected Route: {route}\n")

        if route == Route.LLM:
            print("[DEBUG] Entering LLM branch")
            return self.llm.ask(question)

        elif route == Route.RAG:
            print("[DEBUG] Entering RAG branch")
            return self.rag.answer(
                question=question,
                chat_history=""
            )
        elif route == Route.CREW:
            print("[DEBUG] Entering CREW branch")

            workspace_route = self.router.determine_workspace_route(question)

            print(f"[DEBUG] Selected Workspace Route: {workspace_route}")

            return self.crew.run(
                route=workspace_route.value,
                question=question,
                google_email=self.google_email,
            )
        
        raise RuntimeError("Unknown route.")
    
    #Research the latest advancements in quantum computing and prepare a detailed report.