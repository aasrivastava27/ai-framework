from enum import Enum


class Route(Enum):

    LLM = "llm"

    RAG = "rag"

    CREW = "crew"

class RequestRouter:

    def determine_route(self, question: str) -> Route:
        return Route.LLM