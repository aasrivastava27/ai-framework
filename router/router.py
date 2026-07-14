from enum import Enum

from llm.groq_client import groq_client


class Route(Enum):
    LLM = "llm"
    RAG = "rag"
    CREW = "crew"


class RequestRouter:

    def __init__(self):
        self.llm = groq_client

    def determine_route(self, question: str) -> Route:

        route = self.llm.classify_route(question)

        valid_routes = {
            "LLM": Route.LLM,
            "RAG": Route.RAG,
            "CREW": Route.CREW
        }

        return valid_routes.get(route, Route.LLM)