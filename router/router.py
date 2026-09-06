from enum import Enum

from llm.groq_client import groq_client

from orchestrator.execution_plan import (
    ExecutionStep,
    parse_execution_plan
)


class Route(Enum):

    LLM = "llm"
    RAG = "rag"
    CREW = "crew"


class WorkspaceRoute(Enum):

    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    DOCS = "docs"
    MAPS = "maps"


class RequestRouter:

    def __init__(self):

        self.llm = groq_client


    # =================================================
    # Determine Main Route
    # =================================================

    def determine_route(
        self,
        question: str
    ) -> Route:

        route = self.llm.classify_route(
            question
        )

        valid_routes = {

            "LLM": Route.LLM,

            "RAG": Route.RAG,

            "CREW": Route.CREW

        }

        return valid_routes.get(
            route,
            Route.LLM
        )


    # =================================================
    # Determine Workspace Route
    # =================================================

    def determine_workspace_route(
        self,
        question: str
    ) -> WorkspaceRoute:

        route = self.llm.classify_workspace_route(
            question
        )

        valid_routes = {

            "GMAIL": WorkspaceRoute.GMAIL,

            "CALENDAR": WorkspaceRoute.CALENDAR,

            "DRIVE": WorkspaceRoute.DRIVE,

            "DOCS": WorkspaceRoute.DOCS,

            "MAPS": WorkspaceRoute.MAPS,

        }

        return valid_routes.get(
            route,
            WorkspaceRoute.GMAIL
        )


    # =================================================
    # Determine Execution Plan
    # =================================================

    def determine_execution_plan(
        self,
        question: str
    ) -> list[ExecutionStep]:

        plan = self.llm.create_execution_plan(
            question
        )

        return parse_execution_plan(
            plan
        )