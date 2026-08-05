from typing import ClassVar

from crewai.tools import BaseTool
import asyncio
from workspace_mcp.client import WorkspaceMCPClient


class SearchGmailTool(BaseTool):
    name: str = "Search Gmail"
    description: str = (
        "Search Gmail using Gmail search syntax. "
        "Examples: from:, subject:, has:attachment, newer_than:7d, older_than:30d."
    )

    def _run(
        self,
        user_google_email: str,
        query: str,
    ):
        client = WorkspaceMCPClient()

        return asyncio.run(
            client.search_gmail_messages(
                user_google_email=user_google_email,
                query=query,
            )
        )


class GetEmailContentTool(BaseTool):
    name: str = "Get Gmail Messages"

    description: str = (
        "Retrieve the contents of Gmail messages using a list of message IDs."
    )

    MAX_BODY_LENGTH: ClassVar[int] = 1500

    def _run(
        self,
        user_google_email: str,
        message_ids: list[str],
    ):
        client = WorkspaceMCPClient()

        result = asyncio.run(
            client.get_gmail_messages_content_batch(
                user_google_email=user_google_email,
                message_ids=message_ids,
            )
        )

        print(type(result))
        print(result)

        return result
    
class SendGmailTool(BaseTool):
    name: str = "Send Gmail Message"
    description: str = (
        "Send an email.\n"
        "Inputs:\n"
        "- user_google_email\n"
        "- to\n"
        "- subject\n"
        "- body\n\n"
        "Use the user's wording for the body. "
        "If no subject is provided, generate a short natural subject."
    )

    def _run(
        self,
        user_google_email: str,
        to: str,
        subject: str,
        body: str,
    ):
        client = WorkspaceMCPClient()

        result = asyncio.run(
            client.send_gmail_message(
                user_google_email=user_google_email,
                to=to,
                subject=subject,
                body=body,
            )
        )

        # MCP tools generally return the actual result in .data
        return result.data if hasattr(result, "data") else result
