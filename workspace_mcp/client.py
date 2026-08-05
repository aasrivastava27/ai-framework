import re
from fastmcp import Client


class WorkspaceMCPClient:
    """
    Client for communicating with the Google Workspace MCP server.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8000/mcp"):
        self.server_url = server_url
        self.client = Client(server_url)

    async def search_gmail_messages(
        self,
        user_google_email: str,
        query: str,
    ):
        async with self.client:
            result = await self.client.call_tool(
                "search_gmail_messages",
                {
                    "user_google_email": user_google_email,
                    "query": query,
                },
            )

        # Extract the plain text returned by MCP
        text = result.data if hasattr(result, "data") else result

        emails = []

        # Find every Message ID and Thread ID pair
        pattern = (
            r"Message ID:\s*([^\n]+).*?"
            r"Thread ID:\s*([^\n]+)"
        )

        matches = re.findall(pattern, text, re.DOTALL)

        for message_id, thread_id in matches:
            emails.append(
                {
                    "message_id": message_id.strip(),
                    "thread_id": thread_id.strip(),
                }
            )

        return emails

    async def get_gmail_messages_content_batch(
        self,
        user_google_email: str,
        message_ids: list[str],
    ):
        async with self.client:
            result = await self.client.call_tool(
                "get_gmail_messages_content_batch",
                {
                    "user_google_email": user_google_email,
                    "message_ids": message_ids,
                },
            )
        print(type(result))
        print(dir(result))
        text = result.data

        subject = re.search(r"Subject:\s*(.*)", text)
        sender = re.search(r"From:\s*(.*)", text)
        date = re.search(r"Date:\s*(.*)", text)

        body = ""

        if "--- BODY ---" in text:
            body = text.split("--- BODY ---", 1)[1].strip()

        return {
            "subject": subject.group(1) if subject else "",
            "from": sender.group(1) if sender else "",
            "date": date.group(1) if date else "",
            "body": body[:1500],
        }

    async def send_gmail_message(
        self,
        user_google_email: str,
        to: str,
        subject: str,
        body: str,
    ):
        async with self.client:
            return await self.client.call_tool(
                "send_gmail_message",
                {
                    "user_google_email": user_google_email,
                    "to": to,
                    "subject": subject,
                    "body": body,
                },
            )