import re
from fastmcp import Client


class WorkspaceMCPClient:
    """
    Client for communicating with the Google Workspace MCP server.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8000/mcp"):
        self.server_url = server_url
        self.client = Client(server_url)

    @staticmethod
    def _get_result_data(result):
        """Extract plain data from a FastMCP tool result."""
        return result.data if hasattr(result, "data") else result

    @staticmethod
    def _clean_optional_string(value):
        """Convert empty strings to None."""
        if value == "":
            return None
        return value

    @staticmethod
    def _clean_attendees(attendees):
        """
        Normalize attendees so the MCP schema receives either:
        - None
        - list[str]
        """
        if attendees is None or attendees == "":
            return None

        if isinstance(attendees, str):
            return [attendees]

        if isinstance(attendees, list):
            return attendees

        return None

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

        text = self._get_result_data(result)

        emails = []

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

        text = self._get_result_data(result)

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
            result = await self.client.call_tool(
                "send_gmail_message",
                {
                    "user_google_email": user_google_email,
                    "to": to,
                    "subject": subject,
                    "body": body,
                },
            )

        return self._get_result_data(result)

    async def list_calendars(
        self,
        user_google_email: str,
    ):
        async with self.client:
            result = await self.client.call_tool(
                "list_calendars",
                {
                    "user_google_email": user_google_email,
                },
            )

        text = self._get_result_data(result)

        calendars = []

        pattern = r'- "(.*?)"(?: \(Primary\))? \(ID: (.*?)\)'

        matches = re.findall(pattern, text)

        for summary, calendar_id in matches:
            calendars.append(
                {
                    "summary": summary,
                    "calendar_id": calendar_id,
                }
            )

        return calendars

    async def get_events(
        self,
        user_google_email: str,
        calendar_id: str = "primary",
        event_id: str | None = None,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 25,
        query: str | None = None,
        detailed: bool = False,
        include_attachments: bool = False,
    ):
        async with self.client:
            result = await self.client.call_tool(
                "get_events",
                {
                    "user_google_email": user_google_email,
                    "calendar_id": calendar_id,
                    "event_id": event_id,
                    "time_min": time_min,
                    "time_max": time_max,
                    "max_results": max_results,
                    "query": query,
                    "detailed": detailed,
                    "include_attachments": include_attachments,
                },
            )

        return self._get_result_data(result)

    async def manage_event(
        self,
        user_google_email: str,
        action: str,
        summary: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        event_id: str | None = None,
        calendar_id: str = "primary",
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        timezone: str | None = None,
        add_google_meet: bool = False,
        recurrence: list[str] | None = None,
        send_updates: str = "all",
    ):
        # Normalize optional values before sending them to MCP.
        summary = self._clean_optional_string(summary)
        start_time = self._clean_optional_string(start_time)
        end_time = self._clean_optional_string(end_time)
        event_id = self._clean_optional_string(event_id)
        description = self._clean_optional_string(description)
        location = self._clean_optional_string(location)
        timezone = self._clean_optional_string(timezone)

        attendees = self._clean_attendees(attendees)

        async with self.client:
            result = await self.client.call_tool(
                "manage_event",
                {
                    "user_google_email": user_google_email,
                    "action": action,
                    "summary": summary,
                    "start_time": start_time,
                    "end_time": end_time,
                    "event_id": event_id,
                    "calendar_id": calendar_id,
                    "description": description,
                    "location": location,
                    "attendees": attendees,
                    "timezone": timezone,
                    "add_google_meet": add_google_meet,
                    "recurrence": recurrence,
                    "send_updates": send_updates,
                },
            )

        return self._get_result_data(result)
