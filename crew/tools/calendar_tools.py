from crewai.tools import BaseTool
import asyncio
from workspace_mcp.client import WorkspaceMCPClient
from pydantic import BaseModel, Field
from typing import Optional


class ListCalendarsTool(BaseTool):
    name: str = "List Calendars"

    description: str = (
        "List all Google Calendars available to the user."
    )

    def _run(
        self,
        user_google_email: str,
    ):
        client = WorkspaceMCPClient()

        return asyncio.run(
            client.list_calendars(
                user_google_email=user_google_email,
            )
        )


# ---------------------------------------------------------
# Get Calendar Events
# ---------------------------------------------------------

class GetCalendarEventsToolSchema(BaseModel):
    user_google_email: str = Field(
        ...,
        description="User's Google email"
    )

    calendar_id: str = Field(
        default="primary",
        description="Calendar ID. Defaults to primary."
    )

    time_min: Optional[str] = Field(
        default=None,
        description="Start of the time range in ISO 8601 format."
    )

    time_max: Optional[str] = Field(
        default=None,
        description="End of the time range in ISO 8601 format."
    )

    max_results: int = Field(
        default=10,
        description="Maximum number of events to return."
    )

    query: Optional[str] = Field(
        default=None,
        description="Search text for the event title or details."
    )

    detailed: bool = Field(
        default=True,
        description="Whether to return detailed event information."
    )


class GetCalendarEventsTool(BaseTool):
    name: str = "Get Calendar Events"

    description: str = (
        "Retrieve events from the user's Google Calendar. "
        "Use this tool whenever the user asks about calendar events, "
        "meetings, schedules, or events on a particular date or date range. "
        "time_min and time_max must be ISO 8601 datetime strings or None. "
        "If the user asks for today and no valid datetime is available, "
        "leave time_min and time_max as None."
    )

    args_schema: type[BaseModel] = GetCalendarEventsToolSchema

    def _run(
        self,
        user_google_email: str,
        calendar_id: str = "primary",
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 10,
        query: str | None = None,
        detailed: bool = True,
    ):

        client = WorkspaceMCPClient()

        if time_min in (
            "None",
            "null",
            "",
            "Get Current Date and Time",
            "Current Date and Time",
            "today",
        ):
            time_min = None

        if time_max in (
            "None",
            "null",
            "",
            "Get Current Date and Time",
            "Current Date and Time",
            "today",
        ):
            time_max = None

        return asyncio.run(
            client.get_events(
                user_google_email=user_google_email,
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
                query=query,
                detailed=detailed,
            )
        )


# ---------------------------------------------------------
# Create Calendar Event
# ---------------------------------------------------------

class CreateCalendarEventTool(BaseTool):
    name: str = "Create Calendar Event"

    description: str = (
        "Create a new event in the user's Google Calendar. "
        "Use this tool whenever the user asks to create, schedule, "
        "or add a calendar event or meeting."
    )

    def _run(
        self,
        user_google_email: str,
        summary: str,
        start_time: str,
        end_time: str,
        calendar_id: str = "primary",
        description: str = None,
        location: str = None,
        attendees: list[str] = None,
        timezone: str = "Asia/Kolkata",
        add_google_meet: bool = False,
    ):
        client = WorkspaceMCPClient()

        return asyncio.run(
            client.manage_event(
                user_google_email=user_google_email,
                action="create",
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                calendar_id=calendar_id,
                description=description,
                location=location,
                attendees=attendees,
                timezone=timezone,
                add_google_meet=add_google_meet,
            )
        )


# ---------------------------------------------------------
# Update Calendar Event
# ---------------------------------------------------------

class UpdateCalendarEventToolSchema(BaseModel):
    user_google_email: str = Field(
        ...,
        description="User's Google email"
    )

    event_id: str = Field(
        ...,
        description="ID of the calendar event to update"
    )

    summary: Optional[str] = Field(
        default=None,
        description="New event title. Only provide this if changing the title."
    )

    start_time: Optional[str] = Field(
        default=None,
        description="New start time in ISO 8601 format. Only provide this if changing the time."
    )

    end_time: Optional[str] = Field(
        default=None,
        description="New end time in ISO 8601 format. Only provide this if changing the time."
    )

    calendar_id: str = Field(
        default="primary",
        description="Calendar ID. Defaults to primary."
    )

    description: Optional[str] = Field(
        default=None,
        description="New event description. Only provide this if changing it."
    )

    location: Optional[str] = Field(
        default=None,
        description="New event location. Only provide this if changing it."
    )

    timezone: str = Field(
        default="Asia/Kolkata",
        description="Timezone for the event."
    )

    attendees: Optional[list[str]] = Field(
        default=None,
        description="List of attendee email addresses. Omit this field if attendees are not being changed."
    )


class UpdateCalendarEventTool(BaseTool):
    name: str = "Update Calendar Event"

    description: str = (
        "Update an existing Google Calendar event. "
        "Use this when the user wants to rename, reschedule, "
        "or modify an existing event. "
        "Only provide fields that need to be changed. "
        "Do not provide empty strings for optional fields. "
        "For attendees, provide a list of email addresses or omit the field."
    )

    args_schema: type[BaseModel] = UpdateCalendarEventToolSchema

    def _run(
        self,
        user_google_email: str,
        event_id: str,
        summary: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        calendar_id: str = "primary",
        description: Optional[str] = None,
        location: Optional[str] = None,
        timezone: str = "Asia/Kolkata",
        attendees: Optional[list[str]] = None,
    ):
        client = WorkspaceMCPClient()

        return asyncio.run(
            client.manage_event(
                user_google_email=user_google_email,
                action="update",
                event_id=event_id,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                calendar_id=calendar_id,
                description=description,
                location=location,
                timezone=timezone,
                attendees=attendees,
            )
        )