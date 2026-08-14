from crewai.tools import BaseTool
from datetime import datetime
from zoneinfo import ZoneInfo


class CurrentDateTimeTool(BaseTool):
    name: str = "Get Current Date and Time"

    description: str = (
        "Get the actual current date and time. "
        "Use this BEFORE performing Calendar operations involving "
        "relative dates such as today, tomorrow, this week, or next week. "
        "This tool returns an actual ISO 8601 datetime. "
        "Do NOT copy the tool name into another tool's arguments."
    )

    def _run(self):
        now = datetime.now(ZoneInfo("Asia/Kolkata"))

        return now.isoformat()