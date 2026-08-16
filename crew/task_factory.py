from crewai import Task


class TaskFactory:
    """
    Creates CrewAI tasks for Google Workspace requests.
    """

    @staticmethod
    def create_workspace_task(
        agent,
        user_prompt: str,
        google_email: str,
    ):
        return Task(
            description=f"""
User email: {google_email}

User request:
{user_prompt}

Rules:
- Use Workspace tools when the request requires Gmail or Calendar data/action.
- Never invent information.
- Never claim an action succeeded unless the tool confirms it.
- Use the user's email whenever required.
- Use only the tools necessary for the request.

Gmail:
- If the user asks about emails or Gmail, you MUST use a Gmail tool.
- Do NOT answer that you lack access to Gmail.
- To find an email, use Search Gmail.
- To read the contents of an email, use Search Gmail first,
  then Get Email Content.
- To send an email, use Send Gmail Message.
- Never invent email information.
- Never claim an action succeeded unless the tool confirms it.

Calendar:

- Use Calendar tools for Calendar requests.
- Default calendar: primary.

- If the user provides an explicit date such as
  "August 14, 2026" or "Friday, August 14",
  DO NOT use Get Current Date and Time.

- Only use Get Current Date and Time when the user uses a
  relative date such as:
  "today", "tomorrow", "yesterday", "this week", or "next week".

- For an update with an event ID, update directly.

- If the user does not provide an event ID:
  1. Use Get Calendar Events to find the event.
  2. Use the event title and the user's specified date/time range.
  3. Obtain the event ID from the result.
  4. Use Update Calendar Event with that event ID.

- If the user asks only to rename an event:
  update ONLY the event title.

- If the user says not to change the date or time:
  DO NOT provide start_time or end_time to Update Calendar Event.

- Never invent an event ID.

- Never claim the update succeeded unless
  Update Calendar Event confirms success.
""",

            expected_output=(
                "Give a concise final answer based only on actual tool results. "
                "Confirm successful actions only when the tool confirms success."
            ),

            agent=agent,
        )