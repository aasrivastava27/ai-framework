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
- Use Workspace tools when the request requires external data or an action.
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

Maps:

- Use Maps tools for location, route, distance, and nearby-place requests.
- Use Geocode Location when the user needs a location identified.
- Use Get Route when the user asks for directions or a route.
- Use Get Distance when the user asks how far one location is from another.
- Use Find Nearby Places when the user asks for nearby restaurants,
  cafes, hospitals, hotels, pharmacies, or other places.
- Use Get Place Details when detailed information about a specific
  OpenStreetMap place is required.

- For a chained Maps request, a previous pipeline result may
  be provided in the user request.

- When using a place returned by a previous Maps operation,
  prefer the exact latitude and longitude from that result.

- If latitude and longitude are available, pass them to the
  routing tool instead of routing using only the place name.

- Never substitute a different place with the same name.

- Never invent Maps information.
- Use the actual tool result as the source of truth.
- The current task may be one step in a larger multi-step workflow.

- If the current task references the result of a previous step,
  use that previous result as input.

- Do ONLY the current task. Do not perform tasks that belong
  to later steps.

- If a previous Maps result contains a place with exact
  latitude and longitude, use those coordinates for subsequent
  Maps operations.

- Never resolve an ambiguous place name again when exact
  coordinates are already available.

- Never substitute another place with the same name.
""",

            expected_output=(
                "Give a concise final answer based only on actual tool results. "
                "Confirm successful actions only when the tool confirms success."
            ),

            agent=agent,
        )