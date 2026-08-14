import asyncio

from workspace_mcp.client import WorkspaceMCPClient


async def main():

    client = WorkspaceMCPClient()

    result = await client.manage_event(
        user_google_email="aashutoshsr27@gmail.com",
        action="create",
        summary="CrewAI Calendar Test",
        start_time="2026-08-16T15:00:00",
        end_time="2026-08-16T16:00:00",
        timezone="Asia/Kolkata",
        description="Test event created through WorkspaceMCPClient",
    )

    print("\nRESULT:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())