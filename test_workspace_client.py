import asyncio

from workspace_mcp.client import WorkspaceMCPClient


async def main():
    client = WorkspaceMCPClient()

    result = await client.search_gmail_messages(
        user_google_email="aasrivastava@ucsd.edu",
        query="New York Times Games",
    )

    print("RESULT:")
    print(result)


asyncio.run(main())