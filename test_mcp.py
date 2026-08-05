import asyncio

from workspace_mcp.client import WorkspaceMCPClient


async def main():
    client = WorkspaceMCPClient()

    result = await client.search_gmail_messages(
        user_google_email="aashutoshsr27@gmail.com",
        query="newer_than:30d",
    )

    print(result)


asyncio.run(main())