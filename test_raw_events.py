import asyncio
from workspace_mcp.client import WorkspaceMCPClient

async def main():
    client = WorkspaceMCPClient()

    async with client.client:
        result = await client.client.call_tool(
            "get_events",
            {
                "user_google_email": "aashutoshsr27@gmail.com",
                "calendar_id": "primary",
                "time_min": "2026-08-17T00:00:00+05:30",
                "time_max": "2026-08-23T23:59:59+05:30",
                "max_results": 25,
                "query": None,
                "detailed": True,
                "include_attachments": False,
            },
        )

    print("TYPE:", type(result))
    print("RESULT:")
    print(result)
    print("DATA:")
    print(getattr(result, "data", "NO DATA ATTRIBUTE"))

asyncio.run(main())
