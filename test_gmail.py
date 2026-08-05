import asyncio
from workspace_mcp.client import WorkspaceMCPClient


async def main():
    client = WorkspaceMCPClient()

    # Step 1: Search for emails
    message_ids = await client.search_gmail_messages(
        user_google_email="aashutoshsr27@gmail.com",
        query="newer_than:7d",
    )

    print("MESSAGE IDS:")
    print(message_ids)

    # Step 2: Read the newest one
    latest_message_id = message_ids[0]["message_id"]

    result = await client.get_gmail_messages_content_batch(
        user_google_email="aashutoshsr27@gmail.com",
        message_ids=[latest_message_id],
    )
    print(type(result))
    print(result)

    print("\nEMAIL CONTENT:")
    print("Subject:", result["subject"])
    print("From:", result["from"])
    print("Date:", result["date"])
    print()
    print(result["body"])


asyncio.run(main())