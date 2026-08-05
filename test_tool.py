from crew.tools.gmail_tools import SearchGmailTool

tool = SearchGmailTool()

result = tool.run(
    user_google_email="aashutoshsr27@gmail.com",
    query="newer_than:30d",
)

print(result)