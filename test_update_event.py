from crew.tools.calendar_tools import UpdateCalendarEventTool

tool = UpdateCalendarEventTool()

result = tool.run(
    user_google_email="aashutoshsr27@gmail.com",
    event_id="6tgj4cr3c8pmcbb468pm6b9kckpjib9pcgp6cb9lc9imccj669gj0dr6cc_20260822T123000Z",
    summary="Updated Zoom Meeting"
)

print("RESULT:")
print(result)