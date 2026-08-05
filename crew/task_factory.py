from crewai import Task


class TaskFactory:
    """
    Creates CrewAI tasks based on the user's intent.
    """

    @staticmethod
    def create_email_search_task(agent, user_prompt: str, google_email: str):
        """
        Task for handling Gmail-related requests.
        The agent is free to use any Gmail tools it has available.
        """

        return Task(
            description=f"""
                The user's Google email is:

                {google_email}

                User request:

                {user_prompt}

                You have three Gmail tools:

                1. Search Gmail
                2. Get Gmail Messages
                3. Send Gmail Message

                Rules:

                - For reading emails:
                    Search -> Read -> Answer

                - For sending emails:
                    Use Send Gmail Message.

                - Never invent email contents.

                - Never claim an email was sent unless the tool succeeds.

                - If the user didn't provide a subject,
                generate a short natural one.

                Always use the appropriate tool.
                """,

                expected_output="""
                    A complete response answering the user's Gmail request.
                    Include email details only if necessary.
                    """,
                
                agent=agent,

            )