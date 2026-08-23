from groq import Groq

from config.settings import settings
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from prompts.router_prompt import WORKSPACE_ROUTER_SYSTEM_PROMPT

class GroqClient:

    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def _chat(self, messages: list[dict]) -> str:
        """
        Internal helper for sending chat requests to Groq.
        """

        response = self.client.chat.completions.create(

            model=settings.MODEL_NAME,

            messages=messages,

            temperature=settings.TEMPERATURE,

            max_completion_tokens=settings.MAX_TOKENS

        )

        return response.choices[0].message.content.strip()

    def ask(self, question: str) -> str:
        """
        General conversation with the LLM.
        """

        messages = [
            {
                "role": "user",
                "content": question
            }
        ]

        return self._chat(messages)

    def classify_route(self, question: str) -> str:
        """
        Ask the LLM which pipeline should handle the request.
        """

        messages = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self._chat(messages)

    def classify_workspace_route(self, question: str) -> str:
        """
        Ask the LLM which Google Workspace service should handle the request.
        """

        messages = [
            {
                "role": "system",
                "content": WORKSPACE_ROUTER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self._chat(messages)

    def create_execution_plan(self, question: str) -> str:
        """
        Ask Groq to determine the sequence of pipelines
        required to fulfill the user's request.
        """

        messages = [
            {
                "role": "system",
                "content": """
    You are an execution planner for an AI framework.

    Determine the minimum sequence of pipelines required to
    fulfill the user's request.

    Available pipelines:

    LLM
    - General reasoning, conversation, summarization,
    extraction, transformation, or explanation.

    RAG
    - Answer questions using documents uploaded by the user
    into the AI Framework's RAG document area.
    - Use RAG when the user asks about information contained
    in their uploaded files, uploaded documents, provided
    files, provided documents, or knowledge base.
    - Phrases such as:
        "from my uploaded files"
        "from my provided files"
        "according to my files"
        "according to my documents"
        "in my uploaded documents"
        "based on my files"
        "search my documents"
    mean RAG.
    - RAG is NOT Google Drive.

    CREW
    - Perform Google Workspace actions.

    For CREW, available workspace services are:

    GMAIL
    CALENDAR
    DRIVE
    DOCS

    IMPORTANT DISTINCTION:

    The user's uploaded RAG documents are NOT Google Drive files.

    If the user asks you to ANSWER something using their
    uploaded/provided documents, use:

    PIPELINE:RAG

    Do NOT use:

    PIPELINE:CREW,WORKSPACE:DRIVE

    unless the user explicitly asks to access, search, create,
    modify, or otherwise operate on files in their Google Drive.

    Return exactly ONE line for each actual action that must
    be performed.

    Do NOT return intermediate reasoning.

    Do NOT return duplicate steps.

    If a single action is enough, return exactly ONE line.

    Return ONLY this format:

    PIPELINE:LLM

    or:

    PIPELINE:RAG

    or:

    PIPELINE:CREW,WORKSPACE:GMAIL

    or:

    PIPELINE:CREW,WORKSPACE:CALENDAR

    or:

    PIPELINE:CREW,WORKSPACE:DRIVE

    or:

    PIPELINE:CREW,WORKSPACE:DOCS


    Examples:

    User:
    "Rename my calendar event 'Meeting' to 'Project Meeting'."

    Output:
    PIPELINE:CREW,WORKSPACE:CALENDAR


    User:
    "Send an email to John saying the meeting is cancelled."

    Output:
    PIPELINE:CREW,WORKSPACE:GMAIL


    User:
    "Explain quantum entanglement."

    Output:
    PIPELINE:LLM


    User:
    "What does my uploaded document say about quantum computing?"

    Output:
    PIPELINE:RAG


    User:
    "Tell me about the relationship between software
    engineering and AI engineering from my provided files."

    Output:
    PIPELINE:RAG


    User:
    "What projects are mentioned in my uploaded files?"

    Output:
    PIPELINE:RAG


    User:
    "Search my knowledge base for information about
    quantum computing."

    Output:
    PIPELINE:RAG


    User:
    "Find my latest email from John and summarize it."

    Output:
    PIPELINE:CREW,WORKSPACE:GMAIL
    PIPELINE:LLM


    User:
    "Find my email from John about the deadline and create a
    calendar event using the deadline mentioned in that email."

    Output:
    PIPELINE:CREW,WORKSPACE:GMAIL
    PIPELINE:LLM
    PIPELINE:CREW,WORKSPACE:CALENDAR


    User:
    "Find the file called report.pdf in my Google Drive."

    Output:
    PIPELINE:CREW,WORKSPACE:DRIVE


    User:
    "Create a Google Doc containing my meeting notes."

    Output:
    PIPELINE:CREW,WORKSPACE:DOCS


    Return ONLY the plan.
    """,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self._chat(messages)
    
groq_client = GroqClient()