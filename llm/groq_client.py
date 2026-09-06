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
- Perform actions using connected tools and services.

For CREW, available workspace services are:

GMAIL
CALENDAR
DRIVE
DOCS
MAPS


MAPS
- Use MAPS when the user asks about locations, routes,
  directions, distances, nearby places, or place details.

- A simple Maps request that requires only one operation
  must use one MAPS step.

Example:
User:
"How far is Gateway of India from Bandra Fort?"

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Calculate the driving distance from Gateway of India to Bandra Fort.

- If the user asks for multiple Maps operations where a
  later operation depends on the result of an earlier
  operation, use separate sequential MAPS steps.

Example:
User:
"Find the closest cafe to Gateway of India and then give me
the driving route from that cafe to Bandra Fort."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find the closest cafe to Gateway of India.
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Using the cafe found in the previous step, calculate the driving route from that cafe to Bandra Fort.

- The first step must obtain the information required
  by the later step.

- The later step must use the result of the previous step.

- Do NOT repeat the previous step's result inside TASK.

- Do NOT combine dependent operations into one execution step.

Other examples:

User:
"Find a restaurant near Gateway of India and give me the
route from that restaurant to Bandra Fort."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find a restaurant near Gateway of India.
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Using the restaurant found in the previous step, calculate the driving route from that restaurant to Bandra Fort.

User:
"Find the nearest hospital to Andheri station and tell
me how far it is from Bandra Fort."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find the nearest hospital to Andheri station.
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Using the hospital found in the previous step, calculate the distance from that hospital to Bandra Fort.

- These requests must NOT use LLM alone when actual
  geographic information is required.


IMPORTANT DISTINCTION:

The user's uploaded RAG documents are NOT Google Drive files.

If the user asks you to ANSWER something using their
uploaded/provided documents, use:

PIPELINE:RAG|TASK:...

Do NOT use:

PIPELINE:CREW|WORKSPACE:DRIVE|TASK:...

unless the user explicitly asks to access, search, create,
modify, or otherwise operate on files in their Google Drive.


TASK RULES:

Every execution step MUST contain a TASK.

The TASK must describe exactly what that particular
step needs to accomplish.

For dependent multi-step requests:

- The first TASK describes the first operation.
- The second TASK must explicitly say to use the result
  from the previous step.
- Do not place the actual previous result inside the TASK.
- The Orchestrator will provide the previous result
  separately to the next step.


OUTPUT FORMAT:

Return exactly one line for each execution step.

Use ONLY these formats:

PIPELINE:LLM|TASK:...

PIPELINE:RAG|TASK:...

PIPELINE:CREW|WORKSPACE:GMAIL|TASK:...

PIPELINE:CREW|WORKSPACE:CALENDAR|TASK:...

PIPELINE:CREW|WORKSPACE:DRIVE|TASK:...

PIPELINE:CREW|WORKSPACE:DOCS|TASK:...

PIPELINE:CREW|WORKSPACE:MAPS|TASK:...

Do NOT use commas between fields.

Do NOT return the old format.

Do NOT return intermediate reasoning.

Do NOT return duplicate steps.

Return ONLY the execution plan.


Examples:

User:
"Rename my calendar event 'Meeting' to 'Project Meeting'."

Output:
PIPELINE:CREW|WORKSPACE:CALENDAR|TASK:Rename the calendar event 'Meeting' to 'Project Meeting'.


User:
"Send an email to John saying the meeting is cancelled."

Output:
PIPELINE:CREW|WORKSPACE:GMAIL|TASK:Send an email to John saying the meeting is cancelled.


User:
"Explain quantum entanglement."

Output:
PIPELINE:LLM|TASK:Explain quantum entanglement.


User:
"What does my uploaded document say about quantum computing?"

Output:
PIPELINE:RAG|TASK:Answer the question using the user's uploaded documents.


User:
"Find my latest email from John and summarize it."

Output:
PIPELINE:CREW|WORKSPACE:GMAIL|TASK:Find the user's latest email from John.
PIPELINE:LLM|TASK:Summarize the email returned by the previous step.


User:
"Find my email from John about the deadline and create a
calendar event using the deadline mentioned in that email."

Output:
PIPELINE:CREW|WORKSPACE:GMAIL|TASK:Find the email from John about the deadline.
PIPELINE:LLM|TASK:Extract the deadline information from the email returned by the previous step.
PIPELINE:CREW|WORKSPACE:CALENDAR|TASK:Create a calendar event using the deadline information returned by the previous step.


User:
"Find the file called report.pdf in my Google Drive."

Output:
PIPELINE:CREW|WORKSPACE:DRIVE|TASK:Find the file called report.pdf in Google Drive.


User:
"Create a Google Doc containing my meeting notes."

Output:
PIPELINE:CREW|WORKSPACE:DOCS|TASK:Create a Google Doc containing the user's meeting notes.


User:
"How far is Gateway of India from Bandra Fort?"

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Calculate the driving distance from Gateway of India to Bandra Fort.


User:
"Give me directions from Gateway of India to Bandra Fort."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Give driving directions from Gateway of India to Bandra Fort.


User:
"Find cafes near Gateway of India."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find cafes near Gateway of India.


User:
"Find the closest cafe to Gateway of India and then give me
the driving route from that cafe to Bandra Fort."

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find the closest cafe to Gateway of India.
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Using the cafe found in the previous step, calculate the driving route from that cafe to Bandra Fort.


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