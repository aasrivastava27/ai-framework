from groq import Groq

from config.settings import settings
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from prompts.router_prompt import WORKSPACE_ROUTER_SYSTEM_PROMPT


class GroqClient:

    def __init__(self):

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    # -------------------------------------------------
    # Internal Chat Helper
    # -------------------------------------------------

    def _chat(self, messages: list[dict]) -> str:
        """
        Internal helper for sending chat requests to Groq.
        """

        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            temperature=settings.TEMPERATURE,
            max_completion_tokens=settings.MAX_TOKENS,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content.strip()

    # -------------------------------------------------
    # General LLM
    # -------------------------------------------------

    def ask(self, question: str) -> str:
        """
        General conversation with the Groq LLM.
        """

        messages = [
            {
                "role": "user",
                "content": question,
            }
        ]

        return self._chat(messages)

    # -------------------------------------------------
    # Pipeline Classification
    # -------------------------------------------------

    def classify_route(self, question: str) -> str:
        """
        Ask Groq which pipeline should handle the request.
        """

        messages = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self._chat(messages)

    # -------------------------------------------------
    # Workspace Classification
    # -------------------------------------------------

    def classify_workspace_route(
        self,
        question: str,
    ) -> str:
        """
        Ask Groq which Google Workspace or Maps
        service should handle the request.
        """

        messages = [
            {
                "role": "system",
                "content": WORKSPACE_ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self._chat(messages)

    # -------------------------------------------------
    # Execution Plan
    # -------------------------------------------------

    def create_execution_plan(
        self,
        question: str,
    ) -> str:
        """
        Ask Groq to determine the sequence of pipelines
        required to fulfill the user's request.

        The output is parsed by RequestRouter.
        """

        planner_prompt = """
You are an execution planner for an AI framework.

Determine the minimum sequence of pipelines required
to fulfill the user's request.

Available pipelines:

DATA:
- Use DATA when a step requires analyzing a CSV file
  using the native CSV Analyzer tool.

LLM
- General reasoning, conversation, summarization,
  extraction, transformation, or explanation.

RAG
- Answer questions using documents uploaded by the user
  into the AI Framework's RAG document area.
- Use RAG when the user asks about information contained
  in their uploaded files, uploaded documents, provided
  files, provided documents, or knowledge base.

CORE PLANNING RULE:

The execution plan must contain a separate pipeline for EVERY
distinct operation required by the user's request.

Do NOT merge different operations into one pipeline.

There are two fundamentally different types of operations:

1. REASONING / CONTENT OPERATIONS
   These use LLM.
   Examples include:
   - analyze or interpret an already-produced result
   - summarize
   - extract information
   - calculate/compare based on previous pipeline output
   - write or draft content

2. EXTERNAL ACTIONS
   These MUST use CREW and the appropriate workspace.
   Examples include:
   - actually send an email → GMAIL
   - actually create/update/delete a calendar event → CALENDAR
   - actually perform a maps operation → MAPS
   - actually analyze a CSV → DATA

IMPORTANT:

If a request contains both reasoning/content generation AND
an external action, create separate sequential pipeline steps.

The LLM pipeline must NEVER be used to perform or represent
an external action.

If the user asks to SEND, CREATE, UPDATE, DELETE, or otherwise
PERFORM an action using an external service, the corresponding
CREW workspace pipeline MUST appear in the execution plan.

The final execution step must still be included even when
its input is the output of an earlier LLM or DATA step.

Examples:
- "from my uploaded files"
- "from my provided files"
- "according to my files"
- "according to my documents"
- "in my uploaded documents"
- "based on my files"
- "search my documents"

RAG is NOT Google Drive.

CREW
- Perform actions using connected tools and services.

Available CREW workspace services:

GMAIL
CALENDAR
DRIVE
DOCS
MAPS


MAPS RULES
-----------

Use MAPS when the user asks about:

- locations
- routes
- directions
- distances
- nearby places
- place details

A simple Maps request requiring one operation
must use one MAPS step.

Example:

User:
"How far is Gateway of India from Bandra Fort?"

Output:
PIPELINE:CREW|WORKSPACE:MAPS|TASK:Calculate the driving distance from Gateway of India to Bandra Fort.


DEPENDENT MAPS OPERATIONS
-------------------------

If the user asks for multiple Maps operations where
a later operation depends on the result of an earlier
operation, use separate sequential MAPS steps.

Example:

User:
"Find the closest cafe to Gateway of India and then
give me the driving route from that cafe to Bandra Fort."

Output:

PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find the closest cafe to Gateway of India.

PIPELINE:CREW|WORKSPACE:MAPS|TASK:Using the place found in the previous step, calculate the driving route from that place to Bandra Fort.

The first step MUST obtain the information required
by the later step.

The later step MUST use the result from the previous step.

Do NOT repeat the previous result inside TASK.

Do NOT combine dependent operations into one step.

If the previous Maps result contains exact latitude
and longitude, the next Maps operation must use those
exact coordinates.


RAG VS GOOGLE DRIVE
-------------------

The user's uploaded RAG documents are NOT Google Drive files.

If the user asks to answer something using their
uploaded/provided documents, use:

PIPELINE:RAG|TASK:...

Do NOT use:

PIPELINE:CREW|WORKSPACE:DRIVE|TASK:...

unless the user explicitly asks to access, search,
create, modify, or otherwise operate on files in
Google Drive.


TASK RULES
----------

Every execution step MUST contain a TASK.

The TASK must describe exactly what that particular
step needs to accomplish.

For dependent multi-step requests:

- The first TASK describes the first operation.
- The next TASK explicitly says to use the result
  from the previous step.
- Do NOT place the actual previous result inside TASK.
- The Orchestrator supplies the previous result separately.
TASK DECOMPOSITION:

When generating TASK text, describe only the operation
belonging to that pipeline.

Do not put a later operation inside an earlier pipeline's TASK.

For example, if the user requests:
- data analysis
- reasoning about the result
- an external action

then produce three separate steps rather than putting
the external action into the LLM TASK.

MULTI-STEP PIPELINE RULES
--------------------------

- Break the user's request into the minimum number of
  required pipeline steps.
- Each pipeline step must represent an actual operation
  required to fulfill the request.
- Preserve the dependency and order between operations.
- Use the output of an earlier pipeline as input to later
  pipelines when necessary.
- Do not use LLM merely to describe, draft, or simulate
  an action when the user actually requested that action
  to be performed.
- If the user requests an external action, route that
  action to the appropriate CREW workspace.
- LLM is for reasoning, transformation, summarization,
  extraction, or generating content.
- CREW is for performing external actions using tools.

Examples of intent:

"draft/write/compose an email"
→ LLM

"send an email"
→ CREW|WORKSPACE:GMAIL

"analyze a CSV"
→ CREW|WORKSPACE:DATA

"create a calendar event"
→ CREW|WORKSPACE:CALENDAR

"find nearby cafes"
→ CREW|WORKSPACE:MAPS

The planner must determine these pipelines from the
user's requested operations rather than matching specific
phrases or predefined examples.

OUTPUT FORMAT
-------------

Return exactly one line for each execution step.

Use ONLY these formats:

PIPELINE:LLM|TASK:...

PIPELINE:RAG|TASK:...

PIPELINE:CREW|WORKSPACE:GMAIL|TASK:...

PIPELINE:CREW|WORKSPACE:CALENDAR|TASK:...

PIPELINE:CREW|WORKSPACE:DRIVE|TASK:...

PIPELINE:CREW|WORKSPACE:DOCS|TASK:...

PIPELINE:CREW|WORKSPACE:MAPS|TASK:...

PIPELINE:CREW|WORKSPACE:DATA|TASK:...

Do NOT use commas between fields.

Do NOT return the old format.

Do NOT return reasoning.

Do NOT return duplicate steps.

Return ONLY the execution plan.


EXAMPLES
--------

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

User:
"Analyze employees.csv and find the average salary by department."

Output:
PIPELINE:CREW|WORKSPACE:DATA|TASK:Analyze employees.csv and calculate the average salary for each department.

User:
"Analyze employees.csv, find the department with the highest average salary, and tell me how many employees are in that department."

Output:
PIPELINE:CREW|WORKSPACE:DATA|TASK:Analyze employees.csv and calculate the average salary and employee count for each department.
PIPELINE:LLM|TASK:Using the result from the previous step, identify the department with the highest average salary and report its employee count.

Return ONLY the plan.
"""

        messages = [
            {
                "role": "system",
                "content": planner_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self._chat(messages)


# -------------------------------------------------
# Global Groq Client
# -------------------------------------------------

groq_client = GroqClient()