from llm.gemini_client import gemini_client
from router.router import RequestRouter
from rag.retrieval_pipeline import RAG
from crew.crew_manager import CrewManager


class Orchestrator:
    """
    Central orchestrator of the AI framework.

    Decides which pipeline(s) are required to fulfill
    the user's request.
    """

    def __init__(self):

        self.llm = gemini_client
        self.router = RequestRouter()
        self.crew = CrewManager()

        # RAG is initialized per user.
        self.rag_instances = {}

    # -------------------------------------------------
    # Get User-Specific RAG
    # -------------------------------------------------

    def get_rag(self, google_email: str):

        if not google_email:
            raise ValueError(
                "Google email is required for RAG."
            )

        if google_email not in self.rag_instances:

            print(
                f"[DEBUG] Creating RAG instance for "
                f"{google_email}"
            )

            self.rag_instances[google_email] = RAG(
                user_google_email=google_email
            )

        return self.rag_instances[google_email]

    # -------------------------------------------------
    # Execute Request
    # -------------------------------------------------

    def execute(
        self,
        question: str,
        google_email: str,
    ) -> str:

        if not google_email:
            raise ValueError(
                "Google email is required."
            )

        steps = self.router.determine_execution_plan(
            question
        )

        print(
            f"\n[DEBUG] Execution Plan: {steps}\n"
        )

        current_result = question
        original_question = question

        for step in steps:

            if (
                step.pipeline == "llm"
                and current_result != original_question
            ):

                current_result = self.llm.ask(
                    f"""
You are the final response generator.

The user asked:
{original_question}

A previous pipeline/tool execution produced this result:
{current_result}

Your job is to answer the user's original request
using the result above.

Do not ask the user what they want to do with the result.

Do not treat the result as merely pasted content.

If the user asked for a summary, summarize it.

If the user asked for specific information,
extract that information.

Only use information contained in the pipeline result.
Do not invent information.
"""
                )

            else:

                current_result = self.execute_step(
                    step=step,
                    question=current_result,
                    original_question=original_question,
                    google_email=google_email,
                )

        if current_result is None:
            raise RuntimeError(
                "Execution produced no result."
            )

        return current_result

    # -------------------------------------------------
    # Execute Individual Pipeline
    # -------------------------------------------------

    def execute_step(
        self,
        step,
        question: str,
        original_question: str,
        google_email: str,
    ) -> str:

        # ---------------------------------------------
        # LLM
        # ---------------------------------------------

        if step.pipeline == "llm":

            print(
                "[DEBUG] Executing LLM step"
            )

            prompt = f"""
You are the final response generator.

The user asked:
{original_question}

A previous pipeline/tool execution produced this result:
{question}

Your job is to answer the user's original request
using the result above.

Do not ask the user what they want to do with the result.

Do not treat the result as merely pasted content.

If the user asked for a summary, summarize it.

If the user asked for specific information,
extract that information.

Only use information contained in the pipeline result.
Do not invent information.
"""

            return self.llm.ask(prompt)

        # ---------------------------------------------
        # RAG
        # ---------------------------------------------

        elif step.pipeline == "rag":

            print(
                f"[DEBUG] Executing RAG step "
                f"for {google_email}"
            )

            rag = self.get_rag(
                google_email
            )

            rag_question = f"""
        The user originally asked:

        {original_question}

        The previous pipeline retrieved this information:

        {question}

        Use the user's uploaded documents to answer the original request
        using the information retrieved from the previous pipeline.

        Compare the retrieved information with the uploaded documents
        where necessary.

        Do not invent information.
        """

            return rag.answer(
                question=rag_question,
                chat_history=""
            )

        # ---------------------------------------------
        # CREW
        # ---------------------------------------------

        elif step.pipeline == "crew":

            print(
                f"[DEBUG] Executing CREW step "
                f"(workspace={step.workspace})"
            )

            result = self.crew.run(
                route=step.workspace,
                question=question,
                google_email=google_email,
            )

            print("\n[DEBUG] CREW RESULT:")
            print(result)
            print("[DEBUG] END CREW RESULT\n")

            return result

        raise RuntimeError(
            f"Unknown pipeline: {step.pipeline}"
        )

