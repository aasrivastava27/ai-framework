from llm.gemini_client import gemini_client
from router.router import RequestRouter, Route
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
        self.rag = RAG()
        self.crew = CrewManager()

    def execute(
        self,
        question: str,
        google_email: str,
    ) -> str:

        steps = self.router.determine_execution_plan(question)

        print(f"\n[DEBUG] Execution Plan: {steps}\n")

        current_result = question
        original_question = question

        for step in steps:

            if step.pipeline == "llm" and current_result != original_question:

                current_result = self.llm.ask(
                    f"""
    You are the final response generator.

    The user asked:
    {original_question}

    A previous pipeline/tool execution produced this result:
    {current_result}

    Your job is to answer the user's original request using the result above.

    Do not ask the user what they want to do with the result.

    Do not treat the result as merely pasted content.

    If the user asked for a summary, summarize it.

    If the user asked for specific information, extract that information.

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
            raise RuntimeError("Execution produced no result.")

        return current_result
    
    def execute_step(
        self,
        step,
        question: str,
        original_question: str,
        google_email: str,
    ) -> str:

        if step.pipeline == "llm":

            print("[DEBUG] Executing LLM step")

            prompt = f"""
        You are the final response generator.

        The user asked:
        {original_question}

        A previous pipeline/tool execution produced this result:
        {question}

        Your job is to answer the user's original request using the result above.

        Do not ask the user what they want to do with the result.

        Do not treat the result as merely pasted content.

        If the user asked for a summary, summarize it.

        If the user asked for specific information, extract that information.

        Only use information contained in the pipeline result.
        Do not invent information.
        """

            return self.llm.ask(prompt)

        elif step.pipeline == "rag":

            print("[DEBUG] Executing RAG step")

            return self.rag.answer(
                question=question,
                chat_history=""
            )

        elif step.pipeline == "crew":

            print(
                f"[DEBUG] Executing CREW step "
                f"(workspace={step.workspace})"
            )

            return self.crew.run(
                route=step.workspace,
                question=question,
                google_email=google_email,
            )

        raise RuntimeError(
            f"Unknown pipeline: {step.pipeline}"
        )