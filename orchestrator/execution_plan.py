from dataclasses import dataclass


@dataclass
class ExecutionStep:
    pipeline: str
    workspace: str | None = None
    task: str | None = None


def parse_execution_plan(plan: str) -> list[ExecutionStep]:

    steps = []

    for line in plan.strip().splitlines():

        pipeline = None
        workspace = None
        task = None

        # New format:
        # PIPELINE:CREW|WORKSPACE:MAPS|TASK:Find the closest cafe...
        parts = [part.strip() for part in line.split("|")]

        for part in parts:

            if part.startswith("PIPELINE:"):
                pipeline = part.split(":", 1)[1].strip()

            elif part.startswith("WORKSPACE:"):
                workspace = part.split(":", 1)[1].strip()

            elif part.startswith("TASK:"):
                task = part.split(":", 1)[1].strip()

        if pipeline:
            steps.append(
                ExecutionStep(
                    pipeline=pipeline.lower(),
                    workspace=workspace.lower() if workspace else None,
                    task=task,
                )
            )

    return steps