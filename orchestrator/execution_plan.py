from dataclasses import dataclass


@dataclass
class ExecutionStep:
    pipeline: str
    workspace: str | None = None


def parse_execution_plan(plan: str) -> list[ExecutionStep]:

    steps = []

    for line in plan.strip().splitlines():

        parts = [part.strip() for part in line.split(",")]

        pipeline = None
        workspace = None

        for part in parts:

            if part.startswith("PIPELINE:"):
                pipeline = part.split(":", 1)[1]

            elif part.startswith("WORKSPACE:"):
                workspace = part.split(":", 1)[1]

        if pipeline:
            steps.append(
                ExecutionStep(
                    pipeline=pipeline.lower(),
                    workspace=workspace.lower() if workspace else None,
                )
            )

    return steps