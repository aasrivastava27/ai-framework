import os

import pandas as pd
from crewai.tools import BaseTool


class CSVAnalyzerTool(BaseTool):

    name: str = "CSV Analyzer"

    description: str = (
        "Analyze a CSV file belonging to the authenticated user. "
        "Can calculate row counts, column information, missing values, "
        "duplicate rows, numeric statistics, and grouped averages."
    )

    def _run(
        self,
        user_google_email: str,
        file_name: str,
        analysis_request: str,
    ):

        csv_path = os.path.join(
            "rag",
            "csv",
            user_google_email,
            os.path.basename(file_name),
        )

        if not os.path.exists(csv_path):

            return {
                "error": (
                    f"CSV file '{file_name}' "
                    "was not found."
                )
            }

        if not csv_path.lower().endswith(".csv"):

            return {
                "error": "Only CSV files are supported."
            }

        try:

            df = pd.read_csv(csv_path)

        except Exception as e:

            return {
                "error": f"Could not read CSV file: {str(e)}"
            }

        result = {
            "file": file_name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing_values": {
                column: int(count)
                for column, count
                in df.isnull().sum().items()
                if count > 0
            },
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
        }

        # -------------------------------------------------
        # Numeric statistics
        # -------------------------------------------------

        numeric_df = df.select_dtypes(
            include="number"
        )

        if not numeric_df.empty:

            result["numeric_statistics"] = (
                numeric_df
                .describe()
                .round(2)
                .to_dict()
            )

        # -------------------------------------------------
        # Group analysis
        # -------------------------------------------------

        request = analysis_request.lower()

        if (
            "average" in request
            or "average salary" in request
            or "mean" in request
        ):

            numeric_columns = list(
                numeric_df.columns
            )

            categorical_columns = [
                column
                for column in df.columns
                if column not in numeric_columns
            ]

            group_column = None

            # Try to find the requested grouping column.
            for column in categorical_columns:

                if column.lower() in request:

                    group_column = column
                    break

            # Otherwise use the first categorical column.
            if (
                group_column is None
                and categorical_columns
            ):

                group_column = categorical_columns[0]

            if (
                group_column is not None
                and numeric_columns
            ):

                group_analysis = {}

                grouped = df.groupby(
                    group_column
                )

                for group_name, group in grouped:

                    group_name = str(group_name)

                    group_analysis[group_name] = {
                        "count": int(len(group))
                    }

                    for numeric_column in numeric_columns:

                        group_analysis[group_name][
                            f"average_{numeric_column}"
                        ] = round(
                            float(
                                group[numeric_column].mean()
                            ),
                            2,
                        )

                result["group_column"] = group_column

                result["group_analysis"] = (
                    group_analysis
                )

        return result