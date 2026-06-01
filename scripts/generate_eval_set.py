import pandas as pd

df = pd.read_csv(
    "data/clinical_trials.csv"
)

df = df.fillna("")

queries = []

sample_df = df.sample(
    n=500,
    random_state=42
)

for _, row in sample_df.iterrows():

    title = str(
        row["Study Title"]
    )

    condition = str(
        row["Conditions"]
    )

    intervention = str(
        row["Interventions"]
    )

    if condition:

        queries.append(
            {
                "query":
                f"What studies investigated {condition}?",
                "relevant_doc":
                condition
            }
        )

    if intervention:

        first_intervention = (
            intervention
            .split("|")[0]
            .replace(
                "DRUG:",
                ""
            )
            .replace(
                "DEVICE:",
                ""
            )
            .replace(
                "OTHER:",
                ""
            )
            .replace(
                "BEHAVIORAL:",
                ""
            )
            .strip()
        )

        if first_intervention:

            queries.append(
                {
                    "query":
                    f"Which trial evaluated {first_intervention}?",
                    "relevant_doc":
                    first_intervention
                }
            )

eval_df = pd.DataFrame(
    queries
)

eval_df = (
    eval_df
    .drop_duplicates()
)

eval_df.to_csv(
    "data/evaluation_queries.csv",
    index=False
)

print(
    f"Generated {len(eval_df)} queries"
)