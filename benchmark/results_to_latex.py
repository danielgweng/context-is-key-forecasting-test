import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from benchmark import TASK_NAME_TO_WEIGHT, ALL_TASKS


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
)
parser.add_argument(
    "--output",
    type=str,
)
parser.add_argument(
    "--weight_average",
    action="store_true",
)
parser.add_argument(
    "--weight_skills",
    action="store_true",
)
args = parser.parse_args()


def find_indices_of_search_string(elements, search_string):
    return [index for index, element in enumerate(elements) if search_string in element]


def extract_mean_std(performance_str):
    try:
        # Split the string at ' ± ' and return the mean and std as floats
        mean, std = performance_str.split(" ± ")
        return float(mean), float(std)
    except:
        return np.nan, np.nan


# Model name map
model_name_map = {
    "CC-GPT-4o": "Direct Prompt - GPT-4o",
    "CC-GPT-4o (no ctx)": "Direct Prompt - GPT-4o (no context)",
    "CC-Llama-3.1-405b-instruct": "Direct Prompt - Llama-3.1-405B-Instruct",
    "CC-Llama-3.1-405b-instruct (no ctx)": "Direct Prompt - Llama-3.1-405B-Instruct (no context)",
    "CC-GPT-4o-mini": "Direct Prompt - GPT-4o-mini",
    "CC-GPT-4o-mini (no ctx)": "Direct Prompt - GPT-4o-mini (no context)",
    "LLama3-8B": "LLMP - LLama3-8B",
    "LLama3-8B (no ctx)": "LLMP - LLama3-8B (no context)",
    "LLama3-8B-instruct": "LLMP - LLama3-8B-Instruct",
    "LLama3-8B-instruct (no ctx)": "LLMP - LLama3-8B-Instruct (no context)",
    "LLama3-70B-instruct": "LLMP - LLama3-70B-Instruct",
    "LLama3-70B-instruct (no ctx)": "LLMP - LLama3-70B-Instruct (no context)",
    "Statsmodels": "Exponential Smoothing (no context)",
    "Lag-Llama": "Lag-Llama (no context)",
}

skill_name_map = {
    "instruction following": "Instruction Following",
    "retrieval: context": "Retrieval: Context",
    "retrieval: memory": "Retrieval: Memory",
    "reasoning: deduction": "Reasoning: Deduction",
    "reasoning: analogy": "Reasoning: Analogy",
    "reasoning: math": "Reasoning: Math",
    "reasoning: causal": "Reasoning: Causal",
}

# Desired skill order
desired_skill_order = [
    "instruction following",
    "retrieval: context",
    "retrieval: memory",
    "reasoning: deduction",
    "reasoning: analogy",
    "reasoning: math",
    "reasoning: causal",
]

desired_context_source_order = ["c_i", "c_h", "c_f", "c_cov", "c_causal"]

# Tasks to ignore
tasks_to_ignore = [
    "SimilarLocationDaySolarForecastTask",
    "SimilarLocationWithReferenceDaySolarForecastTask",
    "ExplicitSimilarLocationDaySolarForecastTask",
]  #  "SimilarLocationDaySolarForecastTask", "SimilarLocationWithReferenceDaySolarForecastTask", "ExplicitSimilarLocationDaySolarForecastTask" --> Llama-3-8B performs terribly; also cannot run Llama-405B
models_to_ignore = ["Naive_random", "Naive_oracle"]

# Track ignored tasks and ignored models
ignored_tasks = []
ignored_models = []

# Read the results csv
data = pd.read_csv(args.input)

# Apply the function to each performance cell, excluding the 'Task' column
performance_means = data.drop(columns=["Task"]).applymap(
    lambda x: extract_mean_std(x)[0]
)
performance_stderrs = data.drop(columns=["Task"]).applymap(
    lambda x: extract_mean_std(x)[1]
)

# Create a copy of the performance_means DataFrame to avoid modifying the original
performance_means_copy = performance_means.copy()
performance_stderrs_copy = performance_stderrs.copy()

# Step 1: Remove tasks where all models have NaN
for index, row in performance_means_copy.iterrows():
    task = data["Task"].iloc[index]

    # If all models for this task have NaN, ignore the task
    if row.isna().all() or task in tasks_to_ignore:
        ignored_tasks.append(task)
        # Drop the task from the DataFrame
        performance_means_copy.drop(index, inplace=True)
        performance_stderrs_copy.drop(index, inplace=True)

# Step 2: Remove models that have NaN in any task or if it is in models_to_ignore
for model in performance_means_copy.columns:
    if performance_means_copy[model].isna().any() or model in models_to_ignore:
        ignored_models.append(model)
        # Drop the model from the DataFrame if it has any NaN
        performance_means_copy.drop(columns=[model], inplace=True)
        performance_stderrs_copy.drop(columns=[model], inplace=True)

# Dynamically extract skills and context sources using TaskName._skills and TaskName._context_sources
skills_map = {}
context_sources_map = {}

# To get class
TASKS_STR_TO_TASK = {x.__name__: x for x in ALL_TASKS}

for task in data["Task"]:
    associated_class = TASKS_STR_TO_TASK[task]
    # Access the _skills attribute, excluding "forecasting" and "natural language processing"
    skills_map[task] = [
        s
        for s in associated_class._skills
        if not s in {"forecasting", "natural language processing"}
    ]
    # Access the _context_sources attribute
    context_sources_map[task] = associated_class._context_sources

# For each model, calculate the average performance for each skill and context source

# 1. Create the LaTeX table for skills
skill_data = {}
for model in performance_means_copy.columns:
    entry = {}
    for skill in desired_skill_order:
        total_weight = 0
        total_values = 0.0
        total_variances = 0.0
        for task, task_skills in skills_map.items():
            if (
                skill in task_skills
                and not task in ignored_tasks
                and not task in tasks_to_ignore
            ):
                weight = TASK_NAME_TO_WEIGHT[task] if args.weight_skills else 1
                value = performance_means_copy.loc[data["Task"] == task, model].values[
                    0
                ]
                stderr = performance_stderrs_copy.loc[
                    data["Task"] == task, model
                ].values[0]
                total_weight += weight
                total_values += weight * value
                total_variances += (weight * stderr) ** 2
        mean = total_values / total_weight
        variance = total_variances / (total_weight**2)
        stderr = variance**0.5
        entry[skill] = f"{mean:.3f} ± {stderr:.3f}"

    total_weight = 0
    total_values = 0.0
    total_variances = 0.0
    for task, task_skills in skills_map.items():
        if not task in ignored_tasks and not task in tasks_to_ignore:
            weight = TASK_NAME_TO_WEIGHT[task] if args.weight_average else 1
            value = performance_means_copy.loc[data["Task"] == task, model].values[0]
            stderr = performance_stderrs_copy.loc[data["Task"] == task, model].values[0]
            total_weight += weight
            total_values += weight * value
            total_variances += (weight * stderr) ** 2
    mean = total_values / total_weight
    variance = total_variances / (total_weight**2)
    stderr = variance**0.5
    entry["Average"] = f"{mean:.3f} ± {stderr:.3f}"

    skill_data[model] = entry
skill_df = pd.DataFrame(skill_data).T


# Sort the DataFrame by "Average Rank" (best to worst)
skill_df = skill_df.sort_values(by="Average")

# Reorder the columns of the skill DataFrame based on the desired order
skill_df = skill_df[desired_skill_order + ["Average"]]
# skill_df.to_excel("skill_df.xlsx", sheet_name="Results skill wise")

# Map row and column name
skill_df = skill_df.rename(index=model_name_map)
skill_df = skill_df.rename(columns=skill_name_map)

# Create LaTeX table for skills
latex_skill_table = skill_df.to_latex(index=True)
latex_skill_table = latex_skill_table.replace("±", r"$\pm$")

# 2. Create the LaTeX table for context sources
context_data = {}
for model in performance_means_copy.columns:
    entry = {}
    for context in desired_context_source_order:
        total_weight = 0
        total_values = 0.0
        total_variances = 0.0
        for task, task_contexts in context_sources_map.items():
            if (
                context in task_contexts
                and not task in ignored_tasks
                and not task in tasks_to_ignore
            ):
                weight = TASK_NAME_TO_WEIGHT[task] if args.weight_skills else 1
                value = performance_means_copy.loc[data["Task"] == task, model].values[
                    0
                ]
                stderr = performance_stderrs_copy.loc[
                    data["Task"] == task, model
                ].values[0]
                total_weight += weight
                total_values += weight * value
                total_variances += (weight * stderr) ** 2
        mean = total_values / total_weight
        variance = total_variances / (total_weight**2)
        stderr = variance**0.5
        entry[context] = f"{mean:.3f} ± {stderr:.3f}"

    total_weight = 0
    total_values = 0.0
    total_variances = 0.0
    for task, task_contexts in context_sources_map.items():
        if not task in ignored_tasks and not task in tasks_to_ignore:
            weight = TASK_NAME_TO_WEIGHT[task] if args.weight_average else 1
            value = performance_means_copy.loc[data["Task"] == task, model].values[0]
            stderr = performance_stderrs_copy.loc[data["Task"] == task, model].values[0]
            total_weight += weight
            total_values += weight * value
            total_variances += (weight * stderr) ** 2
    mean = total_values / total_weight
    variance = total_variances / (total_weight**2)
    stderr = variance**0.5
    entry["Average"] = f"{mean:.3f} ± {stderr:.3f}"

    context_data[model] = entry
context_df = pd.DataFrame(context_data).T


# Sort the DataFrame by "Average Rank" (best to worst)
context_df = context_df.sort_values(by="Average")

# Reorder the columns of the DataFrame based on the desired order
context_df = context_df[desired_context_source_order + ["Average"]]

# Map row and column name
context_df = context_df.rename(index=model_name_map)

# Create LaTeX table for context sources
latex_context_table = context_df.to_latex(index=True)
latex_context_table = latex_context_table.replace("±", r"$\pm$")

# Save LaTeX tables to a file on your local machine
with open(args.output, "w") as f:
    f.write(latex_skill_table)
    f.write("\n\n")
    f.write(latex_context_table)

# Step 6: Print ignored tasks and models
print("\nIgnored Tasks:", ignored_tasks)
print("\nIgnored Models:", ignored_models)
