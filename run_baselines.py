"""
Run all baselines on all tasks and save the results to a Pandas dataframe.

"""

import argparse
import json
import inspect
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from cik_benchmark.baselines.direct_prompt import DirectPrompt
# from cik_benchmark.baselines.lag_llama import lag_llama
# from cik_benchmark.baselines.chronos import ChronosForecaster
# from cik_benchmark.baselines.moirai import MoiraiForecaster
# from cik_benchmark.baselines.llm_processes import LLMPForecaster
# from cik_benchmark.baselines.timellm import TimeLLMForecaster
# from cik_benchmark.baselines.unitime import UniTimeForecaster
# from cik_benchmark.baselines.timegen import timegen1
from cik_benchmark.baselines.naive import oracle_baseline, random_baseline
# from cik_benchmark.baselines.statsmodels import (
#     ExponentialSmoothingForecaster,
# )
# from cik_benchmark.baselines.r_forecast import R_ETS, R_Arima
from cik_benchmark.evaluation import evaluate_all_tasks
from cik_benchmark.config import RESULT_CACHE_PATH


logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def experiment_naive(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Naive baselines (random and oracle)

    """
    results = []
    results.append(
        (
            "random",
            evaluate_all_tasks(
                random_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/random/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    results.append(
        (
            "oracle",
            evaluate_all_tasks(
                oracle_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/oracle/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    return results, {}


# def experiment_lag_llama(
#     n_samples, output_folder, max_parallel=10, skip_cache_miss=False
# ):
#     """
#     Lag LLAMA baseline

#     """
#     results = evaluate_all_tasks(
#         lag_llama,
#         n_samples=n_samples,
#         output_folder=f"{output_folder}/lag_llama/",
#         max_parallel=max_parallel,
#         skip_cache_miss=skip_cache_miss,
#     )
#     return results, {}


# def experiment_chronos(
#     model_size, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
# ):
#     """
#     Chronos baselines

#     """
#     results = evaluate_all_tasks(
#         ChronosForecaster(model_size=model_size),
#         n_samples=n_samples,
#         output_folder=f"{output_folder}/chronos/",
#         max_parallel=max_parallel,
#         skip_cache_miss=skip_cache_miss,
#     )
#     return results, {}


# def experiment_moirai(
#     model_size, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
# ):
#     """
#     Moirai baselines

#     """
#     results = evaluate_all_tasks(
#         MoiraiForecaster(model_size=model_size),
#         n_samples=n_samples,
#         output_folder=f"{output_folder}/moirai/",
#         max_parallel=max_parallel,
#         skip_cache_miss=skip_cache_miss,
#     )
#     return results, {}


# def experiment_statsmodels(
#     n_samples, output_folder, max_parallel=None, skip_cache_miss=False
# ):
#     """
#     Statsmodels baselines (Exponential Smoothing)

#     """
#     return (
#         evaluate_all_tasks(
#             ExponentialSmoothingForecaster(),
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/exp_smoothing/",
#             max_parallel=max_parallel,
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


# def experiment_r_ets(
#     n_samples, output_folder, max_parallel=None, skip_cache_miss=False
# ):
#     """
#     Baseline using the R "forecast" package: ETS

#     """
#     return (
#         evaluate_all_tasks(
#             R_ETS(),
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/r_ets/",
#             max_parallel=max_parallel,
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


# def experiment_r_arima(
#     n_samples, output_folder, max_parallel=None, skip_cache_miss=False
# ):
#     """
#     Baseline using the R "forecast" package: Arima

#     """
#     return (
#         evaluate_all_tasks(
#             R_Arima(),
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/r_arima/",
#             max_parallel=1,  # Hardcoded as it's buggy with None
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


def experiment_directprompt(
    llm,
    use_context,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
    batch_size=None,
    batch_size_on_retry=5,
    n_retries=3,
    temperature=1.0,
    seeds=5,
):
    """
    DirectPrompt baselines
    """
    # Costs per 1000 tokens
    openai_costs = {
        "gpt-4o": {"input": 0.005, "output": 0.015},  # Same price Azure and OpenAI
        "gpt-35-turbo": {"input": 0.002, "output": 0.002},
        "gpt-3.5-turbo": {"input": 0.003, "output": 0.006},  # OpenAI API
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # OpenAI API
        "llama-3.1-405b": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3.1-405b-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-2-7B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-2-70B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-8B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-8B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-70B": {"input": 0.0, "output": 0.0},  # Toolkit
        "llama-3-70B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "mixtral-8x7B": {"input": 0.0, "output": 0.0},  # Toolkit
        "mixtral-8x7B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "phi-3-mini-128k-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-9B": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-9B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-27B": {"input": 0.0, "output": 0.0},  # Toolkit
        "gemma-2-27B-instruct": {"input": 0.0, "output": 0.0},  # Toolkit
    }
    if not llm.startswith("openrouter-") and llm not in openai_costs:
        raise ValueError(f"Invalid model: {llm} -- Not in cost dictionary")

    # Get cache name first
    temp_forecaster = DirectPrompt(
        model=llm,
        use_context=use_context,
        temperature=temperature,
        dry_run=True
    )
    cache_name = temp_forecaster.cache_name
    del temp_forecaster

    # Create output directory
    full_output_dir = output_folder / cache_name
    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Create the real forecaster with the correct output path
    dp_forecaster = DirectPrompt(
        model=llm,
        use_context=use_context,
        token_cost=openai_costs[llm] if not llm.startswith("openrouter-") else None,
        batch_size=batch_size,
        batch_size_on_retry=batch_size_on_retry,
        n_retries=n_retries,
        temperature=temperature,
        dry_run=skip_cache_miss,
        output_folder=full_output_dir
    )
    
    results = evaluate_all_tasks(
        dp_forecaster,
        n_samples=n_samples,
        output_folder=full_output_dir,
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
        seeds=seeds,
    )
            
    total_cost = dp_forecaster.total_cost
    extra_info = {
        "total_cost": total_cost
    }
    del dp_forecaster

    return results, extra_info


# def experiment_timellm(
#     use_context,
#     dataset,
#     pred_len,
#     n_samples,
#     output_folder,
#     max_parallel=1,
#     skip_cache_miss=False,
# ):
#     """
#     TimeLLM baselines
#     Doesn't use n_samples as it is not implemented in the TimeLLMForecaster

#     """
#     timellm_forecaster = TimeLLMForecaster(
#         use_context=use_context,
#         dataset=dataset,
#         pred_len=pred_len,
#         dry_run=skip_cache_miss,
#     )

#     return (
#         evaluate_all_tasks(
#             timellm_forecaster,
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/{timellm_forecaster.cache_name}",
#             max_parallel=max_parallel,
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


# def experiment_unitime(
#     use_context,
#     pred_len,
#     n_samples,
#     output_folder,
#     dataset="",
#     per_dataset_checkpoint=False,
#     max_parallel=1,
#     skip_cache_miss=False,
# ):
#     """
#     TimeLLM baselines
#     Doesn't use n_samples as it is not implemented in the TimeLLMForecaster

#     """
#     unitime_forecaster = UniTimeForecaster(
#         use_context=use_context,
#         dataset=dataset,
#         pred_len=pred_len,
#         per_dataset_checkpoint=per_dataset_checkpoint,
#         dry_run=skip_cache_miss,
#     )

#     return (
#         evaluate_all_tasks(
#             unitime_forecaster,
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/{unitime_forecaster.cache_name}",
#             max_parallel=max_parallel,
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


# def experiment_timegen1(
#     n_samples, output_folder, max_parallel=10, skip_cache_miss=False
# ):
#     """
#     Nixtla TimeGEN-1 baseline

#     """
#     results = evaluate_all_tasks(
#         timegen1,
#         n_samples=n_samples,
#         output_folder=f"{output_folder}/timegen1/",
#         max_parallel=max_parallel,
#         skip_cache_miss=skip_cache_miss,
#     )
#     return results, {}


# def experiment_llmp(
#     llm, use_context, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
# ):
#     """
#     LLM Process baselines

#     """
#     llmp_forecaster = LLMPForecaster(
#         llm_type=llm, use_context=use_context, dry_run=skip_cache_miss
#     )
#     return (
#         evaluate_all_tasks(
#             llmp_forecaster,
#             n_samples=n_samples,
#             output_folder=f"{output_folder}/{llmp_forecaster.cache_name}",
#             max_parallel=max_parallel,
#             skip_cache_miss=skip_cache_miss,
#         ),
#         {},
#     )


def compile_results(results, extra_infos, output_folder, cap=None):
    # Compile results into Pandas dataframe
    errors = defaultdict(list)
    missing = defaultdict(list)
    results_ = {
        "Task": [task for task in list(results.values())[0]],
    }
    
    # Store forecasts and actuals
    forecasts = defaultdict(dict)
    actuals = defaultdict(dict)
    timestamps = defaultdict(dict)
    prompts = defaultdict(dict)
    
    for method, method_results in results.items():
        _method_results = []
        for task in results_["Task"]:
            task_results = []

            for seed_res in method_results[task]:
                # Keep track of exceptions and missing results
                seed_res["task"] = task
                if "error" in seed_res:
                    if "cache miss" in seed_res["error"].lower():
                        missing[method].append(seed_res)
                    else:
                        errors[method].append(seed_res)
                else:
                    if cap == None:
                        score = seed_res["score"]
                    else:
                        score = min(seed_res["score"], cap)
                    task_results.append(score)
                    
                    # Store forecasts, actuals and timestamps if available
                    if "samples" in seed_res:
                        forecasts[method][task] = seed_res["samples"]
                    if "actuals" in seed_res:
                        actuals[method][task] = seed_res["actuals"]
                    if "timestamps" in seed_res:
                        timestamps[method][task] = seed_res["timestamps"]
                    
                    # Store prompt if available in extra_info
                    if method in extra_infos and "prompt" in extra_infos[method]:
                        prompts[method][task] = extra_infos[method]["prompt"]

            mean = np.mean(task_results)
            std = np.std(task_results, ddof=1)
            stderr = std / np.sqrt(len(task_results))
            _method_results.append(f"{mean.round(3): .3f} ± {stderr.round(3) :.3f}")

        results_[method] = _method_results

    results = pd.DataFrame(results_).sort_values("Task").set_index("Task")
    del results_

    # Save forecasts and actuals as CSV
    print(f"Saving forecasts and actuals")
    if forecasts and actuals and timestamps:
        for method in forecasts:
            for task in forecasts[method]:
                # Create method-specific folder
                method_folder = output_folder / method
                task_folder = method_folder / task
                task_folder.mkdir(parents=True, exist_ok=True)
                
                # Get data for this task
                task_forecasts = forecasts[method][task]
                task_actuals = actuals[method][task]
                task_timestamps = timestamps[method][task]
                
                # Create DataFrame
                df = pd.DataFrame()
                df['timestamp'] = task_timestamps
                df['actual'] = task_actuals.flatten()
                
                # Add each forecast sample as a column
                for i in range(task_forecasts.shape[0]):
                    df[f'forecast_{i+1}'] = task_forecasts[i, :, 0]
                
                # Save to CSV
                df.to_csv(task_folder / "predictions.csv", index=False)

    return results, missing, errors, forecasts, actuals, timestamps, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=25,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to evaluate on each task",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results/",
        help="Output folder for results",
    )
    parser.add_argument(
        "--exp-spec",
        type=str,
        help="Experiment specification file",
    )
    parser.add_argument(
        "--list-exps",
        action="store_true",
        help="List available experiments and their parameters",
    )
    parser.add_argument(
        "--skip-cache-miss",
        action="store_true",
        help="Skip tasks that have not already been computed",
    )
    parser.add_argument(
        "--cap",
        type=float,
        help="Cap value to cap each instance's metric",
    )

    args = parser.parse_args()
    output_folder = Path(args.output)

    # List all available experiments
    if args.list_exps:
        print("Available experiments:")
        # Filter globals to only include functions that start with "experiment_"
        exp_funcs = [
            v
            for k, v in globals().items()
            if k.startswith("experiment_") and inspect.isfunction(v)
        ]

        # Print each experiment function name with a list of its parameters
        for func in exp_funcs:
            # Get the function signature
            signature = inspect.signature(func)
            # List of parameters excluding 'n_samples', 'output_folder', and 'skip_cache_miss'
            params = [
                name
                for name, param in signature.parameters.items()
                if name not in ["n_samples", "output_folder", "skip_cache_miss"]
            ]
            # Print the function name and its parameters
            print(f"\t{func.__name__}({', '.join(params)})")

        exit()

    # Run all experiments
    all_results = {}
    extra_infos = {}
    # ... load specifications
    with open(args.exp_spec, "r") as f:
        exp_spec = json.load(f)
    # ... run each experiment
    for exp in exp_spec:
        current_results = {}
        print(f"Running experiment: {exp['label']}")
        exp_label = exp["label"]
        # ... extract configuration
        config = {k: v for k, v in exp.items() if k != "method" and k != "label"}
        config["n_samples"] = args.n_samples
        config["output_folder"] = output_folder / exp_label
        config["skip_cache_miss"] = args.skip_cache_miss
        config["seeds"] = args.seeds  # Add seeds to config
        print(f"\tConfig: {config}")
        # ... do it!
        function = globals().get(f"experiment_{exp['method']}")
        # ... process results
        res, extra_info = function(**config)
        if isinstance(res, list):
            all_results.update({f"{exp_label}_{k}": v for k, v in res})
            current_results.update({f"{exp_label}_{k}": v for k, v in res})
        else:
            all_results[exp_label] = res
            current_results[exp_label] = res
        extra_infos[exp_label] = extra_info

        # Compile results
        current_results, missing, errors, forecasts, actuals, timestamps, prompts = compile_results(
            current_results, 
            extra_infos, 
            output_folder=output_folder / exp_label,  # Pass the output folder
            cap=args.cap
        )
        print(current_results)
        print("Number of missing results:", {k: len(v) for k, v in missing.items()})
        print("Number of errors:", {k: len(v) for k, v in errors.items()})

        # Save results to CSV
        filename = "results.csv" if not args.cap else f"results-cap-{args.cap}.csv"
        print(f"Saving results to {output_folder/exp_label}/{filename}")
        current_results.to_csv(output_folder / exp_label / filename)
        
        # Save extra info including prompts
        print(f"Saving extra info to {output_folder/exp_label}/extra_info.json")
        with open(output_folder / exp_label / "extra_info.json", "w") as f:
            json.dump(extra_infos[exp_label], f)
            
        # Save missing results and errors
        print(f"Saving missing results to {output_folder/exp_label}/missing.json")
        with open(output_folder / exp_label / "missing.json", "w") as f:
            json.dump(missing, f)
        print(f"Saving errors to {output_folder/exp_label}/errors.json")
        with open(output_folder / exp_label / "errors.json", "w") as f:
            json.dump(errors, f)


if __name__ == "__main__":
    main()
