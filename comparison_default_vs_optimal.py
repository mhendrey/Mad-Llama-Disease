import argparse
import numpy as np
import optuna

from mad_llama_disease import Objective


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study_name",
        type=str,
        help="Optuna study name to create and  store results under",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        help="Optimal min_p found in robust study. Default is 0.0113",
        default=0.0113,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Optimal repetition_penalty found in robust study. Default is 1.0498",
        default=1.0498,
    )
    parser.add_argument(
        "--n_evaluations",
        type=int,
        help=(
            "Number of rounds of evalution in each trial. "
            + "One evaluation is 150 requests. Default is 8"
        ),
        default=8,
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Average request rate to submit to OpenAI API endpoint. Default is 90.0",
        default=90.0,
    )
    parser.add_argument(
        "--reuse_study",
        type=str,
        help=("Reuse the parameter configurations of the provided study name"),
        default=None,
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="SQLite database file to store study results in",
        default="sqlite:///mad_llama_disease_comparison.db",
    )

    return parser.parse_args()


def main():
    """
    Launch vLLM with the following command:

    vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 \
        --max-model-len 9216 \
        --max-num-seqs 64 \
        --gpu_memory_utilization 0.95 \
        --served-model-name Llama-3.1-8B
    """
    args = parse_cmd_line()
    study_name = args.study_name
    n_evaluations = args.n_evaluations
    request_rate = args.request_rate
    storage = args.storage
    reuse_study = args.reuse_study

    min_p = args.min_p
    repetition_penalty = args.repetition_penalty

    with open("test_prompts.txt", "r") as f:
        prompts = [line.strip() for line in f]

    system_prompt = (
        "You are a college student taking an exam. You have worked hard in this class "
        + "all semester and want to get an A in the class. You have one hour to write "
        + "a response to the question as completely as you can to demonstrate that you "
        + "have mastered the material. You answer should be well reasoned and "
        + "structured in a logical way. Your answer should have at least 3000 words."
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(),
    )

    if reuse_study:
        print(f"Reusing trial params from {reuse_study}")
        old_trials = optuna.load_study(study_name=reuse_study, storage=storage).trials
        n_trials = len(old_trials)
        print(f"{reuse_study} has {n_trials} that will be reused")
        for trial in old_trials:
            study.enqueue_trial(trial.params)
    else:
        n_trials = 0
        for temperature in np.linspace(0.5, 1.0, 6):
            # Add in default values
            study.enqueue_trial(
                {
                    "temperature": temperature,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                }
            )
            n_trials += 1
            # Add in optimal values
            study.enqueue_trial(
                {
                    "temperature": temperature,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                }
            )
            n_trials += 1

    study.optimize(
        Objective(
            n_evaluations=n_evaluations,
            prompts=prompts,
            system_prompt=system_prompt,
            request_rate=request_rate,
        ),
        n_trials=n_trials,
    )


if __name__ == "__main__":
    main()
