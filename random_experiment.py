import argparse
import optuna

from mad_llama_disease import Objective, make_contour_plot


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study_name",
        type=str,
        help="Optuna study name to create and  store results under",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        help="Number of randomly sampled parameter configurations to run",
        default=81,
    )
    parser.add_argument(
        "--n_evaluations",
        type=int,
        help=(
            "Number of rounds of evalution in each trial. "
            + "One evaluation is 150 requests."
        ),
        default=4,
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Average request rate to submit to OpenAI API endpoint. Default is 75.0",
        default=75.0,
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
        default="sqlite:///mad_llama_disease_random.db",
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
    n_trials = args.n_trials
    n_evaluations = args.n_evaluations
    request_rate = args.request_rate
    storage = args.storage
    reuse_study = args.reuse_study

    with open("test_prompts.txt", "r") as f:
        prompts = [line.strip() for line in f]

    system_prompt = (
        "You are a college student taking an exam. You have worked hard in this class "
        + "all semester and want to get an A in the class. You have one hour to write "
        + "a response to the question as completely as you can to demonstrate that you "
        + "have mastered the material. You answer should be well reasoned and "
        + "structured in a logical way. Your answer should have at least 3000 words."
    )

    # vLLM, version 0.6.5 or earlier, which is what we currently run in production
    vllm_defaults = {"temperature": 0.7, "min_p": 0.0, "repetition_penalty": 1.0}

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
        # Let's add in vLLM's default parameters to the study's queue so we can
        # have it evaluated for comparision sake.
        study.enqueue_trial(vllm_defaults)

    study.optimize(
        Objective(
            n_evaluations=n_evaluations,
            prompts=prompts,
            system_prompt=system_prompt,
            request_rate=request_rate,
        ),
        n_trials=n_trials,
    )

    make_contour_plot(
        study,
        f"images/{study_name}_contour_repetition_vs_temperature.png",
        params=["repetition_penalty", "temperature"],
    )


if __name__ == "__main__":
    main()
