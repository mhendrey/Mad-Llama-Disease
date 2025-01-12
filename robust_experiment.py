import argparse
import optuna

from mad_llama_disease import RobustObjective, make_contour_plot


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
        help="Number of Optuna trials to run. Default is 24",
        default=24,
    )
    parser.add_argument(
        "--n_evaluations",
        type=int,
        help=(
            "Number of rounds of evalution in each trial. "
            + "One evaluation is 150 requests."
        ),
        default=12,
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="Average request rate to submit to OpenAI API endpoint. Default is 90.0",
        default=90.0,
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="SQLite database file to store study results in",
        default="sqlite:///mad_llama_disease_robust.db",
    )

    return parser.parse_args()


def main():
    """
    Launch vLLM with the following command:

    vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 \
        --max-model-len 9216 \
        --max-num-seqs 96 \
        --gpu_memory_utilization 0.95 \
        --served-model-name Llama-3.1-8B
    """
    args = parse_cmd_line()
    study_name = args.study_name
    n_trials = args.n_trials
    n_evaluations = args.n_evaluations
    request_rate = args.request_rate
    storage = args.storage

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
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(
        RobustObjective(
            n_evaluations=n_evaluations,
            prompts=prompts,
            system_prompt=system_prompt,
            request_rate=request_rate,
        ),
        n_trials=n_trials,
    )

    make_contour_plot(
        study,
        f"images/{study_name}_contour.png",
    )


if __name__ == "__main__":
    main()
