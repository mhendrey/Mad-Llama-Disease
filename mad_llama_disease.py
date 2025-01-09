import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import optuna
from pprint import pprint
from time import sleep


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study_name_base",
        type=str,
        help="Base name will have '_phase1' & '_phase2' added to it",
    )

    return parser.parse_args()


class Objective:
    def __init__(
        self,
        n_evaluations: int,
        prompts: list[str],
        system_prompt: str = "",
        temperature_min: float = 0.7,
        temperature_max: float = 0.9,
        min_p_min: float = 0.0,
        min_p_max: float = 0.1,
        repetition_penalty_min: float = 1.0,
        repetition_penalty_max: float = 1.1,
        n_requests: int = 60 * 2,
        request_rate: float = 60.0,
        max_completion_tokens: int = 8192,
        model_url: str = "http://localhost:8000/v1",
        model_id: str = "Llama-3.1-8B",
        seed=None,
    ) -> None:
        if not prompts:
            raise ValueError("Prompts must not be empty")

        self.client = OpenAI(api_key="EMPTY", base_url=model_url)
        available_models = [model.id for model in self.client.models.list().data]
        if model_id not in available_models:
            raise ValueError(f"{model_id} is not in {available_models=:}")
        self.model_id = model_id

        self.n_evaluations = n_evaluations
        self.prompts = prompts
        self.system_prompt = system_prompt

        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.min_p_min = min_p_min
        self.min_p_max = min_p_max
        self.repetition_penalty_min = repetition_penalty_min
        self.repetition_penalty_max = repetition_penalty_max

        self.n_requests = n_requests
        self.request_rate = request_rate
        self.max_completion_tokens = max_completion_tokens
        self.rng = np.random.default_rng(seed=seed)

    def get_text(
        self,
        user_prompt: str,
        temperature: float = 0.8,
        min_p: float = 0.05,
        repetition_penalty: float = 1.075,
    ):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        result = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_completion_tokens,
            temperature=temperature,
            extra_body={"min_p": min_p, "repetition_penalty": repetition_penalty},
        )

        return result.usage.completion_tokens

    def run_trial(
        self,
        eval_id: int,
        temperature: float,
        min_p: float,
        repetition_penalty: float,
    ) -> float:
        print(
            f"  {datetime.now()} Starting evaluation {eval_id+1}: "
            + f"{temperature:.03f}, {min_p:.02f}, "
            + f"{repetition_penalty:.02f}"
        )
        arrival_times = np.cumsum(
            self.rng.exponential(scale=60.0 / self.request_rate, size=self.n_requests)
        )
        prompts = self.rng.choice(self.prompts, self.n_requests, replace=True)

        start = datetime.now()
        n_failed = 0
        with ThreadPoolExecutor(256) as executor:
            futures = {}
            for i, arrival_time, prompt in zip(
                range(self.n_requests), arrival_times, prompts
            ):
                current_time = (datetime.now() - start).total_seconds()
                wait_time = arrival_time - current_time
                if wait_time > 0.0:
                    sleep(wait_time)
                elif abs(wait_time) > 0.01:
                    print(
                        f"  {datetime.now()} Request {i:04} fell behind "
                        + f"schedule by {wait_time}"
                    )

                # while arrival_time - current_time > 0.05:
                #    sleep(0.1)
                #    current_time += 0.1

                futures[
                    executor.submit(
                        self.get_text,
                        prompt,
                        temperature,
                        min_p,
                        repetition_penalty,
                    )
                ] = i
                done = []
                for future in futures:
                    if future.done():
                        done.append(future)
                        try:
                            completion_tokens = future.result()
                            request_id = futures[future]
                        except Exception as exc:
                            raise RuntimeError(
                                f"Error: {exc}. Is the model server running?"
                            )
                        else:
                            if completion_tokens == self.max_completion_tokens:
                                n_failed += 1
                                print(f"  Failed {request_id=}")
                for future in done:
                    futures.pop(future)

            for future in as_completed(futures):
                try:
                    completion_tokens = future.result()
                    request_id = futures[future]
                except Exception as exc:
                    raise RuntimeError(f"Error: {exc}. Is the model server running?")
                else:
                    if completion_tokens == self.max_completion_tokens:
                        n_failed += 1
                        print(f"  Failed {request_id=}")

        failure_rate = 100.0 * n_failed / self.n_requests
        print(f"  Finished with {n_failed=:} and {failure_rate=:.03f}")

        return failure_rate

    def __call__(self, trial):
        temperature = trial.suggest_float(
            "temperature",
            self.temperature_min,
            self.temperature_max,
            step=0.025,
        )
        min_p = trial.suggest_float(
            "min_p",
            self.min_p_min,
            self.min_p_max,
            step=0.01,
        )
        repetition_penalty = trial.suggest_float(
            "repetition_penalty",
            self.repetition_penalty_min,
            self.repetition_penalty_max,
            step=0.01,
        )

        print(f"Trial: {temperature=:.03f}, {min_p=:.02f}, {repetition_penalty=:.02f}")
        results = [
            self.run_trial(eval_id, temperature, min_p, repetition_penalty)
            for eval_id in range(self.n_evaluations)
        ]
        print(f"Trial finished: {results}")
        return np.mean(results), repetition_penalty


def make_pareto_plot(study, filename: str, special_trials: list[int] = []) -> None:
    axes = optuna.visualization.matplotlib.plot_pareto_front(
        study,
        target_names=["Failure Rate (%)", "Reptition Penalty"],
    )
    # Add black boxes trials listed in special_trials
    for trial_number in special_trials:
        axes.plot(
            *study.trials[trial_number].values,
            "s",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2,
        )

    figure = axes.get_figure()
    figure.savefig(filename)


def get_candidate_trials(
    study, n_candidates: int = 10
) -> tuple[list[dict], list[float]]:
    """
    Sort trials by failure rate, repetition penalty, min_p, and temperature.
    Drop any duplicate parameter sets (keeping lowest failure rate) and return
    the top n_candidates parameters and their corresponding failure rates.

    Parameters
    ----------
    study : optuna.study.Study
    n_candidates : int, optional
        Number of top scoring trials to return, by default 10

    Returns
    -------
    tuple[list[dict], list[float]]
        List of candidate trial parameters and corresponding failure rates
    """
    df = study.trials_dataframe()[
        ["values_0", "params_repetition_penalty", "params_min_p", "params_temperature"]
    ].rename(
        columns={
            "values_0": "failure_rate",
            "params_repetition_penalty": "repetition_penalty",
            "params_min_p": "min_p",
            "params_temperature": "temperature",
        }
    )

    df_candidates = (
        df.sort_values(["failure_rate", "repetition_penalty", "min_p", "temperature"])
        .drop_duplicates(["repetition_penalty", "min_p", "temperature"], keep="first")
        .head(n_candidates)
    )

    candidate_trials = df_candidates[
        ["repetition_penalty", "temperature", "min_p"]
    ].to_dict(orient="records")
    candidate_failure_rates = df_candidates.failure_rate.tolist()

    return candidate_trials, candidate_failure_rates


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
    study_name_base = args.study_name_base

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

    ## Phase 1: Find Candidate Trials for longer runs in Phase 2
    # Create the initial study
    study = optuna.create_study(
        study_name=f"{study_name_base}_phase1",
        storage="sqlite:///mad_llama_disease.db",
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(),
    )

    # Let's add in vLLM's default parameters to the study's queue so we can
    # have it evaluated for comparision sake.
    study.enqueue_trial(vllm_defaults)

    study.optimize(
        Objective(
            n_evaluations=5,
            prompts=prompts,
            system_prompt=system_prompt,
        ),
        n_trials=24,
    )

    make_pareto_plot(
        study,
        f"images/{study_name_base}_phase1_pareto_front.png",
        special_trials=[0],
    )

    ## Phase 2: Longer runs for the most promising trials + vLLM default parameters
    candidate_trials, candidate_failure_rates = get_candidate_trials(
        study, n_candidates=10
    )

    print("Selected candidates")
    for candidate_trial, candidate_failure_rate in zip(
        candidate_trials, candidate_failure_rates
    ):
        print(f" {candidate_failure_rate:.04f}", end=", ")
        pprint(candidate_trial)

    study2 = optuna.create_study(
        study_name=f"{study_name_base}_phase2",
        storage="sqlite:///mad_llama_disease.db",
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(),
    )
    # Add in vLLM default
    study2.enqueue_trial(vllm_defaults)
    n_trials = 1
    # Add in candidate trials if they don't match vLLM default
    for params in candidate_trials:
        if params != vllm_defaults:
            study2.enqueue_trial(params)
            n_trials += 1

    study2.optimize(
        Objective(
            n_evaluations=15,
            prompts=prompts,
            system_prompt=system_prompt,
        ),
        n_trials=n_trials,
    )

    make_pareto_plot(
        study2, f"images/{study_name_base}_phase2_pareto_front.png", special_trials=[0]
    )


if __name__ == "__main__":
    main()
