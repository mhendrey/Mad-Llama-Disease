import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from openai import OpenAI
import optuna
from time import sleep


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
        "--storage",
        type=str,
        help="SQLite database file to store study results in",
        default="sqlite:///mad_llama_disease_robust.db",
    )

    return parser.parse_args()


class Objective:
    def __init__(
        self,
        n_evaluations: int,
        prompts: list[str],
        system_prompt: str = "",
        temperature_min: float = 0.5,
        temperature_max: float = 1.0,
        min_p_min: float = 0.00,
        min_p_max: float = 0.04,
        repetition_penalty_min: float = 1.00,
        repetition_penalty_max: float = 1.05,
        n_requests: int = 150,
        request_rate: float = 90.0,
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

        # Using evenly spaced temperatures for each evaluation in a trial
        # Perhaps could use some other distribution that is centered around default
        # vLLM of 0.7
        self.temperatures = np.linspace(
            temperature_min, temperature_max, n_requests
        ).tolist()
        self.rng.shuffle(self.temperatures)

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
        min_p: float,
        repetition_penalty: float,
    ) -> float:
        print(
            f"  {datetime.now()} Starting evaluation {eval_id+1}: "
            + f"{min_p:.04f}, {repetition_penalty:.04f}"
        )
        arrival_times = np.cumsum(
            self.rng.exponential(scale=60.0 / self.request_rate, size=self.n_requests)
        )
        prompts = self.rng.choice(self.prompts, self.n_requests, replace=True)

        start = datetime.now()
        n_failed = 0
        with ThreadPoolExecutor(256) as executor:
            futures = {}
            for i, arrival_time, prompt, temperature in zip(
                range(self.n_requests), arrival_times, prompts, self.temperatures
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
        print(f"  Finished with {n_failed=:} and {failure_rate=:.04f}")

        return failure_rate

    def __call__(self, trial):
        min_p = trial.suggest_float(
            "min_p",
            self.min_p_min,
            self.min_p_max,
            step=0.0001,
        )
        repetition_penalty = trial.suggest_float(
            "repetition_penalty",
            self.repetition_penalty_min,
            self.repetition_penalty_max,
            step=0.0001,
        )

        print(f"Trial {trial.number}: {min_p=:.04f}, {repetition_penalty=:.04f}")
        results = [
            self.run_trial(eval_id, min_p, repetition_penalty)
            for eval_id in range(self.n_evaluations)
        ]
        print(f"Trial finished: {results}")
        return np.mean(results)


def make_contour_plot(study, filename: str) -> None:
    axes = optuna.visualization.matplotlib.plot_contour(
        study, params=["min_p", "repetition_penalty"], target_name="Failure Rate"
    )

    figure = axes.get_figure()
    figure.savefig(filename)


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
        Objective(
            n_evaluations=n_evaluations, prompts=prompts, system_prompt=system_prompt
        ),
        n_trials=n_trials,
    )

    make_contour_plot(
        study,
        f"images/{study_name}_contour.png",
    )


if __name__ == "__main__":
    main()
