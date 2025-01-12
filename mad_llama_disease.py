from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from openai import OpenAI
import optuna
from time import sleep


class Objective:
    """
    Objective class that uses sampled temperature, min_p, repetition_penalty for each
    trial in an Optuna study.
    """

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
        request_rate: float = 75.0,
        max_completion_tokens: int = 8192,
        model_url: str = "http://localhost:8000/v1",
        model_id: str = "Llama-3.1-8B",
        seed=None,
    ) -> None:
        """Initialize an Objective class to be used by optuna.study.optimize()

        Parameters
        ----------
        n_evaluations : int
            Number of rounds of n_requests to use per trial
        prompts : list[str]
            List of prompts to randomly sample from
        system_prompt : str, optional
            System prompt to provide to LLM, by default ""
        temperature_min : float, optional
            Minimum temperature to investigate, by default 0.5
        temperature_max : float, optional
            Maximum temperature to investigate, by default 1.0
        min_p_min : float, optional
            Minimum min_p to investigate, by default 0.00
        min_p_max : float, optional
            Maximum min_p to investigate, by default 0.04
        repetition_penalty_min : float, optional
            Minimum repetition_penalty to investigate, by default 1.00
        repetition_penalty_max : float, optional
            Maximum repetition penalty to investigate, by default 1.05
        n_requests : int, optional
            Number of requests to LLM per evaluation round, by default 150
        request_rate : float, optional
            Average request rate to simulate traffice, by default 75.0
        max_completion_tokens : int, optional
            Maximum number of generate tokens allowed, by default 8192
        model_url : _type_, optional
            URL endpoint for OpenAI API, by default "http://localhost:8000/v1"
        model_id : str, optional
            Name of the model to use, by default "Llama-3.1-8B"
        seed : _type_, optional
            Seed to set on Numpy's rng, by default None

        Raises
        ------
        ValueError
            Raised if model_id not available at model_url
        ValueError
            Raised if prompts are empty
        """
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
            + f"{temperature:.03f}, {min_p:.04f}, "
            + f"{repetition_penalty:.04f}"
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
            step=0.001,
        )
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

        print(
            f"Trial {trial.number}: {temperature=:.03f}, {min_p=:.04f}, {repetition_penalty=:.04f}"
        )
        results = np.array(
            [
                self.run_trial(eval_id, temperature, min_p, repetition_penalty)
                for eval_id in range(self.n_evaluations)
            ]
        )
        print(f"Trial finished: {results.round(4)}")
        return results.mean()


class RobustObjective(Objective):
    """
    Objective class that uses sampled min_p, repetition_penalty for each trial in an
    Optuna study. Each trial uses uniform spacing of temperatures between
    temperature_min and temperature_max. Goal is to find a set of min_p &
    repetition_penalty that minimize the error acros the full range of temperatures
    that clients could submit.
    """

    def __init__(
        self,
        n_evaluations,
        prompts,
        system_prompt="",
        temperature_min=0.5,
        temperature_max=1,
        min_p_min=0,
        min_p_max=0.04,
        repetition_penalty_min=1,
        repetition_penalty_max=1.05,
        n_requests=75 * 2,
        request_rate=75,
        max_completion_tokens=8192,
        model_url="http://localhost:8000/v1",
        model_id="Llama-3.1-8B",
        seed=None,
    ):
        super().__init__(
            n_evaluations,
            prompts,
            system_prompt,
            temperature_min,
            temperature_max,
            min_p_min,
            min_p_max,
            repetition_penalty_min,
            repetition_penalty_max,
            n_requests,
            request_rate,
            max_completion_tokens,
            model_url,
            model_id,
            seed,
        )
        # Using evenly spaced temperatures for each evaluation in a trial
        # Perhaps could use some other distribution that is centered around default
        # vLLM of 0.7
        self.temperatures = np.linspace(
            temperature_min, temperature_max, n_requests
        ).tolist()
        self.rng.shuffle(self.temperatures)

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
        results = np.array(
            [
                self.run_trial(eval_id, min_p, repetition_penalty)
                for eval_id in range(self.n_evaluations)
            ]
        )
        print(f"Trial finished: {results.round(4)}")
        return results.mean()


def make_contour_plot(study, filename: str, params: list[str] = None) -> None:
    axes = optuna.visualization.matplotlib.plot_contour(
        study, params=params, target_name="Failure Rate"
    )

    figure = axes.get_figure()
    figure.savefig(filename)
