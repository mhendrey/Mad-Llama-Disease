# Mad Llama Disease: Investigating and Fixing Endless Generation Issues in Llama-3.1

## Introduction

Large Language Models (LLMs) have become critical components in many organizations' tech stacks, powering everything from chatbots to automated workflows. However, running these models in production presents unique challenges. In this blog post, I'll share our experience addressing a specific issue with the Llama-3.1 model family: the occasional failure to generate an end-of-string (EOS) token.

## The Problem

Like many teams, we leverage the [vLLM engine](https://docs.vllm.ai) for running self-hosted LLMs in production. While running Llama-3.1 models (both 8B and 70B variants), we've observed an interesting failure mode, which others in the LLM community of observed too, where the model occasionally fails to generate an end-of-string token. The model starts off generating acceptable outputs, but eventually degrades into endless gibberish. If clients use the default `max_tokens` (or `max_completion_tokens` for OpenAI API), which of course nearly everyone does, then this causes the model to continue generating tokens until it reaches the maximum context window size of 128K tokens, resulting in two significant problems:

1. **Performance Impact**: Generating 128K tokens of mostly gibberish takes approximately 30 minutes, creating an unacceptable user experience
2. **Stability Issues**: The extended generation fills up the KV cache, exhausting the GPU's VRAM, and eventually causing the container to crash (occurring 7-10 times daily)

## Initial Mitigation

By [default](https://github.com/vllm-project/vllm/blob/730e9592e97c643474aa44e9d3dbe6f55c4b9ad9/vllm/entrypoints/openai/serving_chat.py#L190), vLLM sets the `max_tokens` to be equal to the remaining tokens in the context window. The simplest solution is to set the `max_tokens` parameter in the request to a lower value. By limiting generation to 8,192 tokens, we can prevent the worst impacts of this failure mode. This is also inline with major commercial models as well:

* [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table) (8192)
* [Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.5-pro) (8192)
* [OpenAI](https://platform.openai.com/docs/models) (ranges from 4K-16K dpending on the model)

Currently this would require adding this limit to every client's incoming request via a proxy server. While this is a crucial first step that we recommend implementing immediately, we wanted to explore whether we could reduce the probability of this failure through optimized sampling parameters.

## Initial Random Experiment
The initial phase of this work investigates the impact of three key sampling parameters on this particular failure mode. We selected the following parameters due to their popularity with clients (temperature) and from discussions on social media and Github issues to try and control this behavior:

1. **temperature**: Controls sampling randomness (higher values = more random)
2. **min_p**: Sets the minimum probability threshold for token consideration
3. **repetition_penalty**: Penalizes tokens based on their previous appearances in generated tokens

### Methodology
We conducted our experiments using a 4-bit quantized version of Llama-3.1, which is used in our production service because it optimizes GPU VRAM usage and total token throughput while [maintaining model quality](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/). Our testing used the [quantized Llama-3.1-8B model](https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16). We used [Optuna](https://optuna.readthedocs.io) to run a study using the `RandomSampler`.
 
* We explored the three sampling parameters within the following ranges:
  * temperature: 0.5 - 1.0
  * min_p: 0.0 - 0.04
  * repetition_penalty: 1.0 - 1.05
* Performed 81 trials
  * First trial used vLLM's defaults (temperature=0.7, min_p=0.0, repetition_penalty=1.0)
  * Remaining 80 trials were randomly sampled uniformly between ranges listed above
* Each trial simulated real-world conditions by:
  * Sending requests at an exponentially-distributed rate of 75 per minute
  * Running a total 600 requests (~8 minutes) in 4 independent sets of 150
  * Using randomly sampled prompts requesting lengthy exam responses
  * Measuring failure rate based on percentage of requests hitting the max_tokens limit (8,192)

### Results
Optuna includes a number of visualizations of a study that we used to create the following plots. The figure below shows a measure of the parameter importance in minimizing the failure rate.

![Parameter importance plot](images/int4_random_param_importance.png)

We note that the importance of the repetition_penalty and temperature dominate over that of the min_p request parameter. Given that, let's look at the contour plot of the temperature and repetition_penalty to see how they affect the failure rate.

![Contour Plot](images/int4_random_contour_repetition_vs_temperature.png)

Of the 81 trials, there are 22 trials that had a failure rate of 0.0. These are seen in the upper right corner of the contour plot. In comparison, the vLLM default values (highlighted with a magenta box) had a failure rate of 1.5%.

### Effect of Model Quantization
Since we are using a 4-bit quantized model, one question that needs to be asked is whether the 4-bit quantization is causing an increase in the failure rate. To investigate this, we rerun the first 50 trials of the random experiment to compare the number of failures observed between the 4-bit quantized model and an [FP8 quantized Llama-3.1-8B](https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8)

Here is the contingency table for the overall success/failure counts between the two models across the 50 trials in common.

| Model | Success | Failure | Failure Rate (%) |
|:-----:|:-------:|:-------:|:----------------:|
| INT4  | 29,834  |  166    |    0.553         |
| FP8   | 29,955  |   45    |    0.150         |

Using Fisher's Exact Test, we can reject the null hypothesis that the INT4 & FP8 failure rates are the same with a p-value of 1.52e-17. So we do see that using a quantized model does increase the failure rate by ~3.7x, but even using the FP8 model the failure rate is still not zero across the randomly selected request parameters in the study.

Here is the empirical cummulative distribution of the difference between the INT4 failure rate and the FP8 failure rate over the 50 trials.

![ECDF of Diff in INT4-FP8 Failure Rate](images/ecdf_failure-rate_INT4-FP8.png)

There are a few trials where the INT4 failure rate was lower than the FP8, but as we can see the majority of the time the INT4 failure rate exceeds the FP8 failure rate.

## Temperature Robustness Experiment
Though the previous experiment showed that using a higher temperature would lower the failure rate, the temperature request parameter is the mostly likely parameter to be set by clients who will set it according to their needs. Thus, we would like to find an optimal min_p and repetition_penalty that minimizes the failure rate across the entire range of temperatures (0.5 - 1.0) that are most likely to be set by clients.

For this experiment, we make the following modifications:

1. Use Optuna's TPESampler that minimizes the failure rate using just min_p and repetition_penalty.
2. Each trial (given min_p and repetition_penalty) will submit requests where the temperature for each request is uniformly sampled from temperatures that span the entire [0.5, 1.0] range.
3. Increase the total number of requests to 1,800
4. Perform a total of 24 trials with `n_startup_trials` = 10 (TPESampler's default)

### Results
Given that the random experiment showed that a repetition_penalty near 1.05 provided failure rates of 0.0 for a large swath of temperatures (~0.65 - 1.0) it's not surprising that for this study the following optimal values were found to be:
* **min_p = 0.0113**
* **repetition_penalty = 1.0498**

These were the only values that managed to have a failure rate of 0.0% across the 1,800 requests evenly spaced across the temperature range.

![Contour Plot for Robustness Study](images/int4_robust_contour.png)

The contour plot shows that we are better off picking a relative small value for the min_p, ~0.015 or less, which fits well with reported values for improved LLM outputs.

## Comparison Experiment
In our final experiment, we compare the performance of using the default min_p and repetition_penalty (0.0 and 1.0, respectivley) compared to using the optimal values found in the temperature robustness experiment.  For this work we simply measure the failure rate for both sets of min_p/repetition_penalty for temperatures ranging from [0.5, 1.0].  We perform 1,200 total requests for a given trial.

## Implementation Recommendations

Based on our findings, we recommend a two-step approach:

1. **Immediate Action**: Set the `max_tokens` request parameter for all client requests to 8,192 tokens to prevent system crashes
2. **Parameter Optimization**: Set min_p=0.0113 and repetition_penalty=1.0498, which achieved lowest failure rate across all temperatures between 0.5 and 1.0, for all client requests

## Future Steps
These experiments did not take into account the request parameters affect on the quality of the LLMs output. A natural next step would be to use the optimal request parameters found on the various LLM benchmarks to ensure that there isn't a noticable drop in performance. This is would be similar to what Neural Magic did with their [quantization study](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/).


## Optimization Experiment

We designed an experiment to investigate how three key sampling parameters affect the failure rate:

1. **temperature**: Controls sampling randomness (higher values = more random)
2. **min_p**: Sets the minimum probability threshold for token consideration
3. **repetition_penalty**: Penalizes tokens based on their previous appearances

Our optimization had two objectives:

1. **Primary**: Minimize the failure rate (percentage of requests generating `max_tokens` output tokens)
2. **Secondary**: Minimize deviation from the default repetition_penalty value of 1.0

### Methodology

We used [Optuna](https://optuna.readthedocs.io) to optimize Llama-3.1's sampling parameters through a two-phase approach:

#### Phase 1: Parameter Space Exploration
* We explored three key sampling parameters within these ranges:
  * temperature: 0.7-0.9
  * min_p: 0.0-0.1
  * repetition_penalty: 1.0-1.1    
* The exploration consisted of 24 trials using Optuna's TPE Sampler:
  * First trial used vLLM's defaults (temperature=0.7, min_p=0.0, repetition_penalty=1.0)
  * Remaining 23 trials were selected by the TPE Sampler
* Each trial simulated real-world conditions by:
  * Sending requests at an exponentially-distributed rate of 60 per minute
  * Running 600 requests (~10 minutes) in 5 independent sets
  * Using randomly sampled prompts requesting lengthy exam responses
  * Measuring failure rate based on percentage of requests hitting the max_tokens limit (8,192)

#### Phase 2: Deep Evaluation
* We selected the 10 most promising parameter combinations from Phase 1
* Each combination was tested with 1,800 requests (~30 minutes) to refine failure rates
* We maintained vLLM's default parameters as a baseline throughout

## Results with 4-bit Quantized Llama-3.1-8B

We conducted our experiments using a 4-bit quantized version of Llama-3.1, which is used in our production service because it optimizes GPU VRAM usage and total token throughput while [maintaining model quality](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/). Our testing used the [quantized Llama-3.1-8B model](https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16).

### Phase 1 Results

Initial testing revealed several promising parameter combinations that achieved a 0% failure rate, compared to the default vLLM's settings 0.5% failure rate. The most promising combinations balanced low failure rates with minimal parameter adjustments.

![Phase 1 Pareto Plot](images/int4_pareto_plot_phase1.png)

The image shows the Pareto plot for the 24 trials that were run. The trials that lie on the Pareto frontier
are colored red and those that do not are colored blue. The trial run with the vLLM defaults has a black box around it. Note that there are actually three trials that lie on the Pareto frontier, but two of them have the exact same failure rate and repetition penalty (0.0, 1.03).

There are a number of trials that had a failure rate of 0.0%. While we optimized to minimize both the failure rate and the repetition penalty, the failure rate is of much more importance to us.

The top candidates for phase 2's longer testing are shown in the table. The top two, in bold, are along the Pareto frontier while the vLLM default, in italics, is at the bottom for comparison purposes.

| Failure Rate (%) | Repetition Penalty |   min_p  | Temperature |
|:----------------:|:------------------:|:--------:|:-----------:|
|     **0.000**    |    **1.03**        | **0.00** |   **0.80**  |
|     **0.000**    |    **1.03**        | **0.03** |   **0.80**  |
|       0.000      |      1.04          |   0.03   |     0.80    |
|       0.000      |      1.05          |   0.06   |     0.85    |
|       0.000      |      1.05          |   0.07   |     0.90    |
|       0.000      |      1.06          |   0.01   |     0.90    |
|       0.000      |      1.07          |   0.02   |     0.90    |
|       0.000      |      1.07          |   0.10   |     0.85    |
|       0.000      |      1.08          |   0.10   |     0.75    |
|       0.000      |      1.09          |   0.05   |     0.85    |
|       0.000      |      1.1           |   0.03   |     0.70    |
|      *0.500*     |     *1.00*         |  *0.00*  |     0.70    |


### Phase 2 Results

Extended testing showed that only six parameter combinations, out of the 11 candidates, maintained their 0% failure rate over the longer testing period. The vLLM defaults showed a slightly increased failure rate of 0.722%. 

![Pareto Surface for longer runs](images/int4_pareto_plot_phase2.png)

Here are the successful configurations, with our recommended settings in bold. Again, we include vLLM's default parameters in italics at the bottom:

| Failure Rate (%) | Repetition Penalty |   min_p  | Temperature |
|:----------------:|:------------------:|:--------:|:-----------:|
|    **0.000**     |     **1.03**       | **0.00** |  **0.80**   |
|      0.000       |       1.05         |   0.07   |    0.90     |
|      0.000       |       1.06         |   0.01   |    0.90     |
|      0.000       |       1.08         |   0.10   |    0.75     |
|      0.000       |       1.09         |   0.05   |    0.85     |
|      0.000       |       1.10         |   0.03   |    0.70     |
|     *0.722*      |      *1.00*        |  *0.00*  |   *0.70*    |

## Implementation Recommendations

Based on our findings, we recommend a two-step approach:

1. **Immediate Action**: Set the `max_tokens` request parameter to 8,192 tokens to prevent system crashes
2. **Parameter Optimization**: Use temperature=0.8 and repetition_penalty=1.03, which achieved optimal results with minimal parameter adjustment

## Future Steps

There are two next steps that should probably be taken. The first is to calculate how the LLM performs with these suggested request parameters on the various benchmarks to ensure that there isn't a noticable drop in performance. This is would be similar to what Neural Magic did with their [quantization study](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/).

Redo this experiment using an 8-bit quantized model instead of the 4-bit to see if there is a noticable change in performance caused by the lower quantization.