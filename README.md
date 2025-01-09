# Mad Llama Disease: Investigating and Fixing Endless Generation Issues in Llama-3.1

## Introduction

Large Language Models (LLMs) have become critical components in many organizations' tech stacks, powering everything from chatbots to automated workflows. However, running these models in production presents unique challenges. In this blog post, I'll share our experience addressing a specific issue with the Llama-3.1 model family: the occasional failure to generate an end-of-string (EOS) token.

## The Problem

While running Llama-3.1 models in production (both 8B and 70B variants), we've observed an interesting failure mode, which others in the LLM community of observed too, where the model occasionally fails to generate an end-of-string token. The model starts off generating acceptable outputs, but eventually degrades into endless gibberish. This causes the model to continue generating tokens until it reaches the maximum context window size of 128K tokens, resulting in two significant problems:

1. **Performance Impact**: Generating 128K tokens of mostly gibberish takes approximately 30 minutes, creating an unacceptable user experience
2. **Stability Issues**: The extended generation fills up the GPU VRAM with the KV cache, eventually causing the container to crash (occurring 7-10 times daily)

## Initial Mitigation

By [default](https://github.com/vllm-project/vllm/blob/730e9592e97c643474aa44e9d3dbe6f55c4b9ad9/vllm/entrypoints/openai/serving_chat.py#L190), vLLM sets the `max_tokens` to be equal to the remaining tokens in the context window. The simplest solution is to set the `max_tokens` parameter in the request. By limiting generation to 8,192 tokens, we can prevent the worst impacts of this failure mode. While this is a crucial first step that we recommend implementing immediately, we wanted to explore whether we could reduce the probability of this failure through optimized sampling parameters.

This is also inline with the major commercial models as well

* [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table) (8192)
* [Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.5-pro) (8192)
* [OpenAI](https://platform.openai.com/docs/models) (ranges from 4K-16K dpending on the model)

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