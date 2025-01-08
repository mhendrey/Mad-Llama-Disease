# Mad-Llama-Disease
Investigating Llama-3.1's penchant for failing to generate the end-of-string token and thus generating gibberish until the context window has been filled.

## Introduction

Large Language Models (LLMs) have become critical components in many organizations' tech stacks, powering everything from chatbots to automated workflows. However, running these models in production comes with its own set of challenges. In this blog post, I'll share our experience addressing a specific issue we've encountered with the Llama-3.1 model family: the occasional failure to generate an end-of-string (eos) token.

## The Problem

When running Llama-3.1 models in production (both 8B and 70B variants), we've observed an interesting failure mode where the model occasionally fails to generate an end-of-string token. This leads to the model continuing to generate tokens until it hits the maximum context window size of 128K tokens. This behavior creates two significant problems:

1. **Performance Impact**: Generating 128K tokens takes approximately 30 minutes, creating an unacceptable user experience.
2. **Stability Issues**: The extended generation fills up the GPU VRAM with the KV cache, eventually causing the container to crash. We observe this happening 7-10 times daily.

## Initial Mitigation

The most straightforward solution to this problem is setting the `max_tokens` parameter in the request. By limiting generation to something like 8,192 tokens, we can prevent the worst impacts of this failure mode. While this is a crucial first step that we recommend implementing immediately, we wanted to explore whether we could reduce the probability of this failure occurring in the first place.

## Experimental Setup

We designed an experiment to investigate how three key request sampling parameters affect the failure rate:

1. **temperature**: Controls sampling randomness (higher values = more random)
2. **min_p**: Sets the minimum probability threshold for token consideration
3. **repetition_penalty**: Penalizes tokens based on their previous appearances

### Technical Details

- Model: 4-bit quantized Llama-3.1-8B (matching our production environment)
- Request Rate: 60 requests/minute
- Trial Size: 600 requests per parameter combination
- Trial Structure: 5 independent runs of 120 requests each (to protect GPU VRAM)
- Optimization Framework: Optuna for multi-objective optimization

### Optimization Goals

We defined two objectives for our optimization:

1. **Primary**: Minimize the failure rate (percentage of requests hitting max tokens)
2. **Secondary**: Minimize deviation from the default repetition_penalty (1.0)

The second objective reflects our preference to keep parameters close to their defaults unless we see significant benefits from changing them.

## Experimental Implementation

Each trial in our experiment tests a specific combination of temperature, min_p, and repetition_penalty parameters. To simulate real-world conditions, we generate requests at a rate of 60 per minute using exponentially distributed arrival times. This approach helps us understand how the model behaves under production-like conditions.

To properly stress test the model's behavior, we carefully designed our prompts to encourage longer responses. We used a system prompt that frames the model as a student taking an exam, specifically requesting well-reasoned answers of at least 3,000 words. This approach helps us better evaluate the model's tendency to generate endlessly under heavy loads, as shorter responses might not sufficiently exercise the failure mode we're investigating.

A key aspect of our implementation is how we evaluate the failure rate. For each request in a trial, we track whether it generates the maximum allowed number of tokens (8,192 in our case). Here's a snippet showing how we calculate the failure rate for a single trial:

```python
n_failed = 0
if completion_tokens == self.max_completion_tokens:
    n_failed += 1
    print(f"  Failed {request_id=}")

failure_rate = 100.0 * n_failed / self.n_requests
```

To protect against GPU memory issues, we split each trial of 600 requests into 5 independent runs of 120 requests each. The final failure rate for a parameter combination is the average across these runs, providing a more robust estimate of the true failure rate.

We utilize the `optuna.sampler.TPESampler` for a total of 24 trials. The first 10 trials randomly select parameter sets which leaves the remaining 14 trials to leverage the TPE sampler.

## Initial Results
Let's see the Pareto Surface plot to see how we did. The trials that lie on the Pareto Surface are colored red. A trial run with the vLLM default sampling parameters (temperature=0.7, min_p=0.0, repetition_penalty=0.0) has a black box around it. Note that there are actually three trials that lie on the Pareto surface, but two of them have the exact same failure rate and repetition penalty (0.0, 1.03).

![Pareto Surface Plot](images/int4_pareto_plot_phase1.png)

As we see in the figure, there are a number of trials that had a failure rate of 0.0%. While we optimized to minimize both the failure rate and the repetition_penalty, let's take a look at the trials that had a failure rate of 0.0. These would make good candidates for further testing to see if the zero failure rates hold up under longer testing. In addition, we also list Trial 22 (italicized) which matches the default vLLM parameters that are currently used in our production system and included here for comparison. Trial 18 and 16 (bold) lie on the Pareto surface in the figure above.

| Trial  | Temperature |  min_p  | repetition_penalty | Failure Rate (%) |
|:------:|:-----------:|:-------:|:------------------:|:----------------:|
| **18** |  **0.8**    | **0.0** |  **1.03**          |      **0.0**     |
| **16** |  **0.8**    | **0.03**|  **1.03**          |      **0.0**     |
|  20    |    0.8      |   0.03  |    1.04            |        0.0       |
|  4     |    0.85     |   0.06  |    1.05            |        0.0       |
|  15    |    0.9      |   0.07  |    1.05            |        0.0       |
|  8     |    0.9      |   0.01  |    1.06            |        0.0       |
|  9     |    0.9      |   0.02  |    1.07            |        0.0       |
|  1     |    0.85     |   0.1   |    1.07            |        0.0       |
|  14    |    0.75     |   0.1   |    1.08            |        0.0       |
|  12    |    0.85     |   0.05  |    1.09            |        0.0       |
|  7     |    0.7      |   0.03  |    1.10            |        0.0       |
| *22*   |   *0.7*     |  *0.0*  |   *1.00*           |       *0.5*      |

## Phase 2
For the second phase of this work, we do a longer run on the parameters listed in the table above. For each configuration, we do 1800 requests (~30 minutes at our rate of 60 requests / minute). For these longer runs, we see that some of the configurations' failure rates increases above 0.0. 

![Pareto Surface for longer runs](images/int4_pareto_plot_phase2.png)

This leaves us with the following parameter values that never encountered our failure to generate a stop token as shown in the table below. Again, we include the vLLM default (italics) values which has a failure rate of 0.72%.  Our recommend settings are the top choice (bold) since it needs just a small repetition penalty to stop the failures from occurring.

| Temperature |  min_p  | repetition_penalty | Failure Rate (%) |
|:-----------:|:-------:|:------------------:|:----------------:|
|   **0.8**   | **0.00** |  **1.03**         |   **0.00**       |
|     0.9     |  0.07   |    1.05            |     0.00         |
|     0.9     |  0.01   |    1.06            |     0.00         |
|     0.75    |  0.10   |    1.08            |     0.00         |
|     0.85    |  0.05   |    1.09            |     0.00         |
|     0.70    |  0.03   |    1.10            |     0.00         |
|    *0.70*   | *0.00*  |   *1.00*           |    *0.72*        |


## Implementation Recommendations

Based on our initial findings, we recommend a two-pronged approach:

1. **Immediate Action**: Implement max_tokens limit of 8,192 tokens to prevent system crashes
2. **Parameter Optimization**: Raise the temperature to 0.8 and apply a small repetition penalty of 1.03

## Future Work

This experiment focused on the 4-bit quantized Llama-3.1-8B model. It would interesting to see if similar behavior is observed when using an 8-bit quantized model. 