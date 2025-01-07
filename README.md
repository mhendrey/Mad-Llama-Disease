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

Our experiment uses a ThreadPoolExecutor to simulate realistic request patterns with exponentially distributed arrival times. We implemented comprehensive error handling and logging to ensure reliable results. The code tracks the number of requests that hit the maximum token limit and calculates the failure rate for each parameter combination.

Here's a key section of our implementation showcasing the parameter ranges we explored:

```python
temperature_min = 0.7
temperature_max = 0.9
min_p_min = 0.0
min_p_max = 0.1
repetition_penalty_min = 1.0
repetition_penalty_max = 1.1
```

## Results

[RESULTS TO BE INSERTED WHEN EXPERIMENT COMPLETES]

Expected data to be included:
- Optimal parameter values discovered
- Failure rate comparison (baseline vs. optimized)
- Trade-offs observed between parameters
- Visualization of the parameter space exploration

## Implementation Recommendations

Based on our initial findings, we recommend a two-pronged approach:

1. **Immediate Action**: Implement max_tokens limit of 8,192 tokens to prevent system crashes
2. **Parameter Optimization**: Apply the discovered optimal parameters once results are available

## Future Work

This experiment focuses on the Llama-3.1 model family, but similar issues might affect other LLMs. Future work could include:

- Extending the experiment to other model families
- Investigating the relationship between model size and failure rates
- Exploring additional sampling parameters
- Analyzing the types of prompts that tend to trigger this behavior

## Conclusion

While we await the final results of our parameter optimization experiment, the investigation has already yielded valuable insights into managing LLMs in production. The combination of setting hard limits through max_tokens while optimizing sampling parameters provides a robust approach to preventing endless generation issues.

Remember that your specific use case might require different parameter values, but this experimental framework provides a solid foundation for finding the optimal configuration for your deployment.
