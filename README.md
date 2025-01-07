# Mad-Llama-Disease
Investigating Llama-3.1's penchant for failing to generate the end-of-string token and thus generating gibberish until the context window has been filled.


# Investigating End-of-String Token Generation Failures in Llama-3.1

When running large language models in production, you occasionally encounter surprising failure modes that can impact service reliability. In our production environment, we've been grappling with an interesting issue in the Llama-3.1 model family: occasional failures to generate end-of-string tokens, leading to runaway text generation that continues until hitting the context window limit and often leading to crashing the container running the model.

## The Problem

Our production service runs Llama-3.1 models (8B and 70B) to power both a web-based chatbot and API endpoints for other teams' applications. Several times per day, we observe the model entering a state where it fails to generate an end-of-string token, instead producing continuous text until it exhausts the maximum context length. This not only wastes computational resources but also impacts the user experience and service reliability.

## Experimental Setup

To investigate this issue systematically, we set up an experiment using the 4-bit quantized version of Llama-3.1-8B running on our local infrastructure. The quantization helps us optimize GPU memory usage while still maintaining model functionality for testing purposes.

Our experiment focuses on three key parameters that influence the model's token generation behavior:

1. **Temperature** - Controls sampling randomness, with lower values producing more deterministic outputs
2. **Minimum P (min_p)** - Sets a threshold for token probability relative to the most likely token
3. **Repetition Penalty** - Adjusts token probabilities based on their previous occurrence in the context

We developed a test harness that simulates real-world usage patterns by:
- Sending requests at a rate of 60 per minute
- Running multiple concurrent requests using thread pooling
- Testing various parameter combinations using Optuna for optimization
- Tracking the failure rate for each parameter combination

The test prompts simulate exam responses, intentionally encouraging longer outputs to better observe the failure mode. Each prompt includes a system message that instructs the model to generate comprehensive, well-structured responses of at least 3000 words.

Here's a key section of our experimental setup:

```python
def run_trial(
    self,
    eval_id: int,
    temperature: float,
    min_p: float,
    repetition_penalty: float,
) -> float:
    arrival_times = np.cumsum(
        self.rng.exponential(scale=60.0 / self.request_rate, size=self.n_requests)
    )
    prompts = self.rng.choice(self.prompts, self.n_requests, replace=True)
    
    # Track failures where the model hits max tokens
    n_failed = 0
    with ThreadPoolExecutor(256) as executor:
        # Implementation of concurrent request processing
        # and failure detection
```

## Parameter Ranges Tested

Our experiment explores the following parameter ranges:
- Temperature: 0.7 to 0.9
- min_p: 0.0 to 0.1
- Repetition Penalty: 1.0 to 1.1

[RESULTS SECTION TO BE ADDED]

## Implementation Notes

For those interested in reproducing our results, we're running our quantized model using vLLM with the following configuration:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 \
    --max-model-len 9216 \
    --max-num-seqs 64 \
    --gpu_memory_utilization 0.95 \
    --served-model-name Llama-3.1-8B
```

## Next Steps

Once we have the experimental results, we plan to:
1. Implement parameter guardrails based on the optimal values found
2. Develop monitoring systems to detect this failure mode in production
3. Consider implementing a token budget system to prevent runaway generation

Stay tuned for the complete results of our parameter optimization study and our recommendations for mitigating this issue in production environments.