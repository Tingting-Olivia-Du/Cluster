# scripts desciption

## for hidden states

- entropy and hidden states.ipynb

generate model's(deepseek7b) answer to gsm8k samples, with logits, entropy, infomation, hidden states vector

- get error token index.ipynb

call gpt4.1 to locate the first sentence that produces error

- hidden states and turning points.ipynb

use `from scipy.signal import find_peaks` to find semantic boundaries

- hidden_vector_semantic_boundary_vs_error_index.ipynb

plot boundary and error index in comparison

## for entropy only 

same way of generation model's answer to gsm8k and locate error index

- entropy_peak .ipynb

plot entropy peak and error index