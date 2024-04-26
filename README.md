# on-the-fly-tokenization-profiling

## Num_workers

setup:

```
dataset_name = "roneneldan/TinyStories"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
seq_len = 1024
batch_size = 1024
num_batches = 100
```
 
|                         | Num_workers  | Time Per Batch(s)  |
|-------------------------|--------------|--------------------|
| pre-tokenization        | 1            |   0.8321           |
| pre-tokenization        | 8            |   0.3444           |
| on-the-fly tokenization | 1            |   2.0477           |
| on-the-fly tokenization | 8            |   0.2946           |

The reason that on-the-fly tokenization is faster when `num_workers` is large is that on-the-fly tokenization reads fewer bytes from disk compared to pre-tokenization, given that one word usually corresponds to multiple (~2) tokens.
