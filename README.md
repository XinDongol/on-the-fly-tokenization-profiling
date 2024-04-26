# on-the-fly-tokenization-profiling

## Num_Workers

setup:

```
dataset_name = "roneneldan/TinyStories"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
seq_len = 1024
batch_size = 1024
```
 
|                         | Num_workers  | Time(s)  |
|-------------------------|--------------|----------|
| pre-tokenization        | 1            |   0.8321 |
| pre-tokenization        | 8            |   0.3444 |
| on-the-fly tokenization | 1            |   2.0477 |
| on-the-fly tokenization | 8            |   0.2946 |