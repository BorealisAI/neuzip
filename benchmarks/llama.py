import csv
import logging
import os
import time

import datasets
import torch
import transformers

import neuzips

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

model_id = "meta-llama/llama-2-7b-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n".join(dataset["text"])
encodings = tokenizer(text, return_tensors="pt")


def eval_perplexity(model, tokenizer):
    max_length = getattr(model.config, "n_positions", 1024)
    stride = max_length // 2
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.stack(nlls)


def benchmark_llama(manager: neuzips.Manager, model: torch.nn.Module):
    comp_bytes = 0
    raw_bytes = 0

    comp_time = 0
    decomp_time = 0
    max_absolute_error = 0

    for p in model.parameters():
        _comp_time = time.time()
        e, f = manager.split_and_compress(p)
        comp_time += time.time() - _comp_time

        _decomp_time = time.time()
        result = torch.empty_like(p)
        manager.decompress_and_merge(e, f, result)
        decomp_time += time.time() - _decomp_time

        comp_bytes += e.nbytes + f.nbytes
        raw_bytes += p.data.nbytes
        max_absolute_error = max((result - p).abs().max().item(), max_absolute_error)
        p.data.copy_(result)

    log_probs = eval_perplexity(model, tokenizer)
    byte_level_perplexity = torch.exp((log_probs / len(text.encode("utf-8"))).sum()).item()
    word_level_perplexity = torch.exp((log_probs / len(text.split())).sum()).item()
    token_level_perplexity = torch.exp((log_probs / encodings.input_ids.size(1)).sum()).item()

    return {
        "ratio": comp_bytes / raw_bytes,
        "compression_time": comp_time,
        "decompression_time": decomp_time,
        "byte_level_perplexity": byte_level_perplexity,
        "word_level_perplexity": word_level_perplexity,
        "token_level_perplexity": token_level_perplexity,
        "max_absolute_error": max_absolute_error,
    }


def update_csv_row(filename, new_row):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
    else:
        rows = [[
            "precision",
            "ratio",
            "compression_time",
            "decompression_time",
            "byte_level_perplexity",
            "word_level_perplexity",
            "token_level_perplexity",
            "max_absolute_error",
        ]]

    rows.append(new_row)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    filename = ".logs/benchmarks_llama.csv"
    if os.path.exists(filename):
        logger.warning(f"Removing existing file {filename}")
        os.remove(filename)

    for precision in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        manager = neuzips.Manager(precision=precision)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
        logger.info(f"Running benchmark with precision {precision}")
        result = benchmark_llama(manager, model)
        update_csv_row(filename, [precision, *result.values()])
        del model, manager
