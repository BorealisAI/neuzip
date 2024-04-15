# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gc
import json
import logging
import os
import time

import datasets
import torch
import transformers
import wandb
from tqdm.auto import tqdm

import neuzip

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def eval_perplexity(model, tokenizer, dataset):
    text = "\n".join(dataset["text"])
    full_input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

    max_length = 1024
    stride = max_length
    seq_len = full_input_ids.size(1)

    nbytes = len(text.encode("utf-8"))
    nwords = len(text.split())
    ntokens = full_input_ids.size(1)

    time_taken = 0.0
    nlls = []
    prev_end_loc = 0
    niterations = 0
    for begin_loc in tqdm(
        range(0, seq_len, stride),
        dynamic_ncols=True,
        desc="Evaluating",
    ):
        niterations += 1
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = full_input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            model.eval()
            start = time.perf_counter()
            outputs = model(input_ids, labels=target_ids, use_cache=False)
            time_taken += time.perf_counter() - start
            neg_log_likelihood = outputs.loss * trg_len

        assert torch.isfinite(neg_log_likelihood).all().item()
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    log_probs = torch.stack(nlls)
    token_level_perplexity = torch.exp((log_probs / ntokens).sum()).item()
    word_level_perplexity = torch.exp((log_probs / nwords).sum()).item()
    byte_level_perplexity = torch.exp((log_probs / nbytes).sum()).item()

    results = {
        "byte_level_perplexity": byte_level_perplexity,
        "word_level_perplexity": word_level_perplexity,
        "token_level_perplexity": token_level_perplexity,
        "time_taken": time_taken,
        "speed": niterations / time_taken,
    }

    logger.info(f"Token level perplexity: {token_level_perplexity}")
    logger.info(f"Word level perplexity: {word_level_perplexity}")
    logger.info(f"Byte level perplexity: {byte_level_perplexity}")

    return {"results": results}


def main(args):
    if args.memory_efficient == "neuzip":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        manager = neuzip.Manager(
            precision=args.precision,
            normalizer_size=args.block_size if args.precision < 7 else 0,
            algorithm=neuzip.Algorithm.ans,
        )

        model = manager.convert(model).cuda()
        name = f"p{args.precision}-b{args.block_size}"

    elif args.memory_efficient == "qlora":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=args.quant_type in ["nf4", "fp4"],
            load_in_8bit=args.quant_type in ["int8"],
            bnb_4bit_quant_type=args.quant_type
            if args.quant_type in ["nf4", "fp4"]
            else "nf4",
            bnb_4bit_use_double_quant=(
                args.bnb_4bit_use_double_quant
                if args.quant_type in ["nf4", "fp4"]
                else False
            ),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map="cuda:0",
        )

        name = f"qlora-{args.quant_type}-d{args.bnb_4bit_use_double_quant}"
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )
        name = "none"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    wandb.init(project="neuzips", name=name)

    run_config = vars(args)
    model_size = sum(p.numel() for p in model.parameters())
    run_config["model_size"] = model_size
    wandb.config.update(run_config)

    convert_memory = torch.cuda.max_memory_allocated()
    logger.info(f"Conversion memory: {convert_memory / 2**30} GiB")
    wandb.log({"convert_memory": convert_memory / 2**30})

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_memory = torch.cuda.memory_allocated()
    logger.info(f"Start memory: {start_memory / 2**30} GiB")
    wandb.log({"start_memory": start_memory / 2**30})

    results = eval_perplexity(model, tokenizer, dataset)

    wandb.log(results)

    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    logger.info(f"End memory: {end_memory / 2**30} GiB")
    logger.info(f"Peak memory: {peak_memory / 2**30} GiB")
    wandb.log({"end_memory": end_memory / 2**30, "peak_memory": peak_memory / 2**30})

    config = vars(args)
    config.update(
        {
            "start_memory": start_memory,
            "end_memory": end_memory,
            "peak_memory": peak_memory,
            "convert_memory": convert_memory,
        }
    )

    return {**config, **results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-id", type=str, default="meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=f".logs/{os.path.basename(__file__)}.json"
    )
    parser.add_argument("-p", "--precision", type=int, default=7)
    parser.add_argument("-b", "--block-size", type=int, default=512)
    parser.add_argument(
        "--memory_efficient", choices=["qlora", "neuzip", "none"], default="none"
    )
    parser.add_argument("--quant_type", choices=["nf4", "fp4", "int8"], default="nf4")
    parser.add_argument("-dq", "--bnb_4bit_use_double_quant", action="store_true")

    args = parser.parse_args()
    output_file = args.output

    new_results = main(args)
    timestamp = time.time_ns()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    results[timestamp] = new_results

    print(json.dumps(new_results, indent=4, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
