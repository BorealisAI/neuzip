# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gc
import json
import logging
import math
import os
import time

import accelerate
import datasets
import evaluate
import torch
import transformers
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import neuzip

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def eval_perplexity(dataloader, model):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    dataset = [batch.to(torch.cuda.current_device()) for batch in dataloader]
    time_taken = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(dataset, desc="Evaluating"):
            start = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            time_taken += time.perf_counter() - start
            total_samples += 1

    loss = total_loss / total_samples
    perplexity = math.exp(loss)

    results = {
        "loss": loss,
        "token_level_perplexity": perplexity,
        "time_taken": time_taken,
        "speed": total_samples / time_taken,
    }
    logger.info(f"Perplexity: {perplexity}")
    logger.info(f"Loss: {loss}")
    return {"results": results}


def eval_generation(dataloader, model, tokenizer):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    metric = evaluate.load("sacrebleu")
    model.eval()

    dataset = [batch.to(torch.cuda.current_device()) for batch in dataloader]
    time_taken = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(dataset, desc="Evaluating"):
            start = time.perf_counter()
            labels = batch.pop("labels")
            generated_tokens = model.generate(
                inputs=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=1,
                do_sample=False,
            )
            time_taken += time.perf_counter() - start
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    results["time_taken"] = time_taken
    results["speed"] = len(dataset) / time_taken
    logger.info(f"BLEU: {results['score']}")
    return {"results": results}


def main(args):
    if args.memory_efficient == "neuzip":
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
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
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        bnb_config = accelerate.utils.BnbQuantizationConfig(
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
        model = accelerate.utils.load_and_quantize_model(model, bnb_config)
        name = f"qlora-{args.quant_type}-d{args.bnb_4bit_use_double_quant}"
    else:
        model = model.cuda()
        name = "none"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)

    raw_dataset = datasets.load_dataset("wmt14", "de-en", split="test")

    source_lang = "en"
    target_lang = "de"
    prefix = "translate English to German: "
    padding = False

    def preprocess_function(_examples):
        examples = _examples["translation"]
        inputs = [ex[source_lang] for ex in examples]
        targets = [ex[target_lang] for ex in examples]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(lb if lb != tokenizer.pad_token_id else -100) for lb in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = raw_dataset.column_names

    dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    dataloader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=args.batch_size
    )

    wandb.init(project="neuzips", name=name)

    run_config = vars(args)
    run_config["model_size"] = sum(p.numel() for p in model.parameters())
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

    results = eval_perplexity(dataloader, model)
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
    parser.add_argument("-m", "--model-id", type=str, default="google-t5/t5-large")
    parser.add_argument(
        "-o", "--output", type=str, default=f".logs/{os.path.basename(__file__)}.json"
    )
    parser.add_argument("-p", "--precision", type=int, default=7)
    parser.add_argument("-b", "--block-size", type=int, default=512)
    parser.add_argument(
        "--memory_efficient", choices=["qlora", "neuzip", "none"], default="none"
    )
    parser.add_argument("--batch_size", type=int, default=8)
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
