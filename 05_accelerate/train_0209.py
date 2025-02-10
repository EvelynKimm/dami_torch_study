import math
import os
import random
import time
from argparse import ArgumentParser, Namespace
from math import exp

import torch
from accelerate import Accelerator
from bitsandbytes.optim import Adam8bit
from datasets import Dataset, load_dataset
from loguru import logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

argparser = ArgumentParser("Train script for Hanja restoration")
argparser.add_argument("--model-uri", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path.")
argparser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
argparser.add_argument("--tensorboard-dir", type=str, default="tensorboard")
argparser.add_argument("--log-dir", type=str, default="logs")
argparser.add_argument("--run-name", type=str, required=True)
argparser.add_argument("--epochs", type=int, default=6)
argparser.add_argument("--max-lr", type=float, default=1e-5)
argparser.add_argument("--logging-interval", type=int, default=10)
argparser.add_argument("--eval-interval", type=int, default=100)
argparser.add_argument("--model-save-interval", type=int, default=500)
argparser.add_argument("--gradient-accumulation-steps", type=int, default=1)
argparser.add_argument("--weight-decay", type=float, default=1e-3)
argparser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
argparser.add_argument("--batch-size", type=int, default=32)


def preprocess_dataset(dataset):
    processed_data = []
    for example in dataset:
        if "conversations" in example and isinstance(example["conversations"], list):
            human_text = next((item["value"] for item in example["conversations"] if item["from"] == "human"), "")
            gpt_text = next((item["value"] for item in example["conversations"] if item["from"] == "gpt"), "")
            if human_text and gpt_text:
                text = f"Human: {human_text}\nGPT: {gpt_text}"
                processed_data.append({"text": text})
    print(f"✔️ Processed dataset size: {len(processed_data)}")
    return Dataset.from_list(processed_data) if processed_data else Dataset.from_dict({"text": []})


def load_dataset_and_split(
    dataset_name="coastral/korean-writing-style-instruct", train_size=10_000, val_size=2_000, test_size=2_000, seed=42
):
    dataset = load_dataset(dataset_name, cache_dir="./evelyn_cache")
    print(dataset)
    data = dataset["train"].shuffle(seed=seed)
    data = preprocess_dataset(data)

    total_size = min(len(data), train_size + val_size + test_size)
    sampled_indices = random.sample(range(len(data)), total_size)
    sampled_data = data.select(sampled_indices)

    data_train = sampled_data.select(range(min(train_size, total_size)))
    data_val = sampled_data.select(range(min(train_size, total_size), min(train_size + val_size, total_size)))
    data_test = sampled_data.select(range(min(train_size + val_size, total_size), total_size))

    return {"train": data_train, "val": data_val, "test": data_test}


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    accelerator: Accelerator,
    tb_writer: SummaryWriter,
    args: Namespace,
):
    step = 0

    model.eval()
    valid_loss, valid_ppl = evaluate(model, valid_loader, accelerator)
    logger.info(f"[Valid] Before Training - loss: {valid_loss:.4f}, PPL: {valid_ppl:.4f}")
    tb_writer.add_scalar("Loss/valid", valid_loss, step)
    tb_writer.add_scalar("Perplexity/valid", valid_ppl, step)
    model.train()

    for epoch in range(args.epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                input_ids = batch["input_ids"]
                labels = batch["labels"]
                # logger.info(f"input_ids: {input_ids}, labels: {labels}")

                with accelerator.autocast():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

            step += 1

            if step % args.logging_interval == 0:
                loss_val = loss.item()
                ppl = exp(loss_val)
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"[Train] Epoch {epoch}, Step {step} - loss: {loss_val:.4f}, PPL: {ppl:.4f}, lr: {lr:.4e}")
                tb_writer.add_scalar("Loss/train", loss_val, step)
                tb_writer.add_scalar("Perplexity/train", ppl, step)
                tb_writer.add_scalar("LearningRate", lr, step)

            if step % args.eval_interval == 0:
                model.eval()
                valid_loss, valid_ppl = evaluate(model, valid_loader, accelerator)
                logger.info(f"[Valid] Epoch {epoch}, Step {step} - loss: {valid_loss:.4f}, PPL: {valid_ppl:.4f}")
                tb_writer.add_scalar("Loss/valid", valid_loss, step)
                tb_writer.add_scalar("Perplexity/valid", valid_ppl, step)
                model.train()

            if step % args.model_save_interval == 0:
                logger.info("Saving model checkpoint...")
                ckpt_dir = os.path.join(args.checkpoint_dir, f"step-{step}")
                model.save_pretrained(
                    ckpt_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                logger.info(f"Model checkpoint saved at {ckpt_dir}")


def evaluate(model: PreTrainedModel, valid_loader: DataLoader, accelerator: Accelerator) -> tuple:

    model.eval()
    losses = []

    for batch in tqdm(valid_loader, disable=not accelerator.is_main_process, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            batch_size = batch["input_ids"].shape[0]
            # 배치 당 loss를 배치 크기만큼 반복하여 각 샘플의 loss처럼 처리함
            losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)

    return eval_loss.item(), perplexity


def main():
    args = argparser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.tensorboard_dir = os.path.join(args.tensorboard_dir, args.run_name)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    logger.add(os.path.join(args.log_dir, "train.log"), level="INFO")
    tb_writer = SummaryWriter(args.tensorboard_dir)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    logger.info(f"Process index: {accelerator.process_index}")
    logger.info(f"Current GPU device: {torch.cuda.current_device()}")

    dataset_split = load_dataset_and_split()

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_uri)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            args.model_uri,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
        # labels는 input_ids 복사
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        tokenized_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in tokenized_inputs["labels"]
        ]
        return tokenized_inputs

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset_split["train"] = dataset_split["train"].map(tokenize_function, batched=True, remove_columns=["text"])
    dataset_split["val"] = dataset_split["val"].map(tokenize_function, batched=True, remove_columns=["text"])
    dataset_split["test"] = dataset_split["test"].map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataloader = DataLoader(dataset_split["train"], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(dataset_split["val"], batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset_split["test"], batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)

    total_train_steps = len(train_dataloader) * args.epochs

    decay_parameters = [name for name in get_parameter_names(model, [torch.nn.LayerNorm]) if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = Adam8bit(optimizer_grouped_parameters, lr=args.max_lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_train_steps * args.lr_warmup_ratio), total_train_steps)

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    train(model, train_dataloader, val_dataloader, optimizer, scheduler, accelerator, tb_writer, args)

    total_time = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated() / (1024**2)

    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Peak GPU memory usage: {max_memory:.2f} MB")

    test_dataloader = accelerator.prepare(test_dataloader)
    test_loss, test_ppl = evaluate(model, test_dataloader, accelerator)

    logger.info(f"[Test] - loss: {test_loss:.4f}, PPL: {test_ppl:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
