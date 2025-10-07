"""LoRA fine-tuning script for causal language models on a JSONL dataset."""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class ScriptArguments:
    """Container for argument defaults."""

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    train_file: str = "train.jsonl"
    validation_file: Optional[str] = None
    output_dir: str = "lora-finetuned-model"
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    template: str = "### Input\n{text}\n\n### Response\n{target}"
    input_field: str = "text"
    target_field: str = "target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default=ScriptArguments.model_name_or_path)
    parser.add_argument("--train-file", default=ScriptArguments.train_file)
    parser.add_argument("--validation-file", default=ScriptArguments.validation_file)
    parser.add_argument("--output-dir", default=ScriptArguments.output_dir)
    parser.add_argument("--max-seq-length", type=int, default=ScriptArguments.max_seq_length)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=ScriptArguments.per_device_train_batch_size,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=ScriptArguments.per_device_eval_batch_size,
    )
    parser.add_argument("--learning-rate", type=float, default=ScriptArguments.learning_rate)
    parser.add_argument("--num-train-epochs", type=float, default=ScriptArguments.num_train_epochs)
    parser.add_argument("--lr-scheduler-type", default=ScriptArguments.lr_scheduler_type)
    parser.add_argument("--warmup-ratio", type=float, default=ScriptArguments.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=ScriptArguments.weight_decay)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=ScriptArguments.gradient_accumulation_steps,
    )
    parser.add_argument("--logging-steps", type=int, default=ScriptArguments.logging_steps)
    parser.add_argument("--save-steps", type=int, default=ScriptArguments.save_steps)
    parser.add_argument("--save-total-limit", type=int, default=ScriptArguments.save_total_limit)
    parser.add_argument("--lora-r", type=int, default=ScriptArguments.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=ScriptArguments.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=ScriptArguments.lora_dropout)
    parser.add_argument(
        "--template",
        default=ScriptArguments.template,
        help="Template that combines the input and target text. Must contain {text} and {target} placeholders.",
    )
    parser.add_argument("--input-field", default=ScriptArguments.input_field)
    parser.add_argument("--target-field", default=ScriptArguments.target_field)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for reduced memory usage.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization aware training via bitsandbytes (requires proper installation).",
    )
    parser.add_argument(
        "--bnb-dtype",
        default="bfloat16",
        help="Data type for bitsandbytes quantization (e.g., 'float16', 'bfloat16').",
    )
    return parser.parse_args()


def build_prompt(template: str, text: str, target: str) -> str:
    if "{text}" not in template or "{target}" not in template:
        raise ValueError("Template must include '{text}' and '{target}' placeholders.")
    return template.replace("{text}", text).replace("{target}", target)


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    dataset,
    template: str,
    input_field: str,
    target_field: str,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    def _format_and_tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = [
            build_prompt(template, text, target)
            for text, target in zip(batch[input_field], batch[target_field])
        ]
        tokenized = tokenizer(
            prompts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(_format_and_tokenize, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(__import__("torch"), args.bnb_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto" if quantization_config else None,
        quantization_config=quantization_config,
    )

    if quantization_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    dataset_dict = {"train": args.train_file}
    if args.validation_file:
        dataset_dict["validation"] = args.validation_file

    dataset = load_dataset("json", data_files=dataset_dict)

    tokenized_datasets = {
        split: preprocess_dataset(
            tokenizer,
            dataset[split],
            args.template,
            args.input_field,
            args.target_field,
            args.max_seq_length,
        )
        for split in dataset
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if "validation" in tokenized_datasets else "no",
        fp16=not args.use_4bit,
        bf16=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
