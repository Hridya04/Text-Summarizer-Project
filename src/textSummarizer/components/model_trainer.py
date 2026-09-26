import os
import torch
from transformers import (
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    GenerationConfig,
)
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig


# Keys that belong in GenerationConfig, not in model.config
_GEN_KEYS = ["max_length", "min_length", "num_beams",
             "length_penalty", "early_stopping", "forced_eos_token_id"]


def _fix_generation_config(model):
    """Move generation params out of model.config into model.generation_config."""
    if model is None:
        return

    model.generation_config = GenerationConfig(
        max_length=64,
        min_length=10,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    for key in _GEN_KEYS:
        if hasattr(model.config, key):
            delattr(model.config, key)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ModelTrainer] Using device: {device}")
        if device == "cuda":
            print(f"[ModelTrainer] GPU: {torch.cuda.get_device_name(0)}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            gradient_checkpointing=False,
            fp16=True,
            dataloader_num_workers=0,
            optim="adamw_torch",
            max_grad_norm=1.0,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        trainer.train()

        # Final safety: force the correct generation config before saving
        _fix_generation_config(model)

        # Save model + tokenizer
        model.save_pretrained(
            os.path.join(self.config.root_dir, "t5-samsum-model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )