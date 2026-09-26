import os
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from evaluate import load as load_metric
from textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metric_on_test_ds(
        self, dataset, metric, model, tokenizer,
        batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="dialogue", column_summary="summary",
    ):
        article_batches = list(self.generate_batch_sized_chunks(
            dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(
            dataset[column_summary], batch_size))

        for article_batch, target_batch in zip(article_batches, target_batches):
            inputs = tokenizer(
                ["summarize: " + a for a in article_batch],   # T5 prefix
                max_length=256, truncation=True,
                padding="max_length", return_tensors="pt",
            ).to(device)

            summaries = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(
                summaries, skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            metric.add_batch(predictions=decoded, references=target_batch)

        return metric.compute()

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        dataset_samsum_pt = load_from_disk(self.config.data_path)
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load_metric("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt["test"][0:50],   # small subset for speed
            rouge_metric, model, tokenizer,
            batch_size=2, device=device,
        )

        rouge_dict = {rn: score[rn] for rn in rouge_names}
        df = pd.DataFrame(rouge_dict, index=["t5-small"])
        df.to_csv(self.config.metric_file_name, index=False)