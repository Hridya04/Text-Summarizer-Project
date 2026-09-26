import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # T5 requires the "summarize: " prefix on the input
        input_encodings = self.tokenizer(
            ["summarize: " + d for d in example_batch['dialogue']],
            max_length=256,
            truncation=True,
            padding="max_length",
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=64,
                truncation=True,
                padding="max_length",
            )

        # Replace pad token id with -100 so the loss ignores padding
        labels = target_encodings["input_ids"]
        labels = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels
        ]

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, batched=True
        )
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )