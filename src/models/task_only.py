from collections import defaultdict

from data.data_utils import encode_example

# predicts each task's hallucination rate on the training split, ignoring the prompt and answer


class TaskOnlyBaseline:
    def fit(self, train_samples, tokenizer=None):
        response_counts = defaultdict(int)
        hallucinated_responses = defaultdict(int)
        token_counts = defaultdict(int)
        hallucinated_tokens = defaultdict(int)

        for sample in train_samples:
            task = sample["task_type"]
            response_counts[task] += 1
            hallucinated_responses[task] += int(len(sample["labels"]) > 0)  # any annotated span

            # token rates use the same labels the detector trains on
            if tokenizer is not None:
                encoding = encode_example(tokenizer, sample["prompt"], sample["answer"], sample["labels"])
                labels = [label for label in encoding["labels"] if label != -100]
                token_counts[task] += len(labels)
                hallucinated_tokens[task] += sum(labels)

        self.response_rates = {
            task: hallucinated_responses[task] / response_counts[task]
            for task in response_counts
        }
        self.token_rates = {
            task: hallucinated_tokens[task] / token_counts[task]
            for task in token_counts
        }

        return self

    def token_probability(self, task):
        # bias expert for poe_loss: P(token hallucinated | task)
        return self.token_rates[task]

    def predict(self, samples):
        # format data/evaluate.py expects from answer-level models
        return {
            sample["id"]: {"response_score": self.response_rates[sample["task_type"]]}
            for sample in samples
        }
