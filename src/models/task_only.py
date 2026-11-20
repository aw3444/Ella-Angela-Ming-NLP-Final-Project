from collections import defaultdict


class TaskOnlyBaseline:
    def fit(self, train_dataset):
        token_counts = defaultdict(int)
        hallucinated_counts = defaultdict(int)
        response_counts = defaultdict(int)
        hallucinated_responses = defaultdict(int)

        # Calculate hallucination rates for each task
        for example in train_dataset:
            task = example["task_type"]
            labels = [label for label in example["labels"] if label != -100]

            if not labels:
                continue

            token_counts[task] += len(labels)
            hallucinated_counts[task] += sum(labels)
            response_counts[task] += 1
            hallucinated_responses[task] += int(1 in labels)

        # Store token-level and response-level probabilities
        self.token_rates = {
            task: hallucinated_counts[task] / token_counts[task]
            for task in token_counts
        }

        self.response_rates = {
            task: hallucinated_responses[task] / response_counts[task]
            for task in response_counts
        }

        return self

    def token_probability(self, task):
        return self.token_rates[task]

    def predict_proba(self, dataset):
        # Predict answer-level hallucination probabilities
        return [
            self.response_rates[example["task_type"]]
            for example in dataset
        ]

    def predict(self, dataset, threshold=0.5):
        # Predict token labels based only on task category
        predictions = []

        for example in dataset:
            task = example["task_type"]
            prediction = int(self.token_rates[task] >= threshold)

            predictions.append([
                prediction if label != -100 else -100
                for label in example["labels"]
            ])

        return predictions
