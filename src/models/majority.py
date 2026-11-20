import numpy as np

class MajorityBaseline:
    def fit(self, train_dataset):
        labels = np.concatenate([
            np.asarray(example["labels"])
            for example in train_dataset
        ])

        labels = labels[labels != -100]

        self.majority_class = int(
            np.bincount(labels, minlength=2).argmax()
        )

        return self

    def predict(self, dataset):
        return [
            [
                self.majority_class if label != -100 else -100
                for label in example["labels"]
            ]
            for example in dataset
        ]