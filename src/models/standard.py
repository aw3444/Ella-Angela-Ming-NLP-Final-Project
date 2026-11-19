from transformers import AutoModelForTokenClassification


def build_standard_detector(
    model_name: str = "answerdotai/ModernBERT-base",
):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
        label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
    )
