import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data.data_utils import encode_example
from data.evaluate import evaluate, format_table, predicted_spans, prediction_from_encoding, response_score

# inference only (no fine-tuning); lettucedetect runs this checkpoint as a plain token classifier
# on [CLS] prompt [SEP] answer [SEP], the same input encode_example builds

CHECKPOINT = "KRLabsOrg/lettucedect-base-modernbert-en-v1"  # official id, spelled "lettucedect"

# synthetic example from our proposal; "24 hours" is the hallucination
DEMO_SAMPLE = {
    "id": "demo",
    "task_type": "Data2txt",
    "prompt": "Write an overview of the business using only this data: "
              '{"name": "Joe\'s Pizza", "hours": "11am-10pm"}',
    "answer": "Joe's Pizza is open 24 hours.",
    "labels": [{"start": 20, "end": 28}],
}


def predict_samples(model, tokenizer, samples):
    model.eval()
    predictions = {}

    for index, sample in enumerate(samples, start=1):
        encoding = encode_example(tokenizer, sample["prompt"], sample["answer"], sample["labels"])
        inputs = {
            "input_ids": torch.tensor([encoding["input_ids"]], device=model.device),
            "attention_mask": torch.tensor([encoding["attention_mask"]], device=model.device),
        }

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        # P(hallucinated) at every position; prediction_from_encoding keeps only answer tokens
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
        predictions[sample["id"]] = prediction_from_encoding(encoding, probabilities)

        if index % 100 == 0:
            print(f"{index}/{len(samples)} answers done")

    return predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/processed/ragtruth_data.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model_path", default=CHECKPOINT)
    parser.add_argument("--output_dir", default="results/lettucedetect")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)

    if args.demo:
        prediction = predict_samples(model, tokenizer, [DEMO_SAMPLE])["demo"]
        print("Answer:", DEMO_SAMPLE["answer"])
        print("Token probabilities:", [round(p, 3) for p in prediction["token_probs"]])
        print("Predicted spans:", predicted_spans(prediction, 0.5))
        print("Response score:", round(response_score(prediction), 3))
        return

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    samples = [x for x in data if x["split"] == args.split][:args.limit]
    predictions = predict_samples(model, tokenizer, samples)

    # 0.5 is LettuceDetect's own argmax rule; not tuned on dev, since LettuceDetect
    # was trained on RAGTruth's train split, which includes our dev answers
    results = evaluate(samples, predictions, threshold=0.5)
    print(format_table(results))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # separate names so a --limit run never overwrites full-split results
    run_name = args.split if args.limit is None else f"{args.split}_first{args.limit}"
    (output_dir / f"{run_name}_preds.json").write_text(json.dumps(predictions))
    (output_dir / f"{run_name}_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
