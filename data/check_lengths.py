"""Count input lengths with the real ModernBERT tokenizer (run where Hugging Face is reachable).

  python check_lengths.py ragtruth_data.json
"""
import json
import sys
from collections import defaultdict

from transformers import AutoTokenizer

from data_utils import MAX_LENGTH

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
samples = json.load(open(sys.argv[1], encoding="utf-8"))
lengths = defaultdict(list)
for x in samples:
    n = len(tok(x["prompt"], x["answer"], add_special_tokens=True)["input_ids"])
    lengths[x["task_type"]].append(n)
    lengths["ALL"].append(n)
for task, ls in sorted(lengths.items()):
    ls.sort()
    print(f"{task:9s} mean {sum(ls)/len(ls):7.1f}  p99 {ls[int(0.99*len(ls))]:5d}  "
          f"max {ls[-1]:5d}  over {MAX_LENGTH}: {sum(l > MAX_LENGTH for l in ls)}")
