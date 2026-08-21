"""RAGTruth finetuned-detector scorer (Niu et al., NAACL 2024) — the paper's recommended
hallucination scorer (finetuned Llama-2-13B, ~0.80 response-level F1), replacing the weak
prompt-judge (~0.40 F1). Loads the base model + the LoRA adapter we trained on RAGTruth-train,
formats each (task, reference, response) with RAGTruth's EXACT templates, wraps in Llama-2
`[INST] ... [/INST]` (the format it was trained/evaluated with — raw, NOT a chat template),
generates the `{"hallucination list": [...]}` JSON, and returns hallucinated = (list non-empty).

BATCHED generation (left-padded) so it runs in parallel, not item-by-item — required to keep the
whole per-iteration eval suite under 30 min. `max_new_tokens` is capped (the JSON is short).

Env: RAGTRUTH_DETECTOR=<adapter_dir> (enables it), RAGTRUTH_DETECTOR_BASE (default Llama-2-13b-hf),
RAGTRUTH_DETECTOR_BATCH (default 16), RAGTRUTH_DETECTOR_MAXNEW (default 256).
"""
from __future__ import annotations

import json
import os
import re

# Verbatim RAGTruth templates (baseline/dataset.py TEMPLATES) — must match training byte-for-byte.
_HALLU_TASK = (
    "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
    "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
    "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
    "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
    "Output:"
)
TEMPLATES = {
    "QA": ("Below is a question:\n{question}\n\nBelow are related passages:\n{reference}\n\n"
           "Below is an answer:\n{response}\n\n" + _HALLU_TASK),
    "Summary": ("Below is the original news:\n{reference}\n\nBelow is a summary of the news:\n{response}\n" + _HALLU_TASK),
    "Data2txt": ("Below is a structured data in the JSON format:\n{reference}\n\n"
                 "Below is an overview article written in accordance with the structured data:\n{response}\n\n" + _HALLU_TASK),
}
B_INST, E_INST = "[INST]", "[/INST]"


class RagtruthDetector:
    def __init__(self, adapter_dir: str, base: str | None = None,
                 max_new_tokens: int | None = None, batch_size: int | None = None):
        self.adapter_dir = adapter_dir
        self.base = base or os.getenv("RAGTRUTH_DETECTOR_BASE", "meta-llama/Llama-2-13b-hf")
        self.max_new = int(max_new_tokens or os.getenv("RAGTRUTH_DETECTOR_MAXNEW", "256"))
        self.batch_size = int(batch_size or os.getenv("RAGTRUTH_DETECTOR_BATCH", "8"))
        self._holder: dict[str, object] = {}

    def _model(self):
        if "m" not in self._holder:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            tok = AutoTokenizer.from_pretrained(self.base)
            tok.padding_side = "left"          # left-pad so batched continuations align at -1
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(self.base, torch_dtype=torch.bfloat16).to("cuda")
            model = PeftModel.from_pretrained(base, self.adapter_dir).eval()
            self._holder.update(m=model, t=tok, torch=torch)
        return self._holder["m"], self._holder["t"], self._holder["torch"]

    def _prompt(self, task_type: str, question: str, reference: str, response: str) -> str:
        t = TEMPLATES.get(task_type, TEMPLATES["Summary"])
        filled = (t.format(question=question, reference=reference, response=response)
                  if task_type == "QA" else t.format(reference=reference, response=response))
        return f"{B_INST} {filled.strip()} {E_INST}"

    def hallucinated_batch(self, triples: list[tuple]) -> list[bool]:
        """triples = list of (task_type, question, reference, response). BATCHED generation."""
        model, tok, torch = self._model()
        prompts = [self._prompt(t[0], t[1] or "", t[2] or "", t[3] or "") for t in triples]
        out = [False] * len(prompts)
        for b in range(0, len(prompts), self.batch_size):
            chunk = prompts[b:b + self.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j in range(new.shape[0]):
                out[b + j] = _parse_hallucinated(tok.decode(new[j], skip_special_tokens=True))
        return out

    def hallucinated(self, task_type: str, question: str, reference: str, response: str) -> bool:
        return self.hallucinated_batch([(task_type, question, reference, response)])[0]


def _parse_hallucinated(gen: str) -> bool:
    """True iff the detector emits a non-empty hallucination list (response-level)."""
    m = re.search(r'"hallucination list"\s*:\s*\[(.*?)\]', gen, re.S)
    if m:
        return len(m.group(1).strip()) > 0
    return False   # no parseable verdict -> faithful (don't fabricate a hallucination)
