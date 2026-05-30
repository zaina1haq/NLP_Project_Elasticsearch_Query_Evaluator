from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
import torch, json, re

app = Flask(__name__)
CORS(app)

# Tasks
with open("json_format/es_eval_test.json", "r") as f:
    TASKS = json.load(f)

# Load model once
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/finetuned_model/lora_adapter",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded!")

# Rubric
RUBRIC = {
    "0": "Completely invalid query — wrong top-level keys, no valid operator, would throw a parse error.",
    "1": "Major errors — recognizable query type but fundamentally broken (wrong field name, missing nested wrapper).",
    "2": "Partially correct — correct outer structure but a required clause is entirely missing.",
    "3": "Mostly correct with minor issues — one small fix needed (wrong operator, off boundary, wrong sort direction).",
    "4": "Fully correct — right clause types, valid fields, correct operators, returns exactly the right documents."
}

SYSTEM = (
    "You are an expert Elasticsearch query evaluator. "
    "Given a task, a reference query, a student submission, and a rubric, "
    "output ONLY a JSON object with exactly two keys: "
    '"score" (integer 0-4) and "rationale" (string citing [Query Structure], '
    '[Field Validity], [Operator Correctness], [Task Alignment]). '
    "Do not output anything outside the JSON object."
)

# Prompt builder
def build_prompt(task_text, reference, submission):
    ref_str = json.dumps(reference, indent=2)
    sub_str = json.dumps(submission, indent=2)
    rubric_str = "\n".join(f"  {k}: {v}" for k, v in RUBRIC.items())
    return (
        f"<s>[INST] {SYSTEM}\n\n"
        f"### Task\n{task_text}\n\n"
        f"### Reference query (correct answer)\n```json\n{ref_str}\n```\n\n"
        f"### Student submission (to evaluate)\n```json\n{sub_str}\n```\n\n"
        f"### Scoring rubric\n{rubric_str}\n\n"
        f"Evaluate the submission. Respond ONLY with the JSON object, nothing else. [/INST]"
    )

# Output parser
_PROMPT_LEAKAGE = [
    "Evaluate the submission and respond with a JSON object.",
    "Evaluate the submission and respond with a JSON object",
    "respond with a JSON object.",
    "respond with a JSON object",
]

def _clean_rationale(text: str) -> str:
    text = text.replace("\\n", " ")
    for phrase in _PROMPT_LEAKAGE:
        text = text.replace(phrase, "")
    text = re.sub(r" {2,}", " ", text).strip()
    text = re.sub(r"(\.\s*)+$", ".", text).strip()
    return text

def parse_output(text):
    text = text.strip()

    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        blob = text[i:j+1]
                        try:
                            obj = json.loads(blob)
                            score = int(obj.get("score", -1))
                            rationale = obj.get("rationale", "")
                            if 0 <= score <= 4 and rationale:
                                candidates.append((score, _clean_rationale(rationale)))
                        except Exception:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1

    if candidates:
        return candidates[-1]

    try:
        obj = json.loads(text)
        score = int(obj["score"])
        if 0 <= score <= 4:
            return score, _clean_rationale(obj.get("rationale", ""))
    except Exception:
        pass

    matches = list(re.finditer(r'"score"\s*:\s*([0-4])', text))
    if matches:
        last_score = int(matches[-1].group(1))
        nearby = text[matches[-1].start():]
        rm = re.search(r'"rationale"\s*:\s*"([^"]+)"', nearby)
        rationale = rm.group(1) if rm else text
        return last_score, _clean_rationale(rationale)

    return None, _clean_rationale(text)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tasks")
def get_tasks():
    return jsonify(TASKS)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    task_id = int(data.get("task_id", 0))
    submission_text = data.get("submission", "")

    try:
        submission = json.loads(submission_text)
    except Exception:
        return jsonify({"error": "Invalid JSON — please write a valid Elasticsearch query."}), 400

    task = TASKS[task_id]
    prompt = build_prompt(task["task"], task["reference"], submission)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    def stream_response():
        full_text = ""
        for token in streamer:
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        thread.join()
        sep = "=" * 60
        print(f"\n{sep}\nRAW OUTPUT ({len(full_text)} chars):\n{full_text}\n{sep}\n")

        score, rationale = parse_output(full_text)
        print(f"PARSED score={score} | rationale={str(rationale)[:120]}")
        yield f"data: {json.dumps({'done': True, 'score': score, 'rationale': rationale, 'raw': full_text})}\n\n"

    return Response(
        stream_with_context(stream_response()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
