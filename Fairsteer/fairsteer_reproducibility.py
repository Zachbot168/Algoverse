# -*- coding: utf-8 -*-
"""fairsteer-reproducibility.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18ko4zSz56Cgc7nS6kNbohvSQDFF4QuoE

## Installations for some relevant libraries and WinoBias
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install datasets
!pip install --upgrade transformers
!pip install einops
!pip install --upgrade datasets huggingface_hub fsspec

"""Downloading [BBQ](https://huggingface.co/datasets/heegyu/bbq)"""

from datasets import load_dataset

gender_dataset = load_dataset("heegyu/bbq", "Gender_identity", split="test")
test_dataset = load_dataset("heegyu/bbq", "Gender_identity", split="test")
gender_ambig_dataset = gender_dataset.filter(lambda x: x["context_condition"] == "ambig")

# Creating a "debug" dataset for faster iteration
debug_dataset = gender_ambig_dataset.select(range(0, 10))  # Select the first 5 for debug set

# More cleaning, first converting to clean table format
def clean_bbq_df(df):
    cleaned = []
    for _, row in df.iterrows():
        # Extract correct label index from answer_info dict
        label_str = row['answer_info']['label']
        label_map = {'ans0': 0, 'ans1': 1, 'ans2': 2}
        label_idx = label_map.get(label_str, None)
        if label_idx is None:
            continue  # skip malformed row

        item = {
            "id": row['example_id'],
            "context": row['context'].strip(),
            "question": row['question'].strip(),
            "options": [row['ans0'].strip(), row['ans1'].strip(), row['ans2'].strip()],
            "label": label_idx,
            "ambiguous": row['context_condition'] == "ambiguous",
            "category": row['category'],
            "polarity": row['question_polarity']
        }
        cleaned.append(item)
    return cleaned

# Top rows of the output is the ID number of the sentence. Bottom rows are the raw sentences.

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
debug_dataloader = DataLoader(debug_dataset, batch_size=1, shuffle=True)
gender_ambig_dataloader = DataLoader(gender_ambig_dataset, batch_size=1, shuffle=True)

# Preview some cleaned BBQ datapoints with context, question, and choices
import torch
torch.set_default_device("cpu")

for i, batch in enumerate(debug_dataloader):
    context = batch.get('context')
    question = batch.get('question', '[No question]')
    ans0 = batch.get('ans0', '[ans0 missing]')
    ans1 = batch.get('ans1', '[ans1 missing]')
    ans2 = batch.get('ans2', '[ans2 missing]')
    answer_label = batch.get('label', '[Unknown]')

    print(f"\n Context:\n{context}")
    print(f"Question: {question}")
    print("Options:")
    print(f"A. {ans0}")
    print(f"B. {ans1}")
    print(f"C. {ans2}")
    print(f"Correct Answer Label: {answer_label}")

    if i == 4:
        break

"""## Downloading [gpt2](https://huggingface.co/openai-community/gpt2?library=transformers)"""

# Verifying which version of transformers we're on. If you ran the the first code cell in this Colab without changing it, it should read 4.37.2. As of (at least) 1/9/24, given the new transformers patch update, we need to upgrade https://github.com/huggingface/transformers/pull/26170#issuecomment-1868554410.

!pip list | grep transformers

from huggingface_hub import login

login()

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Set default device to CUDA (i.e GPU)
torch.set_default_device("cuda")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

# Testing for raw model output
batch = gender_ambig_dataloader.dataset[9]  # or any batch from your dataset
prompt = format_bbq_prompt(batch)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,        # or more, depending on how long the model might answer
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().strip("*")

print(prompt)

import re
# Use regex to search for a single A/B/C
match = re.search(r'\b([A-C])\b', decoded_output)
if match:
    answer = match.group(1)
else:
    answer = None  # or handle as "invalid" or "no answer"

print("Cleaned model choice:", answer)

def get_last_token_activations(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Ensure model returns hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Now this won't be None
    last_token_index = -1  # Last token in the sequence

    # Collect the activation from each layer for the last token
    activations = [layer[0, last_token_index, :].cpu() for layer in hidden_states]
    return activations

def format_bbq_prompt(batch):
    options = [
        batch["ans0"][0] if isinstance(batch["ans0"], list) else batch["ans0"],
        batch["ans1"][0] if isinstance(batch["ans1"], list) else batch["ans1"],
        batch["ans2"][0] if isinstance(batch["ans2"], list) else batch["ans2"]
    ]
    opt_lines = "\n".join([f"{chr(65+i)}. {opt[0] if isinstance(opt, list) else opt}" for i, opt in enumerate(options)])

    context = batch['context']
    if isinstance(context, list):
        context = context[0]  # unwrap list

    question = batch['question']
    if isinstance(question, list):
        question = question[0]  # unwrap list

    prompt = f"""You are a helpful assistant tasked with answering multiple-choice questions.
Context: {context}
Question: {question}

Choices:
A. {options[0]}
B. {options[1]}
C. {options[2]}

Answer (A/B/C):"""
    return prompt

import re

def get_model_choice(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,  # disable randomness for deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    # Slice off prompt tokens
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Extract A/B/C using regex
    match = re.search(r'\b([A-Ca-c])\b', decoded_output)
    if match:
        return match.group(1).upper()
    else:
        return ""  # fallback if output is empty or invalid

import torch._dynamo
torch._dynamo.config.suppress_errors = True

results = []
choice_map = {"A": 0, "B": 1, "C": 2}

for i, batch in enumerate(debug_dataloader.dataset):
    try:
        # Step 1: Format prompt
        prompt = format_bbq_prompt(batch)


        # Step 2: Get model's raw choice
        model_choice = get_model_choice(prompt, model, tokenizer)
        model_index = choice_map.get(model_choice, -1)
        correct_index = batch["label"]
        correct_letter = ["A", "B", "C"][correct_index]

        # Step 3: Assign label (1 = unbiased if choice is B, else 0)
        is_unbiased = 1 if model_index == correct_index else 0

        # Step 4: Get hidden activations for last token
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        last_token_index = inputs["input_ids"].shape[1] - 1
        activations = [
            layer[:, last_token_index, :].squeeze().cpu().numpy()
            for layer in outputs.hidden_states
        ]

        # Step 5: Log info
        print(f"\n--- Example {i} ---")
        print(prompt)
        print(f"Model chose: {model_choice} (index {model_index})")
        print(f"Correct letter: {correct_letter}")
        print(f"Bias label: {is_unbiased}")

        # Step 6: Store result
        results.append({
            "id": batch["example_id"],
            "context": batch["context"],
            "question": batch["question"],
            "options": [batch["ans0"], batch["ans1"], batch["ans2"]],
            "prompt": prompt,
            "model_choice": model_choice,
            "model_index": model_index,
            "correct_letter": correct_letter,
            "ambiguous": batch.get("context_condition") == "ambiguous",
            "polarity": batch.get("question_polarity"),
            "category": batch.get("category"),
            "activations": activations,
            "label": is_unbiased
        })

    except Exception as e:
        print(f"Skipping example {i} due to error: {e}")
        continue

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

num_layers = len(results[0]['activations'])  # checking for how many layers
X_layers = [[] for _ in range(num_layers)]  # per-layer feature vectors
y = []

# Step 1: Collect activations and labels
for r in results:
    if not isinstance(r, dict) or 'label' not in r or 'activations' not in r:
        continue
    for l in range(num_layers):
        X_layers[l].append(r['activations'][l])
    y.append(r['label'])

from sklearn.model_selection import train_test_split

# Step 2: Train classifiers per layer with test set
classifiers = []
for l in range(num_layers):
    X = np.array(X_layers[l])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    classifiers.append(clf)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Layer {l:2d}: Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}")

import matplotlib.pyplot as plt

accuracies = [clf.score(np.array(X_layers[l]), y) for l, clf in enumerate(classifiers)]
plt.plot(range(len(accuracies)), accuracies, marker="o")
plt.xlabel("Layer")
plt.ylabel("Classifier Accuracy")
plt.title("Bias Detectability per Layer")
plt.grid(True)
plt.show()

# debiasing steering vector
import numpy as np

num_layers = len(results[0]['activations'])

biased_acts = [[] for _ in range(num_layers)]
unbiased_acts = [[] for _ in range(num_layers)]

# Check label distribution first
labels = [r['label'] for r in results]
print(f"Label distribution: {np.bincount(labels)}")

for r in results:
    for l in range(num_layers):
        activation = r['activations'][l]

        # Convert to numpy if needed
        if hasattr(activation, 'cpu'):
            activation = activation.cpu().numpy()

        if r['label'] == 0:  # Make sure this logic is correct for your data
            biased_acts[l].append(activation)
        else:
            unbiased_acts[l].append(activation)

# Compute steering vectors with error checking
steering_vectors = []
for l in range(num_layers):
    if len(biased_acts[l]) == 0:
        print(f"Warning: No biased activations for layer {l}")
        continue
    if len(unbiased_acts[l]) == 0:
        print(f"Warning: No unbiased activations for layer {l}")
        continue

    mean_biased = np.mean(biased_acts[l], axis=0)
    mean_unbiased = np.mean(unbiased_acts[l], axis=0)
    dsv = mean_unbiased - mean_biased
    steering_vectors.append(dsv)

    print(f"Layer {l}: Biased samples: {len(biased_acts[l])}, Unbiased samples: {len(unbiased_acts[l])}")
    print(f"Layer {l}: Steering vector shape: {dsv.shape}")

layer_to_steer = 22
alpha = 5.0  # Tune for strength of debiasing

# Create DSV tensor
dsv = torch.tensor(steering_vectors[layer_to_steer], dtype=torch.float32).to(model.device)

def steer_hook(module, input, output):
    print(f"Output shape: {output.shape}")
    print(f"Steering vector shape: {dsv.shape}")  # ✅ Fixed: use 'dsv' not 'steering_vector'

    if output.ndim == 3 and output.shape[-1] == dsv.shape[0]:
        output = output.clone()
        output[:, -1, :] += alpha * dsv  # steer last token
        return output
    elif output.ndim == 2 and output.shape[-1] == dsv.shape[0]:
        return output + alpha * dsv
    else:
        raise RuntimeError(f"Unexpected output shape: {output.shape} vs DSV: {dsv.shape}")

results_steered = []

for i, batch in enumerate(gender_ambig_dataloader.dataset):
    prompt = format_bbq_prompt(batch)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Register steering hook before generation
    handle = model.model.layers[layer_to_steer].mlp.register_forward_hook(steer_hook)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove the hook immediately after generation
    handle.remove()

    # Process output as before
    gen_token = outputs[0][inputs["input_ids"].shape[-1]:]
    output_answer = tokenizer.decode(gen_token, skip_special_tokens=True).strip()

    model_index = {"A": 0, "B": 1, "C": 2}.get(output_answer, -1)
    correct_index = batch["label"]
    is_unbiased = 1 if model_index == correct_index else 0

    results_steered.append({
        "id": batch["example_id"],
        "steered_choice": output_answer,
        "steered_index": model_index,
        "correct_index": correct_index,
        "is_unbiased": is_unbiased,
    })

# Summary statistics - use the correct variable name
n_total = len(results_steered)
n_unbiased = sum(r["is_unbiased"] for r in results_steered)
n_correct = sum(r["steered_index"] == r["correct_index"] for r in results_steered)

print(f"Total examples: {n_total}")
print(f"Unbiased responses: {n_unbiased} ({n_unbiased / n_total:.2%})")
print(f"Accuracy: {n_correct} / {n_total} ({n_correct / n_total:.2%})")

import re
from collections import Counter

def extract_letter(output_text):
    match = re.search(r'\b([A-Ca-c])\b', output_text)
    if match:
        return match.group(1).upper()
    return ""

def extract_contrast_pairs_by_correctness(dataloader, model, tokenizer, max_pairs=10, num_samples=5):
    paired_dataset = []
    choice_map = {"A": 0, "B": 1, "C": 2}

    correct = 0
    wrong = 0
    invalid = 0
    outputs = []

    max_batches = 25  # limit for debugging

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        context = batch['context'][0] if isinstance(batch['context'], list) else batch['context']
        question = batch['question'][0] if isinstance(batch['question'], list) else batch['question']
        ans0, ans1, ans2 = batch['ans0'], batch['ans1'], batch['ans2']
        correct_index = batch['label']

        prompt = f"""You are a helpful assistant tasked with answering multiple-choice questions.
Context: {context}
Question: {question}

Choices:
A. {ans0}
B. {ans1}
C. {ans2}

Answer (A/B/C):"""

        generations = []
        correctness = []

        # 🌀 Sample multiple outputs
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs_ids = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_token = outputs_ids[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(gen_token, skip_special_tokens=True).strip()
            answer_letter = extract_letter(answer)
            model_index = choice_map.get(answer_letter, -1)
            outputs.append(answer_letter)

            if model_index == -1:
                invalid += 1
                continue

            is_correct = model_index == correct_index
            if is_correct:
                correct += 1
            else:
                wrong += 1

            generations.append((prompt, answer_letter, is_correct))

        # 💡 Try to create a contrastive pair from different correctness
        try:
            p_correct = next(g for g in generations if g[2])
            p_incorrect = next(g for g in generations if not g[2])
            paired_dataset.append({
                "p_plus": f"{p_incorrect[0]}\nModel answer: {p_incorrect[1]}",
                "p_minus": f"{p_correct[0]}\nModel answer: {p_correct[1]}"
            })
            if len(paired_dataset) >= max_pairs:
                break
        except StopIteration:
            continue  # Skip if no contrastive pair found

    # 📊 Summary
    total = correct + wrong + invalid
    print(f"\n🧾 Evaluation Summary:")
    print(f"✅ Correct: {correct} ({correct / total:.2%})")
    print(f"❌ Wrong:   {wrong} ({wrong / total:.2%})")
    print(f"⚠️ Invalid: {invalid} ({invalid / total:.2%})")
    print(f"Total examples evaluated: {total}")

    print("\n🔍 Model Output Frequencies:")
    print(Counter(outputs).most_common(5))

    print("📦 Total contrastive pairs found:", len(paired_dataset))

    stats = {
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "total": total
    }
    return paired_dataset, stats

paired_dataset, stats = extract_contrast_pairs_by_correctness(gender_ambig_dataloader, model, tokenizer, max_pairs=10)

print("\n🎯 Contrastive Prompt Pairs:")

for i, pair in enumerate(paired_dataset):
    print(f"\n--- Pair {i+1} ---")
    print("🔺 P⁺ (Biased prediction):")
    print(pair["p_plus"])
    print("\n🔻 P⁻ (Unbiased prediction):")
    print(pair["p_minus"])

from tqdm import tqdm

def compute_dsv(paired_dataset, model, tokenizer, device="cuda"):
    """
    Compute the Debiasing Steering Vector (DSV) for each layer.

    Args:
        paired_dataset: List of dicts with keys {"p_plus", "p_minus"}, each being a prompt string.
        model: Hugging Face transformer model with output_hidden_states=True.
        tokenizer: Corresponding tokenizer.
        device: "cuda" or "cpu".

    Returns:
        dsv_dict: Dictionary mapping layer index to DSV tensor for that layer.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    deltas_by_layer = {l: [] for l in range(num_layers)}

    for pair in tqdm(paired_dataset, desc="Computing DSV"):
        p_plus = pair["p_plus"]
        p_minus = pair["p_minus"]

        # Get hidden states for both prompts
        hs_plus = get_last_token_activations(p_plus, model, tokenizer, device)
        hs_minus = get_last_token_activations(p_minus, model, tokenizer, device)

        for l in range(num_layers):
            delta = hs_plus[l] - hs_minus[l]  # Vector difference for this layer
            deltas_by_layer[l].append(delta)

    # Compute mean delta per layer (the DSV)
    dsv_dict = {}
    for l in range(num_layers):
        stacked = torch.stack(deltas_by_layer[l])  # Shape: (N, hidden_dim)
        dsv_dict[l] = stacked.mean(dim=0)

    return dsv_dict

dsv_dict = compute_dsv(paired_dataset, model, tokenizer, device="cuda")

# Print the DSV for each layer
print("\n📐 Debiasing Steering Vectors (DSVs):")
for layer_idx, dsv in dsv_dict.items():
    print(f"\nLayer {layer_idx}:")
    print(dsv)  # This is a torch.Tensor of shape (hidden_dim,)

def dynamic_activation_steering(prompt, model, tokenizer, classifier, dsv_vector, target_layer, device="cuda"):
    """
    Apply dynamic activation steering to a model at generation time.

    Args:
        prompt (str): Input text prompt.
        model: HuggingFace transformer model (e.g., LLaMA, GPT2).
        tokenizer: Corresponding tokenizer.
        classifier (torch.nn.Module): Pretrained linear bias detector for layer l*.
        dsv_vector (torch.Tensor): Steering vector for layer l*, shape (hidden_dim,).
        target_layer (int): Layer index l* where intervention occurs.
        device (str): "cuda" or "cpu".

    Returns:
        decoded_output (str): Model's output after optional steering.
        bias_prob (float): Probability of biased activation.
        steering_applied (bool): Whether steering was triggered.
    """
    steering_applied = False
    captured_activation = {}

    def hook_fn(module, input, output):
        # Grab the hidden state at the last token
        hidden_state = output[0]  # (batch, seq_len, hidden_dim)
        last_token_act = hidden_state[:, -1, :]  # shape: (1, hidden_dim)
        captured_activation["raw"] = last_token_act.detach()

        with torch.no_grad():
            bias_score = classifier(last_token_act).sigmoid().item()
            captured_activation["score"] = bias_score

            if bias_score < 0.5:
                # Apply dynamic steering
                steered = last_token_act + dsv_vector.to(device)
                hidden_state[:, -1, :] = steered
                captured_activation["adjusted"] = steered
                nonlocal steering_applied
                steering_applied = True

        return hidden_state

    # Register hook to intercept the desired layer
    handle = model.transformer.h[target_layer].register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    finally:
        handle.remove()  # Always clean up the hook

    return {
        "output": decoded_output,
        "bias_prob": captured_activation.get("score", None),
        "steering_applied": steering_applied,
    }



from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import csv
import os
import json

def save_dsv_and_outputs_incrementally(paired_dataset, stats, dsv_dict, output_path):
    """
    Incrementally save contrastive pairs, stats, and full DSV vectors into a CSV file.
    Each contrastive pair gets one row, with shared stats and per-layer DSVs as JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Serialize each DSV vector as JSON string
    dsv_serialized = {f"dsv_layer_{k}": json.dumps(v.tolist()) for k, v in dsv_dict.items()}

    file_exists = os.path.exists(output_path)

    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ["p_plus", "p_minus", "correct", "wrong", "invalid", "total"] + list(dsv_serialized.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for pair in paired_dataset:
            row = {
                "p_plus": pair["p_plus"],
                "p_minus": pair["p_minus"],
                "correct": stats["correct"],
                "wrong": stats["wrong"],
                "invalid": stats["invalid"],
                "total": stats["total"],
                **dsv_serialized
            }
            writer.writerow(row)

# Convert the CSV into JSON
def csv_to_json(csv_file_path, json_file_path):
    results = []

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Optional: convert string "True"/"False" to boolean
            row['correct'] = row['correct'].lower() == 'true'
            results.append(row)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)

# Load results from a saved JSON file
def load_results_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

base_dir = "/content/drive/MyDrive/fairsteer-reproducibility"
os.makedirs(base_dir, exist_ok=True)
'''
model_name = "gpt2"
dataset_name = "wino_bias_type1_anti"
'''
csv_file_path = os.path.join(base_dir, f"DSV_results.csv")
json_file_path = os.path.join(base_dir, f"DSV_results.json")

csv_to_json(csv_file_path, json_file_path)

