import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# --- Configuration: use an ultra-small model to fit M1 CPU ---
# Why choose bert-tiny:
# 1) Only ~4.4M parameters, suitable for CPU training
# 2) Keeps the full BERT architecture intact, making it easier to understand where Adapters are inserted
# 3) Trains quickly on an M1 chip, great for demos and learning
MODEL_NAME = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
NUM_LABELS = 2  # Binary classification: positive/negative, to keep complexity low and focus on the Adapter mechanism

print("=== Core differences: Adapter fine-tuning vs LoRA fine-tuning ===")
print("1. Adapter: insert small neural network modules between model layers")
print("2. LoRA: modify existing weight matrices via low-rank matrix decomposition")
print("3. Adapter adds new layers; LoRA changes the weights of existing layers")

# --- Manual implementation of an Adapter layer (to highlight Adapter characteristics) ---
class AdapterLayer(nn.Module):
    """
    Core implementation of an Adapter layer — the essence of Adapter fine-tuning.

    Design idea: achieve parameter-efficient fine-tuning via a "bottleneck" structure
    - Down projection: compress input dimension (e.g., 128) to a smaller dimension (e.g., 16) as an information bottleneck
      Purpose: force the model to learn the most important representations and reduce parameter count
    - Activation: ReLU adds non-linearity so the Adapter can learn more complex patterns
    - Up projection: expand features back to the original dimension to stay compatible with the base model
    - Residual connection: preserve the original information flow (crucial!)

    Why use a residual connection?
    1) Even if the Adapter outputs zeros, the original representation still passes through unchanged
    2) The Adapter learns "delta" (incremental) information rather than replacing existing representations
    3) More stable training and helps avoid vanishing gradients
    """
    def __init__(self, input_dim, adapter_dim=16):
        super().__init__()
        # Down projection: large dim -> small dim (information bottleneck)
        self.down_project = nn.Linear(input_dim, adapter_dim)
        # Activation: add non-linearity
        self.activation = nn.ReLU()
        # Up projection: small dim -> original dim (compatibility with base model)
        self.up_project = nn.Linear(adapter_dim, input_dim)
        # Dropout: prevent overfitting (important since Adapters have relatively few parameters)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Forward: down -> activation -> up, then add residual
        adapter_output = self.up_project(self.activation(self.down_project(x)))
        return x + self.dropout(adapter_output)


# Full architecture: base model + Adapters
class ModelWithAdapter(nn.Module):
    """
    Full Adapter fine-tuning architecture.

    Core design:
    1) Keep the pretrained base model unchanged (freeze parameters)
    2) Insert trainable Adapter modules at key locations
    3) Let Adapters learn task-specific knowledge, while the base model provides general language understanding

    Advantages:
    - Parameter efficiency: train <1% of parameters for good performance
    - Modularity: different tasks can use different Adapters while sharing the same base model
    - Stability: does not destroy knowledge learned during pretraining
    """
    def __init__(self, base_model, adapter_dim=16):
        super().__init__()
        self.base_model = base_model

        # Freeze all base model parameters (key property of Adapter fine-tuning)
        # Why freeze?
        # 1) Preserve pretrained knowledge: avoid overfitting on a small dataset and harming general ability
        # 2) Parameter efficiency: train only small Adapter params, greatly reducing compute cost
        # 3) Avoid catastrophic forgetting: ensure the model doesn't forget pretrained language knowledge
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add an Adapter after each Transformer layer in BERT
        # Why add to every layer? Different layers learn different abstraction levels:
        # - Lower layers: lexical/syntactic features
        # - Middle layers: syntax/semantic relations
        # - Higher layers: task-specific abstract concepts
        # Adapters in every layer let the model adapt at multiple abstraction levels
        hidden_size = base_model.config.hidden_size
        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_size, adapter_dim)
            for _ in range(base_model.config.num_hidden_layers)
        ])

        print("✓ Base model parameters frozen")
        print(f"✓ Added {len(self.adapters)} Adapter layers")
        print(f"✓ Params per Adapter layer: {adapter_dim * hidden_size * 2 + adapter_dim + hidden_size}")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Get base model hidden states
        # Why output_hidden_states=True?
        # We need outputs from each layer so we can apply the corresponding Adapter after each layer.
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True  # critical: get all layers' hidden states
        }
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids

        outputs = self.base_model.bert(**bert_inputs)
        hidden_states = outputs.hidden_states  # embedding output + all transformer layer outputs

        # Apply an Adapter to each Transformer layer output
        # Why start from hidden_states[1:]?
        # hidden_states[0] is the embedding output; we start after the first Transformer layer.
        adapted_hidden_states = []
        for hidden_state, adapter in zip(hidden_states[1:], self.adapters):
            adapted_state = adapter(hidden_state)
            adapted_hidden_states.append(adapted_state)

        # Use the final Adapter output for classification
        # Why the last layer?
        # 1) The last layer has the highest-level semantic abstraction, best for classification decisions
        # 2) After all Adapters, it includes task adaptation signals across levels
        # 3) Follows BERT convention: use [CLS] representation from the last layer
        pooled_output = adapted_hidden_states[-1][:, 0, :]  # [CLS] representation

        # Use the original classifier head for prediction
        # Note: classifier is also frozen, so only Adapters learn task-specific knowledge
        logits = self.base_model.classifier(pooled_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


# --- Demo data ---
# Clear positive/negative contrast for easy observation
data = {
    "text": ["Very useful", "Terrible", "Not bad", "Awful", "Excellent", "Very disappointed"],
    "label": [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
}
dataset = Dataset.from_pandas(pd.DataFrame(data)).rename_column("label", "labels")


def tokenize_function(examples):
    # Why max_length=32?
    # 1) Short texts usually fit within 32 tokens, avoiding unnecessary padding
    # 2) Less compute, faster on CPU
    # 3) Enough for sentiment analysis to capture key information
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=32)


tokenized_dataset = dataset.map(tokenize_function, batched=True).remove_columns(["text"])

# --- Create model with Adapters ---
print("\n=== Creating Adapter model ===")
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Why adapter_dim=8?
# 1) Balance quality and efficiency: 8 dims can learn basic task patterns with very few params
# 2) Avoid overfitting: on tiny datasets, too large adapter_dim can overfit
# 3) Compute-friendly: small dims are faster on CPU
adapter_model = ModelWithAdapter(base_model, adapter_dim=8)

# Parameter stats: compare total params vs trainable params to see Adapter efficiency
total_params = sum(p.numel() for p in adapter_model.parameters())
trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
print("💡 Key observation: training <1% of parameters is the core advantage of Adapter fine-tuning!")

# --- Quick training demo ---
training_args = TrainingArguments(
    output_dir="./adapter_output",
    num_train_epochs=2,          # Why 2 epochs?
                                # 1) Adapters have few params and converge quickly; avoids overfitting
                                # 2) Fast demo to see results
                                # 3) Too many epochs on tiny data can overfit
    per_device_train_batch_size=2,  # Small batch works well for small data and CPU
    learning_rate=1e-3,          # Why 1e-3 instead of typical 1e-5?
                                # Because we only train Adapter params, a larger LR speeds up convergence
    logging_steps=1,             # Log every step for easy observation
    save_steps=100,              # Save less often to reduce I/O overhead
    report_to="none",            # No wandb etc., keep the demo simple
    remove_unused_columns=False, # Keep full inputs
)

trainer = Trainer(
    model=adapter_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n=== Start Adapter fine-tuning ===")
print("💡 Core idea: only the inserted Adapter module parameters are trained; the base model stays frozen.")
print("This preserves pretrained knowledge while adapting the model to a new task!")
trainer.train()

# --- Inference demo ---
print("\n=== Adapter inference demo ===")
adapter_model.eval()  # eval mode disables dropout etc.

# Ensure CPU inference (fits M1 environment)
# Why explicitly CPU?
# 1) MPS on M1 can be unstable; CPU is more reliable
# 2) Tiny models are already fast on CPU
# 3) Avoid device mismatch errors
adapter_model = adapter_model.cpu()
test_texts = ["Excellent", "Very disappointed", "It's okay"]

print("💡 What to watch: how the trained Adapter changes the model’s predictions")
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = adapter_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        result = "Positive" if prediction == 1 else "Negative"
        print(f"Text: '{text}' -> Prediction: {result} (confidence: {confidence:.3f})")
