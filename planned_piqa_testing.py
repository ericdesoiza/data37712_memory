import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL_PATH = "IDKYet" 
TOKENIZER_PATH = "SAVE SCRATCH PATH" 
TRAIN_DATA_PATH = "PIQA PIQA"
VALIDATION_DATA_PATH = "PIQA PIQA"
TEST_DATA_PATH = "PIQA SCRATCH PATH" 
OUTPUT_DIR = "RESULTS!!!"

LEARNING_RATE = 1e-5
EPOCHS = 2
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(examples):

    print("Processing!")

    questions = examples["question"]
    option_a = examples["option_a"]
    option_b = examples["option_b"]
    labels = examples["label"]

    inputs = []
    for q, a, b in zip(questions, option_a, option_b):
        prompt = f"Question: {q} Option A: {a} Option B: {b} Answer:"
        inputs.append(prompt)

    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt") 
    tokenized_inputs["labels"] = torch.tensor(labels)

    print("Processed Data!")

    return tokenized_inputs

def train(model, train_dataloader):
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dataloader:
            inputs = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        validation_accuracy = evaluate(model, valid_dataloader)
        print(f"Epoch {epoch+1}: Validation Accuracy: {validation_accuracy}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            print(f"Saving best model checkpoint to {OUTPUT_DIR}")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR) 


def evaluate(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    print("Evaluating!")

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy

if __name__ == "__main__":
    tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = LlamaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(DEVICE)

    train_dataset = load_dataset("csv", data_files=TRAIN_DATA_PATH)
    processed_train_dataset = train_dataset.map(preprocess_function, batched=True)

    train_dataloader = DataLoader(
        processed_train_dataset["train"],
        batch_size=8,
    )

    valid_dataset = load_dataset("csv", data_files=VALIDATION_DATA_PATH)
    processed_valid_dataset = valid_dataset.map(preprocess_function, batched=True)

    valid_dataloader = DataLoader(
        processed_valid_dataset["train"],
        batch_size=8,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    best_validation_accuracy = 0.0

    test_dataset = load_dataset("csv", data_files=TEST_DATA_PATH)
    processed_test_dataset = test_dataset.map(preprocess_function, batched=True)

    test_dataloader = DataLoader(
        processed_test_dataset["train"],
        batch_size=8,
    )

    test_accuracy = evaluate(model, test_dataloader)
    print(f"Test Accuracy: {test_accuracy}")


