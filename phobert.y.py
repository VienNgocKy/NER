import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Define label map
label_map = {label: i for i, label in enumerate(['O', 'PER', 'ORG', 'ADD', 'DATE', 'PHONE', 'EVE', 'PRO', 'SKILL', 'URL', 'EMAIL', 'QUA', 'PTL', 'LOC', 'TIME', 'IP'])}

# Step 1: Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Step 2: Preprocess data
def preprocess_data(data):
    sentences = []
    entities = []
    for item in data:
        sentences.append(item['sentence'])
        entities.append(item['entities'])

    df = pd.DataFrame({'sentence': sentences, 'entities': entities})
    
    tokenized_sentences = []
    labels = []

    for i, row in df.iterrows():
        sentence = row['sentence']
        entities = row['entities']

        tokens = list(sentence)
        label = ['O'] * len(tokens)

        for entity in entities:
            start, end, label_type = entity
            label[start:end] = [label_type] * (end - start)

        tokenized_sentences.append(tokens)
        labels.append(label)
    
    return tokenized_sentences, labels

# Step 3: Convert to Dataset
def convert_to_dataset(sentences, labels, tokenizer, label_map):
    input_ids = []
    attention_masks = []
    label_ids = []

    for i, sentence in enumerate(sentences):
        encoding = tokenizer(sentence, is_split_into_words=True, padding='max_length', truncation=True, max_length=128, clean_up_tokenization_spaces=True)
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

        label_id = [label_map[label] for label in labels[i]]
        label_id += [label_map['O']] * (128 - len(label_id))
        label_ids.append(label_id)

    dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': label_ids
    })

    return dataset

def train_test_split_data(sentences, labels, test_size=0.2):
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=test_size, random_state=42)
    return train_sentences, test_sentences, train_labels, test_labels

def main():
    output_dir = r'D:\Kỳ_TT\results'
    
    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tải dữ liệu
    data = load_data('ner_data.json')

    # Tiền xử lý dữ liệu
    tokenized_sentences, labels = preprocess_data(data)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_sentences, test_sentences, train_labels, test_labels = train_test_split_data(tokenized_sentences, labels)

    # Định nghĩa bản đồ nhãn
    label_map = {label: i for i, label in enumerate(['O', 'PER', 'ORG', 'ADD', 'DATE', 'PHONE', 'EVE', 'PRO', 'SKILL', 'URL', 'EMAIL', 'QUA', 'PTL', 'LOC', 'TIME', 'IP'])}

    # Tải tokenizer và mô hình PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=len(label_map))

    # Chuyển đổi thành dataset
    train_dataset = convert_to_dataset(train_sentences, train_labels, tokenizer, label_map)
    test_dataset = convert_to_dataset(test_sentences, test_labels, tokenizer, label_map)

    # Định nghĩa tham số huấn luyện
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Sử dụng eval_strategy thay vì evaluation_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Định nghĩa Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Huấn luyện mô hình
    trainer.train()

    # Đánh giá mô hình
    results = trainer.evaluate()

    # In kết quả
    print(results)

    # Lưu mô hình
    trainer.save_model(os.path.join(output_dir, "trained_model"))

if __name__ == "__main__":
    main()
