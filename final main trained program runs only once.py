import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from transformers import (
    DeiTForImageClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = r"C:\Users\Utkarsh Shukla\Desktop\project\archive\The IQ-OTHNCCD lung cancer dataset"

model_dir = "./trained_model"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ImageFolder(root=data_dir, transform=train_transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return {
        'pixel_values': torch.stack(images),
        'labels': torch.tensor(labels)
    }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    lr_scheduler_type='cosine_with_restarts',
    fp16=True
)

if os.path.exists(model_dir):
    print("Loading the trained model...")
    model = DeiTForImageClassification.from_pretrained(model_dir)
else:
    print("Training the model...")
    model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224', num_labels=3)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn
    )

    trainer.train()

    model.save_pretrained(model_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_collate_fn
)
eval_results = trainer.evaluate()
print(eval_results)

def visualize_predictions(image_paths, true_labels, predicted_labels, class_names, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 5))
    for i in range(num_images):
        image = Image.open(image_paths[i])
        true_label = class_names[true_labels[i]]
        predicted_label = class_names[predicted_labels[i]]
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPredicted: {predicted_label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

class_names = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

val_predictions = trainer.predict(val_dataset)
y_true = val_predictions.label_ids
y_pred = np.argmax(val_predictions.predictions, axis=1)
visualize_predictions([val_dataset.dataset.samples[i][0] for i in range(len(val_dataset))], y_true, y_pred, class_names)

def predict_and_show_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    predicted_class_name = class_names[predicted_class_idx]
    plt.imshow(Image.open(image_path))
    plt.title(f"Predicted class: {predicted_class_name}")
    plt.axis('off')
    plt.show()
    print(f"Predicted class: {predicted_class_name}")

example_image_path = r"Abnormal.jpg"
predict_and_show_image(example_image_path, model, class_names)
