
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from pathlib import Path
from typing import List
import json
from torchmetrics import Accuracy
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import shutil
import csv

pl.seed_everything(42)


EPOCH = 4
MAX_WORKERS = 8
BATCH_SIZE = 32
DATASET_PATH = "Dataset_rvl"
OCR_PATH = "ocrDataset"
OUTPUT_PATH = "output"


def scale_bounding_box(box: List[int], width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]


class DocumentDataset(Dataset):

    def __init__(self, csv_dict, processor):
        self.dataset_dict = csv_dict
        self.processor = processor

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, item):
        image_path = f"{DATASET_PATH}/Images/{Path(self.dataset_dict[item]['nome']).name}"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        json_path = Path(OCR_PATH)/f"{Path(image_path).stem}_res.json"
        with json_path.open("r") as f:
            ocr_result = json.load(f)
        width_scale  = 1000 / width
        height_scale = 1000 / height
        words = []
        boxes = []
        for row in range(len(ocr_result["rec_boxes"])):
            boxes.append(scale_bounding_box(ocr_result["rec_boxes"][row], width_scale, height_scale))
            words.append(ocr_result["rec_texts"][row])

        label = int(self.dataset_dict[item]['classe'])
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class ModelModule(pl.LightningModule):

    def __init__(self, n_classes: int):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)


    def forward(self, input_ids=None, attention_mask=None, bbox=None, pixel_values=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        output = self.process_batch(batch)
        loss = output.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.train_acc(output.logits, labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        output = self.process_batch(batch)
        loss = output.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.val_acc(output.logits, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        output = self.process_batch(batch)
        loss = output.loss
        self.log("test_loss", loss)
        self.test_acc(output.logits, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.00001)

    def process_batch(self, batch):

        inputs = {}

        if "input_ids" in batch:
            inputs["input_ids"] = batch["input_ids"]

        if "attention_mask" in batch:
            inputs["attention_mask"] = batch[
                "attention_mask"
            ]

        if "bbox" in batch:
            inputs["bbox"] = batch["bbox"]

        if "pixel_values" in batch:
            inputs["pixel_values"] = batch[
                "pixel_values"
            ]

        inputs["labels"] = batch["labels"]

        return self(**inputs)


def predict_document_image(image_path: Path,
                           model: LayoutLMv3ForSequenceClassification,
                           processor: LayoutLMv3Processor
                           ):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    json_path = Path(OCR_PATH)/f"{image_path.stem}_res.json"
    with json_path.open("r") as f:
        ocr_result = json.load(f)
    width_scale  = 1000 / width
    height_scale = 1000 / height
    words = []
    boxes = []
    for row in range(len(ocr_result["rec_boxes"])):
        boxes.append(scale_bounding_box(ocr_result["rec_boxes"][row], width_scale, height_scale))
        words.append(ocr_result["rec_texts"][row])
    encoding = processor(
        image,
        words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.inference_mode():
        output = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            bbox=encoding["bbox"].to(DEVICE),
            pixel_values=encoding["pixel_values"].to(DEVICE)
        )
    predicted_class = output.logits.argmax()
    return predicted_class.item()


feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
    "microsoft/layoutlmv3-base"
)
processor = LayoutLMv3Processor(feature_extractor, tokenizer)


dataset_train = []
dataset_val = []
dataset_test = []

csv_train = Path(f"{DATASET_PATH}/train_dataset.csv")
csv_val = Path(f"{DATASET_PATH}/val_dataset.csv")
csv_test = Path(f"{DATASET_PATH}/test_dataset.csv")

if csv_train.exists():
    with open(csv_train, mode='r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            dataset_train.append(row)

if csv_val.exists():
    with open(csv_val, mode='r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            dataset_val.append(row)

if csv_test.exists():
    with open(csv_test, mode='r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            dataset_test.append(row)


train_set = DocumentDataset(dataset_train, processor)
val_set = DocumentDataset(dataset_val, processor)
test_set = DocumentDataset(dataset_test, processor)
train_data_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=MAX_WORKERS,
)
val_data_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=MAX_WORKERS,
)
test_data_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=MAX_WORKERS,
)
checkpoint_dir = Path(f"{OUTPUT_PATH}/checkpoint_rvl/")
checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)
last_checkpoint = checkpoint_dir / "last.ckpt"

model_module = ModelModule(16)

model_checkpoint = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="{epoch}-{step}-{val_loss:.4f}",
    save_last=True,
    save_top_k=3,
    monitor="val_loss",
    mode="min"
)

trainer = pl.Trainer(
    accelerator="gpu",
    precision=16,
    devices=1,
    max_epochs=EPOCH,
    callbacks=[
        model_checkpoint
    ]
)
if last_checkpoint.exists():
    checkpoint_path = str(last_checkpoint)
else:
    checkpoint_path = None
trainer.fit(model_module, train_data_loader, val_data_loader, ckpt_path=checkpoint_path)
trained_model = ModelModule.load_from_checkpoint(
    model_checkpoint.best_model_path,
    n_classes=16
)
trained_model.model.save_pretrained(Path(f"{OUTPUT_PATH}/checkpoint_rvl/best-model"))




DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
models = []
results = {}
model = LayoutLMv3ForSequenceClassification.from_pretrained(f"{OUTPUT_PATH}/checkpoint_rvl/best-model")
model.eval().to(DEVICE)
labels = []
predictions = []
for item in tqdm(dataset_test):
    labels.append(int(item['classe']))
    predictions.append(predict_document_image(Path(f"{DATASET_PATH}/Images/{Path(item['nome']).name}"), model, processor))
cm = confusion_matrix(labels, predictions)
accuracy = accuracy_score(
    labels,
    predictions
)
macro_f1 = f1_score(
    labels,
    predictions,
    average="macro"
)
weighted_f1 = f1_score(
    labels,
    predictions,
    average="weighted"
)
cr = classification_report(
    labels,
    predictions,
    digits=4,
    zero_division=0
)
results["confusion_matrix"] = cm
results["accuracy"] = accuracy
results["macro_f1"] = macro_f1
results["weighted_f1"] = weighted_f1
results["classification_report"] = cr


with open(f"{OUTPUT_PATH}/output.txt", mode='w') as file:
    file.write(f"confusion_matrix = \n {results["confusion_matrix"]}\n")
    file.write(f"accuracy = \n {results["accuracy"]}\n")
    file.write(f"macro_f1 = \n {results["macro_f1"]}\n")
    file.write(f"weighted_f1 = \n {results["weighted_f1"]}\n")
    file.write(f"classification_report = \n {results["accuracy"]}\n")

shutil.make_archive("output", "zip", "output.zip")