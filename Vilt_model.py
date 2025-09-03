from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/FloodNet')

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import matplotlib.pyplot as plt

all_answers = set()

for split in ["Training Question", "Valid Question"]:   # test'i dışarıda bıraktık
    q_json = f"/content/FloodNet/dataset/Questions/{split}.json"
    with open(q_json, "r") as f:
        data = json.load(f)

    for i, item in data.items():
        if "Ground_Truth" not in item:
            continue
        all_answers.add(str(item["Ground_Truth"]))  # string olarak ekle

answer2id = {ans: i for i, ans in enumerate(sorted(all_answers))}
id2answer = {i: ans for ans, i in answer2id.items()}

print("Toplam cevap sınıfı:", len(answer2id))
print("Örnek mapping:", list(answer2id.items())[:10])

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import matplotlib.pyplot as plt

class FloodNetVQADataset(Dataset):
    def __init__(self, img_dir, q_json, processor, answer2id):
        with open(q_json, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())
        self.img_dir = img_dir
        self.processor = processor
        self.answer2id = answer2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = f"{self.img_dir}/{item['Image_ID']}"
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found at {img_path}. Skipping this item.")
            return None  # Return None for missing image files

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image file {img_path}: {e}. Skipping this item.")
            return None # Return None for corrupted image files

        question = item['Question']
        answer = str(item.get("Ground_Truth", "-1"))  # test set için Ground_Truth yoksa '-1'

        encoding = self.processor(image, question, return_tensors="pt", padding="max_length", truncation=True)
        label = torch.tensor(self.answer2id[answer]) if answer in self.answer2id else torch.tensor(-1)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'pixel_values': encoding['pixel_values'].squeeze(),
            'labels': label
        }

import torch
torch.cuda.empty_cache()

import torch.nn.functional as F

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def collate_fn(batch):
    # Filter out None items first
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Separate the components
    input_ids = [example['input_ids'] for example in batch]
    attention_mask = [example['attention_mask'] for example in batch]
    pixel_values = [example['pixel_values'] for example in batch]
    labels = [example['labels'] for example in batch]

    # Pad input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Resize pixel_values to a fixed size before stacking
    resized_pixel_values = []
    for pv in pixel_values:
        # Assuming the original output from the processor is [C, H, W]
        # Resize to [C, 384, 384]
        # Add a check for empty pixel_values or incorrect dimensions
        if pv.numel() == 0 or pv.dim() != 3:
             print(f"Skipping item with invalid pixel_values shape: {pv.shape}")
             continue
        resized_pv = F.interpolate(pv.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
        resized_pixel_values.append(resized_pv)

    # Check if resized_pixel_values is empty after filtering
    if not resized_pixel_values:
        print("Warning: Batch became empty after processing pixel values.")
        return None

    try:
        pixel_values = torch.stack(resized_pixel_values)
    except RuntimeError as e:
        print(f"Error stacking pixel values after resizing: {e}")
        print("Shapes in batch after resizing:")
        for rpv in resized_pixel_values:
            print(rpv.shape)
        return None # Return None for batches with inconsistent image sizes even after resizing


    labels = torch.stack(labels)


    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels
    }


train_dataset = FloodNetVQADataset(
    img_dir="/content/FloodNet/dataset/Images/Train_Image",
    q_json="/content/FloodNet/dataset/Questions/Training Question.json",
    processor=processor,
    answer2id=answer2id
)
val_dataset = FloodNetVQADataset(
    img_dir="/content/FloodNet/dataset/Images/Valid_Image",
    q_json="/content/FloodNet/dataset/Questions/Valid Question.json",
    processor=processor,
    answer2id=answer2id
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# =====================
# 6. Model
# =====================
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model.config.num_labels = len(answer2id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

gc.collect()



from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

loss_fn = CrossEntropyLoss()
num_epochs = 15
batch_size = 2
accumulation_steps = 4 
scaler = torch.cuda.amp.GradScaler() 

for epoch in range(num_epochs):
 
    model.train()
    train_loss = 0
    correct, total = 0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        if batch is None:
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Filter out samples with label -1
        valid_indices = labels != -1
        if valid_indices.sum() == 0:
            continue

        input_ids = input_ids[valid_indices]
        attention_mask = attention_mask[valid_indices]
        pixel_values = pixel_values[valid_indices]
        labels = labels[valid_indices]

        # Mixed precision
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Step sonrası kalan gradientler
    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_train_loss = train_loss / (total if total > 0 else 1)
    train_acc = correct / (total if total > 0 else 1)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            valid_indices = labels != -1
            if valid_indices.sum() == 0:
                continue

            input_ids = input_ids[valid_indices]
            attention_mask = attention_mask[valid_indices]
            pixel_values = pixel_values[valid_indices]
            labels = labels[valid_indices]

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            val_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total > 0:
      avg_val_loss = val_loss / total
      val_acc = correct / total
    else:
      avg_val_loss = 0.0
      val_acc = 0.0
    val_acc = correct / (total if total > 0 else 1)
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

 
    torch.save(model.state_dict(), f"vilt_vqa_epoch{epoch+1}.pth")

!pip install gradio

import os
import json
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import gradio as gr

# 1. Eğitimde kullanılan cevap sözlüklerini yeniden oluşturun
# Not: Bu kısım, modelin tahmin ettiği ID'leri doğru cevaplara çevirmek için gereklidir.
all_answers = set()
for split in ["Training Question", "Valid Question"]:
    q_json = f"/content/FloodNet/dataset/Questions/{split}.json"
    if os.path.exists(q_json):
        with open(q_json, "r") as f:
            data = json.load(f)

        for i, item in data.items():
            if "Ground_Truth" not in item:
                continue
            all_answers.add(str(item["Ground_Truth"]))

answer2id = {ans: i for i, ans in enumerate(sorted(all_answers))}
id2answer = {i: ans for ans, i in answer2id.items()}

# 2. Cihazı tanımlayın (GPU varsa GPU, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Modeli yükleyin ve yapılandırın
model_path = "/content/drive/MyDrive/vilt_vqa_epoch3 (1).pth"

# ÖNEMLİ: num_labels değerini, hata mesajındaki doğru değere (3129) sabitleyin.
num_labels = 3129

try:
    loaded_model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa",
        num_labels=num_labels
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    print("Model başarıyla yüklendi ve çıkarım için hazır.")

except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    print("Lütfen model dosyasının doğru yolda olduğundan ve bozuk olmadığından emin olun.")
    exit()

# 4. İşlemciyi (processor) yükleyin
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def vqa_pipeline(image: Image.Image, question: str) -> str:
    """
    Gradio arayüzünden gelen görsel ve metin girdilerini alır,
    model ile tahmin yapar ve sonucu döndürür.
    """
    if not image or not question:
        return "Lütfen bir görsel yükleyin ve bir soru girin."

    # Görseli RGB formatına dönüştürün
    image = image.convert("RGB")

    # Girişleri modele uygun hale getirin
    inputs = processor(
        text=question,
        images=image,
        return_tensors="pt"
    ).to(device)

    # Model ile tahmin yapın
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()

    # Tahmini ID'yi gerçek cevaba dönüştürün
    pred_answer = id2answer.get(pred_id, f"Unknown_{pred_id}")
    return pred_answer

# 5. Gradio arayüzünü tanımlayın ve başlatın
iface = gr.Interface(
    fn=vqa_pipeline,
    inputs=[
        gr.Image(type="pil", label="Görsel Yükle"),
        gr.Textbox(label="Bir Soru Sor")
    ],
    outputs=gr.Textbox(label="Tahmin Edilen Cevap"),
    title="ViLT ile Görsel Soru Yanıtlama (VQA)",
    description="Bir görsel yükleyin ve görselle ilgili bir soru sorun."
)

iface.launch(share=True)

