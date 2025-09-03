# ViLT Sel Afeti Analiz Sistemi

ViLT modeli kullanarak sel afeti görüntülerinde soru-cevap analizi yapan sistem.

## 📁 Dosyalar

**🔗 Eğitilmiş Model:** https://drive.google.com/file/d/1MHEEqvNtb5eMo6eEd8FHmc4deO5vcpQE/view?usp=sharing
- Dosya: `vilt_vqa_epoch3.pth`
- Boyut: ~500MB

**📊 Dataset:** https://drive.google.com/file/d/15NI9N9auATXljCUEI7-lfxzfJfln4qp-/view?usp=sharing  
- FloodNet Track 2 veri seti
- 3129 farklı cevap kategorisi

## 🚀 Kullanım

### Kurulum
```bash
pip install torch transformers PIL gradio
```

### Hızlı Başlatma
```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# Model yükleme
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", num_labels=3129)
model.load_state_dict(torch.load("vilt_vqa_epoch3.pth"))
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Tahmin yapma
inputs = processor(text="Bu bölgede hasar var mı?", images=image, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
```

## 📋 Model Bilgileri

- **Model:** ViLT (dandelin/vilt-b32-finetuned-vqa)
- **Dataset:** FloodNet Track 2
- **Epoch:** 15 (en iyi: 3. epoch)
- **Cevap Sınıfı:** 3129

## 🎯 Örnek Kullanım

**Sorular:**
- "Bu bölgede sel hasarı var mı?"
- "Kaç tane bina zarar görmüş?"
- "Yollar geçilebilir durumda mı?"

## 🚧 Gelecek Planları

1. FloodNet Track 1 ile segmentasyon modeli eğitimi
2. Track 2 görsellerinin segmentasyonu 
3. Hibrit sistem (Segmentasyon + Soru-Cevap) geliştirme

---
**Proje:** Staj çalışması | **Versiyon:** 1.0
