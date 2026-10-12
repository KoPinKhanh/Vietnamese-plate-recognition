import csv
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

parseq = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True, trust_repo=True).eval().cuda()

val_csv = Path('datasets/VNLP_processed/val.csv')

# Standard parseq resize: (32, 128)
img_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

records = []
with open(val_csv, 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        records.append(r)

print(f"Testing pretrained PARSeq zero-shot on 100 validation samples...")
correct = 0
total = 100

for item in records[:total]:
    img = Image.open(item['image_path']).convert('RGB')
    # If bbox present, crop
    if item['x1']:
        bbox = [int(float(item['x1'])), int(float(item['y1'])), int(float(item['x2'])), int(float(item['y2']))]
        # expand pad_ratio 0.03
        w, h = img.size
        pad_x = int((bbox[2] - bbox[0]) * 0.03)
        pad_y = int((bbox[3] - bbox[1]) * 0.03)
        b_crop = [max(0, bbox[0] - pad_x), max(0, bbox[1] - pad_y), min(w, bbox[2] + pad_x), min(h, bbox[3] + pad_y)]
        img = img.crop(b_crop)

    tensor = img_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = parseq(tensor)
        pred = logits.softmax(-1)
        label, conf = parseq.tokenizer.decode(pred)
        pred_clean = label[0].upper().replace('-', '').replace('.', '').replace(' ', '')
        gt_clean = item['plate_text'].upper()
        if pred_clean == gt_clean:
            correct += 1

print(f"Zero-shot PARSeq on 100 samples: {correct}/{total} ({correct/total*100:.1f}%)")
