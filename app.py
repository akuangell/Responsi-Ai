from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
import io
import os

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Label ──
CLASSES = ['cat', 'dog', 'wild']
LABELS  = {'cat': 'Kucing', 'dog': 'Anjing', 'wild': 'Liar'}

# ── Load Model ──
def load_model(model_path="best_model.pth"):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
print("Model berhasil dimuat!")

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File kosong"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0]

        scores = {cls: round(probs[i].item() * 100, 1) for i, cls in enumerate(CLASSES)}
        pred   = max(scores, key=scores.get)

        return jsonify({
            "prediction": pred,
            "label": LABELS[pred],
            "scores": scores,
            "labels": LABELS,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
