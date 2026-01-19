import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from torchvision import transforms, models

app = FastAPI(title="Rose Leaf Disease Detection API")

# ------------------ Static & Templates ------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------ CLASS NAMES ------------------6
CLASS_NAMES = [
    "Black Spot",
    "Downy Mildew",
    "Dry Leaf",
    "Healthy leaf",
    "Insect Hole"
]

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model Load ------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("models/mobilenetv2_final.pth", map_location=device))
model.to(device)
model.eval()

# ------------------ Transforms ------------------
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------ Home Page ------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------ Prediction API ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        x = test_tf(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, dim=1)

        predicted_class = CLASS_NAMES[idx.item()]
        confidence = float(conf.item())

        return {
            "success": True,
            "class": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/health")
async def health():
    return {"status": "API is running"}
