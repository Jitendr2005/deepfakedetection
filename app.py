"""
Deepfake Detection — Streamlit App
Updated with Multi-Model Support (Transformer & CNN)
"""
import streamlit as st
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

# Project root
sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path("models")
MODEL_CONFIG = {
    "model_name": "efficientnet_b0",  # Default model
    "available_models": ["vit_base", "efficientnet_b0", "resnet50", "swin_base", "model_a", "model_c", "ensemble"],
    "input_size": 224,
    "num_classes": 2,  # Real vs Fake
    "pretrained": True,
    "dropout": 0.5
}

# Page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #fafbfc 0%, #f0f2f5 100%); }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    h1 { font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 700; color: #1a1d21; letter-spacing: -0.02em; }
    .subtitle { font-size: 1.1rem; color: #5c6370; margin-bottom: 2rem; }
    [data-testid="stFileUploader"] { background: #ffffff; border: 2px dashed #d1d5db; border-radius: 16px; padding: 2rem; }
    .result-card { background: #ffffff; border-radius: 16px; padding: 1.75rem 2rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.06); margin: 1rem 0; }
    .result-real { border-left: 4px solid #059669; }
    .result-fake { border-left: 4px solid #dc2626; }
    .result-label { font-size: 1.5rem; font-weight: 700; }
    .result-real .result-label { color: #059669; }
    .result-fake .result-label { color: #dc2626; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor(model_name: str):
    """Load specific model checkpoint."""
    # Find the best checkpoint for this architecture
    checkpoints = list(MODELS_DIR.glob(f"best_{model_name}*.pth"))
    if not checkpoints:
        # Fallback to any model if exact match not found
        checkpoints = list(MODELS_DIR.glob("*.pth"))
        if not checkpoints: return None
        model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    else:
        model_path = checkpoints[0]

    try:
        from model import get_model
        from data_loader import get_transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Use model name from checkpoint if available, else from args
        actual_model_name = checkpoint.get("model_name", model_name)
        
        model = get_model(
            model_name=actual_model_name,
            num_classes=MODEL_CONFIG["num_classes"],
            dropout=MODEL_CONFIG["dropout"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        transform = get_transforms(is_training=False, input_size=MODEL_CONFIG["input_size"])
        return {"model": model, "transform": transform, "device": device, "name": actual_model_name}
    except Exception as e:
        st.sidebar.error(f"Error loading {model_name}: {e}")
        return None

def predict_image(predictor, image_array: np.ndarray) -> dict:
    transform = predictor["transform"]
    transformed = transform(image=image_array)
    image_tensor = transformed["image"].unsqueeze(0).to(predictor["device"])

    with torch.no_grad():
        outputs = predictor["model"](image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1).item()

    return {
        "prediction": "Fake" if pred == 1 else "Real",
        "confidence": float(probs[0][pred].item()),
        "probabilities": [float(p) for p in probs[0]]
    }

def main():
    st.markdown("# 🛡️ Deepfake Detection")
    st.markdown('<p class="subtitle">Select a model and upload a face image to verify authenticity.</p>', unsafe_allow_html=True)

    # Model Selection in Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox(
            "Select Model Architecture",
            options=MODEL_CONFIG["available_models"],
            index=1 # Default to efficientnet_b0
        )
        
        if selected_model == "ensemble":
            st.info("Ensemble mode uses Face, Eyes, and Nose regions with Majority Voting.")
            # Map region to checkpoint names
            model_paths = {
                'face': str(MODELS_DIR / "best_model_c.pth"),
                'eyes': str(MODELS_DIR / "best_model_a_eyes.pth"),
                'nose': str(MODELS_DIR / "best_model_a_nose.pth")
            }
            
            from ensemble import DeepfakeEnsemble
            try:
                predictor = DeepfakeEnsemble(model_paths, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                predictor.name = "Ensemble (CViT + CNN)"
            except Exception as e:
                predictor = None
                st.warning(f"Note: Ensemble models not found. Demo mode active.")
        else:
            predictor = load_predictor(selected_model)
        
        if predictor:
            name = predictor.name if hasattr(predictor, 'name') else predictor['name']
            st.success(f"Model {name} active")
        else:
            st.warning("No checkpoint found. Showing demo results.")
            st.caption("Train the model using: `python3 train.py --model " + selected_model + "`")

    # File Uploader
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        image_array = np.array(pil_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            from ensemble import DeepfakeEnsemble
            if isinstance(predictor, DeepfakeEnsemble):
                with st.spinner("Processing regions and voting..."):
                    res = predictor.predict(image_array)
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    # Show extracted regions
                    st.write("### 🖼️ Extracted Regions")
                    r_cols = st.columns(3)
                    with r_cols[0]:
                        st.image(res["extracted_regions"]["face"], caption="Aligned Face", use_container_width=True)
                    with r_cols[1]:
                        st.image(res["extracted_regions"]["eyes"], caption="Eyes Region", use_container_width=True)
                    with r_cols[2]:
                        st.image(res["extracted_regions"]["nose"], caption="Nose Region", use_container_width=True)
                    
                    is_real = res["prediction"] == 0
                    pred_text = "Real" if is_real else "Fake"
                    card_color = "result-real" if is_real else "result-fake"
                    
                    st.markdown(f"""
                        <div class="result-card {card_color}">
                            <div class="result-label">{pred_text}</div>
                            <div>Confidence: {res['confidence']:.1%}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show breakdowns
                    with st.expander("Region Breakdown"):
                        for reg, p in res["region_predictions"].items():
                            label = "Real" if p == 0 else "Fake"
                            st.write(f"**{reg.capitalize()}**: {label}")

            else:
                if predictor:
                    with st.spinner("Analyzing..."):
                        res = predict_image(predictor, image_array)
                else:
                    # Demo Mode
                    res = {"prediction": "Real", "confidence": 0.95, "probabilities": [0.95, 0.05]}
                
                is_real = res["prediction"] == "Real"
                card_color = "result-real" if is_real else "result-fake"
                
                st.markdown(f"""
                    <div class="result-card {card_color}">
                        <div class="result-label">{res['prediction']}</div>
                        <div>Confidence: {res['confidence']:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(res['probabilities'][0], text=f"Real: {res['probabilities'][0]:.1%}")
                st.progress(res['probabilities'][1], text=f"Fake: {res['probabilities'][1]:.1%}")

if __name__ == "__main__":
    main()
