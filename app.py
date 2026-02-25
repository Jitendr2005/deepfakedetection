"""
Deepfake Detection — Streamlit App (Images Only)
Well-lit, polished UI for image-based deepfake detection.
"""
import streamlit as st
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

# Project root
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, MODEL_CONFIG

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: well-lit, clean, impressive
st.markdown("""
<style>
    /* Base: light, airy feel */
    .stApp {
        background: linear-gradient(180deg, #fafbfc 0%, #f0f2f5 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Headings */
    h1 {
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 700;
        color: #1a1d21;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #5c6370;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 2rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #9ca3af;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    /* Result cards */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.75rem 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.06);
        margin: 1rem 0;
    }
    
    .result-real {
        border-left: 4px solid #059669;
        background: linear-gradient(90deg, rgba(5,150,105,0.06) 0%, #ffffff 12%);
    }
    
    .result-fake {
        border-left: 4px solid #dc2626;
        background: linear-gradient(90deg, rgba(220,38,38,0.06) 0%, #ffffff 12%);
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .result-real .result-label { color: #059669; }
    .result-fake .result-label { color: #dc2626; }
    
    .result-confidence {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 0.35rem;
    }
    
    /* Progress bar container */
    .confidence-bars {
        margin-top: 1.25rem;
    }
    
    .bar-label {
        font-size: 0.85rem;
        color: #4b5563;
        margin-bottom: 0.25rem;
        display: flex;
        justify-content: space-between;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #374151;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    
    /* Image preview */
    .image-preview-box {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        background: #fff;
    }
    
    /* Metric cards in sidebar */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #1a1d21;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load model once and cache."""
    model_files = list(MODELS_DIR.glob("*.pth"))
    if not model_files:
        return None
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    try:
        from model import get_model
        from data_loader import get_transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model = get_model(
            model_name=MODEL_CONFIG["model_name"],
            num_classes=MODEL_CONFIG["num_classes"],
            dropout=MODEL_CONFIG["dropout"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        transform = get_transforms(is_training=False, input_size=MODEL_CONFIG["input_size"])
        return {
            "model": model,
            "transform": transform,
            "device": device,
        }
    except Exception as e:
        st.sidebar.warning(f"Model load failed: {e}")
        return None


def predict_image(predictor, image_array: np.ndarray) -> dict:
    """Run prediction on RGB image array (HWC)."""
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
        "real_probability": float(probs[0][0].item()),
        "fake_probability": float(probs[0][1].item()),
    }


def demo_result():
    """Placeholder result when no model is loaded (images-only demo)."""
    return {
        "prediction": "Real",
        "confidence": 0.92,
        "real_probability": 0.92,
        "fake_probability": 0.08,
    }


def main():
    # Header
    st.markdown("# 🛡️ Deepfake Detection")
    st.markdown(
        '<p class="subtitle">Upload a <strong>face image</strong> to check if it’s authentic or AI-generated. Images only — no video.</p>',
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Status")
        if predictor is not None:
            st.success("Model loaded")
            st.caption("Using saved checkpoint from `models/`")
        else:
            st.info("Demo mode (no model)")
            st.caption("Train a model and save it in `models/` to run real predictions.")
        st.markdown("---")
        st.markdown("### 📌 About")
        st.caption(
            "This app uses image-only input. Supported formats: JPG, PNG, WebP, BMP."
        )

    # Image upload — images only
    accepted = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
        help="Upload a single face image (no video).",
    )

    if uploaded is None:
        st.markdown("---")
        st.markdown(
            '<p style="text-align: center; color: #6b7280; font-size: 1.05rem;">📤 Use the uploader above to add a <strong>face image</strong> (JPG, PNG, WebP, or BMP).</p>',
            unsafe_allow_html=True,
        )
        return

    # Load image
    pil_image = Image.open(uploaded).convert("RGB")
    image_array = np.array(pil_image)

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.markdown("#### 📷 Your image")
        st.image(pil_image, use_container_width=True)

    with col_result:
        st.markdown("#### 🔍 Result")

        if predictor is not None:
            with st.spinner("Analyzing…"):
                result = predict_image(predictor, image_array)
        else:
            result = demo_result()

        is_real = result["prediction"] == "Real"
        card_class = "result-real" if is_real else "result-fake"
        st.markdown(
            f'<div class="result-card {card_class}">'
            f'<div class="result-label">{result["prediction"]}</div>'
            f'<div class="result-confidence">Confidence: {result["confidence"]:.1%}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="confidence-bars">', unsafe_allow_html=True)
        st.markdown("**Real**")
        st.progress(result["real_probability"])
        st.markdown("**Fake**")
        st.progress(result["fake_probability"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Collapsible details
    with st.expander("📊 Interpretation"):
        st.markdown(
            "- **Real**: The image is likely authentic (not a deepfake).  \n"
            "- **Fake**: The image may be AI-generated or manipulated.  \n"
            "Use this tool as an aid only; always combine with other checks for critical decisions."
        )


if __name__ == "__main__":
    main()
