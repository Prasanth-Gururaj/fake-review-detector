import os
import time
import requests
import streamlit as st

# Point this to your FastAPI backend
API_URL = os.getenv("FAKE_REVIEW_API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Fake Review Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Fake Review Detector")
st.write("Paste a product review and see whether it looks **fake** or **genuine** based on a DistilBERT + feature model optimized with ONNX Runtime.")

with st.form("review_form"):
    text = st.text_area(
        "Review text",
        height=140,
        placeholder="AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!",
    )
    submitted = st.form_submit_button("Analyze review")

if submitted:
    if not text.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner("Contacting model API..."):
            t0 = time.time()
            try:
                resp = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=5,
                )
                latency_ms = (time.time() - t0) * 1000
            except Exception as e:
                st.error(f"Could not reach API: {e}")
            else:
                if resp.status_code != 200:
                    st.error(f"API returned status {resp.status_code}: {resp.text}")
                else:
                    result = resp.json()
                    label = result.get("label", "unknown").upper()
                    is_fake = result.get("is_fake", False)
                    confidence = result.get("confidence", 0.0)
                    fake_score = result.get("fake_score", 0.0)
                    explanation = result.get("explanation", "")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", "FAKE" if is_fake else "GENUINE")
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")

                    st.write(f"**Fake score:** {fake_score:.3f}")
                    st.write(f"**Model latency (client‑side):** {latency_ms:.1f} ms")
                    if explanation:
                        st.write("**Why:**")
                        st.write(explanation)

st.markdown(
    """
---
_Model: fine‑tuned DistilBERT, INT8‑quantized and served via ONNX Runtime  
FP32: 12.99 ms → ONNX: 7.27 ms (≈1.8× faster for better UX)._
"""
)
