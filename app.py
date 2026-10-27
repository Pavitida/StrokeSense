import tempfile
from pathlib import Path

import streamlit as st
from src.fusion.run_pipeline import run_pipeline

st.set_page_config(page_title="StrokeSense", page_icon="🧠")
st.title("StrokeSense")
st.caption("Experimental AI Screening Prototype")
st.info("Research prototype only. Not a medical diagnosis.")

image = st.file_uploader("1. Upload face image", type=["jpg", "jpeg", "png"])
audio = st.file_uploader("2. Upload speech audio", type=["wav"])
video = st.file_uploader("3. Upload movement video", type=["mp4"])

if st.button("Analyze", type="primary"):
    if not all([image, audio, video]):
        st.warning("Please upload all three files.")
    else:
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for uploaded, name in [
                (image, "face.jpg"),
                (audio, "speech.wav"),
                (video, "movement.mp4"),
            ]:
                path = Path(folder) / name
                path.write_bytes(uploaded.getvalue())
                paths.append(str(path))

            try:
                with st.spinner("Analyzing..."):
                    result = run_pipeline(*paths)

                if result.get("status") == "success":
                    st.success("Analysis completed")
                    st.metric("Face score", f"{result['face']['score']:.3f}")
                    st.metric("Speech score", f"{result['speech']['score']:.3f}")
                    st.metric("Movement score", f"{result['movement']['score']:.3f}")
                    st.metric("Experimental combined score", f"{result['fusion']['score']:.3f}")
                    st.info(result["fusion"]["note"])
                else:
                    st.warning("Analysis incomplete")
                    st.write("Failed module:", result.get("failed_modality"))
                    st.write("Reason:", result.get("reason"))

            except Exception as error:
                st.error(f"Processing failed: {error}")
