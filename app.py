import streamlit as st
import cv2
import pandas as pd
import tempfile

st.title("FRC Match Scouting Tool")

uploaded_video = st.file_uploader("Upload match video", type=["mp4", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.write(f"Total frames detected: {frame_count}")

    data = {
        "Metric": ["Frames"],
        "Value": [frame_count]
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="scouting_output.csv"
    )
