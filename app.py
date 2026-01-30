import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os

st.title("FRC Match Scouting Tool with Robot Tracking")

# Create folder to save outputs
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define field zones as rectangles: x1, y1, x2, y2
ZONES = {
    "hub": (100, 50, 400, 300),
    "climb": (500, 100, 700, 300)
}

# Helper function to check if a point is in a zone
def is_in_zone(x, y, zone):
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

# Video uploader
uploaded_video = st.file_uploader("Upload match video", type=["mp4","mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total frames detected: {frame_count}")

    all_data = []

    first_frame = None

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = blur
            continue

        # Motion detection
        frame_delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt_idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            in_hub = is_in_zone(cx, cy, ZONES["hub"])
            in_climb = is_in_zone(cx, cy, ZONES["climb"])

            all_data.append({
                "team": f"robot_{cnt_idx+1}",
                "frame": frame_idx,
                "x": cx,
                "y": cy,
                "in_hub": in_hub,
                "in_climb": in_climb
            })

    cap.release()

    if not all_data:
        st.warning("No motion detected in this video. Try a different video or adjust lighting.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Summary per robot
        summary = df.groupby("team").agg(
            total_frames=("frame","count"),
            hub_frames=("in_hub","sum"),
            climb_frames=("in_climb","sum")
        )
        summary["hub_ratio"] = summary["hub_frames"] / summary["total_frames"]
        summary["climb_ratio"] = summary["climb_frames"] / summary["total_frames"]

        # Display CSV in browser
        st.subheader("Robot Summary")
        st.dataframe(summary)

        # Download CSV
        csv_path = os.path.join(OUTPUT_FOLDER, "scouting_summary.csv")
        summary.to_csv(csv_path)
        st.download_button(
            "Download CSV",
            summary.to_csv(index=True),
            file_name="scouting_summary.csv"
        )

        # Generate heatmaps
        st.subheader("Robot Heatmaps")
        robot_names = df["team"].unique()
        for robot in robot_names:
            robot_data = df[df["team"]==robot]
            plt.figure()
            plt.hist2d(robot_data["x"], robot_data["y"], bins=[50,30], cmap="Reds")
            plt.title(f"{robot} Heatmap")
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.gca().invert_yaxis()
            st.pyplot()
