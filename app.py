import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os


st.title("FRC Match Scouting Tool with Robot Tracking")

# Create folders to save outputs
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define team colors (BGR)
TEAM_COLORS = {
    "1732": (0,0,255),   # Red
    "948": (255,0,0),    # Blue
    # Add more teams here if needed
}

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

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for team, color_bgr in TEAM_COLORS.items():
            # Convert BGR to HSV
            color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            lower = np.array([max(color_hsv[0]-10,0),50,50])
            upper = np.array([min(color_hsv[0]+10,180),255,255])

            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
                in_hub = is_in_zone(cx, cy, ZONES["hub"])
                in_climb = is_in_zone(cx, cy, ZONES["climb"])

                all_data.append({
                    "team": team,
                    "frame": frame_idx,
                    "x": cx,
                    "y": cy,
                    "in_hub": in_hub,
                    "in_climb": in_climb
                })

    cap.release()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Summary per team
    summary = df.groupby("team").agg(
        total_frames=("frame","count"),
        hub_frames=("in_hub","sum"),
        climb_frames=("in_climb","sum")
    )
    summary["hub_ratio"] = summary["hub_frames"] / summary["total_frames"]
    summary["climb_ratio"] = summary["climb_frames"] / summary["total_frames"]

    # Display CSV in browser
    st.subheader("Team Summary")
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
    for team in TEAM_COLORS.keys():
        team_data = df[df["team"]==team]
        if not team_data.empty:
            plt.figure()
            plt.hist2d(team_data["x"], team_data["y"], bins=[50,30], cmap="Reds")
            plt.title(f"Robot {team} Heatmap")
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.gca().invert_yaxis()
            st.pyplot()
