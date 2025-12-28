#!/usr/bin/env python3
import subprocess
import joblib
import pandas as pd
import os
from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_FILE = "live_capture.pcap"
INTERFACE = "enp0s3"        # Change to your network interface
CAPTURE_DURATION = 60        # Capture time in seconds
SPOTIFY_IPS_FILE = "spotify_ips.txt"  # Optional, can be empty

# Load trained models
content_model = joblib.load("content_type_model.pkl")
genre_model = joblib.load("genre_model.pkl")

# -----------------------------
# Helper functions
# -----------------------------
def load_spotify_ips(ip_file):
    """Load Spotify IPs from a text file."""
    if not os.path.exists(ip_file):
        print(f"[!] Spotify IPs file '{ip_file}' not found, proceeding without filtering.")
        return []
    with open(ip_file) as f:
        return [line.strip() for line in f if line.strip()]

def capture_traffic(duration=CAPTURE_DURATION):
    """Capture network traffic using tcpdump for a fixed duration."""
    print(f"[+] Capturing traffic for {duration} seconds on {INTERFACE}...")
    subprocess.run([
        "sudo", "timeout", str(duration),
        "tcpdump", "-i", INTERFACE, "-w", CAPTURE_FILE
    ])
    if os.path.exists(CAPTURE_FILE):
        print(f"[+] Capture complete, saved to {CAPTURE_FILE}")
    else:
        print("[!] Capture failed or file not created.")

def extract_features(packets, spotify_ips=[]):
    """Extract new feature set from captured packets."""
    timestamps = []
    pkt_sizes = []
    downlink_bytes = 0
    flows = set()

    for pkt in packets:
        if IP not in pkt:
            continue
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if spotify_ips and src_ip not in spotify_ips and dst_ip not in spotify_ips:
            continue

        pkt_time = float(pkt.time)
        pkt_len = len(pkt)

        timestamps.append(pkt_time)
        pkt_sizes.append(pkt_len)

        if spotify_ips and dst_ip in spotify_ips:
            downlink_bytes += pkt_len

        if TCP in pkt:
            flows.add((src_ip, dst_ip, pkt[TCP].sport, pkt[TCP].dport, "TCP"))
        elif UDP in pkt:
            flows.add((src_ip, dst_ip, pkt[UDP].sport, pkt[UDP].dport, "UDP"))

    timestamps.sort()
    total_bytes = sum(pkt_sizes)
    duration = timestamps[-1] - timestamps[0] if timestamps else 1  # prevent div by 0

    # Burst detection
    BURST_WINDOW = 1.0
    bursts = []
    window = []
    for t in timestamps:
        window = [x for x in window if t - x <= BURST_WINDOW]
        window.append(t)
        bursts.append(len(window))

    # Packet size histogram
    HIST_BINS = [0, 200, 500, 1000, 1500, 2000]
    hist, _ = np.histogram(pkt_sizes, bins=HIST_BINS)
    hist_pct = hist / hist.sum() if hist.sum() else hist

    # Inter-packet timing
    inter_times = np.diff(timestamps) if len(timestamps) > 1 else [0]

    # ----------------
    # New Features
    # ----------------
    features = {
        "pkt_rate": len(pkt_sizes) / duration if duration else 0,
        "bytes_per_sec": total_bytes / duration if duration else 0,
        "num_flows": len(flows),
        "avg_burst_norm": np.mean(bursts) if bursts else 0,
        "max_burst_norm": max(bursts) if bursts else 0,
        "std_burst": np.std(bursts) if bursts else 0,
        "mean_inter_pkt": np.mean(inter_times) if len(inter_times) else 0,
        "std_inter_pkt": np.std(inter_times) if len(inter_times) else 0,
        "downlink_ratio": downlink_bytes / total_bytes if total_bytes else 0
    }

    # Add packet size histogram bins
    for i, v in enumerate(hist_pct):
        features[f"pkt_size_bin_{i}"] = v

    return features

def predict(features):
    """Predict content type and genre (if music)."""
    if not features:
        return {"content_type": "unknown", "genre": "unknown"}

    X = pd.DataFrame([features])
    content_type = content_model.predict(X)[0]
    result = {"content_type": content_type}

    if content_type.lower() == "music":
        genre = genre_model.predict(X)[0]
        result["genre"] = genre

    return result

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    spotify_ips = load_spotify_ips(SPOTIFY_IPS_FILE)

    # Capture live traffic
    capture_traffic(CAPTURE_DURATION)

    # Load packets
    if not os.path.exists(CAPTURE_FILE):
        print(f"[!] Capture file {CAPTURE_FILE} not found. Exiting.")
        exit(1)

    packets = rdpcap(CAPTURE_FILE)

    # Extract features and predict
    features = extract_features(packets, spotify_ips)
    prediction = predict(features)

    print("\n===== Prediction Result =====")
    for k, v in prediction.items():
        print(f"{k}: {v}")
