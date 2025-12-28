#!/usr/bin/env python3
import subprocess
from scapy.all import rdpcap, TCP, UDP, IP
import numpy as np
import os
import csv
import argparse

CAPTURE_FILE = "capture.pcap"
INTERFACE = "enp0s3"
BURST_WINDOW = 1.0
HIST_BINS = [0, 200, 500, 1000, 1500, 2000]


def capture_traffic(duration):
    print(f"[+] Capturing traffic for {duration} seconds on {INTERFACE}...")
    subprocess.run([
        "sudo", "timeout", str(duration),
        "tcpdump", "-i", INTERFACE,
        "-w", CAPTURE_FILE
    ])
    print("[+] Capture complete")


def load_spotify_ips(ip_file):
    if not os.path.exists(ip_file):
        return []
    with open(ip_file) as f:
        return [line.strip() for line in f if line.strip()]


def extract_features(pcap_file, spotify_ips=[]):
    packets = rdpcap(pcap_file)

    timestamps = []
    pkt_sizes = []
    downlink_bytes = 0
    uplink_bytes = 0
    flows = set()

    for pkt in packets:
        if IP not in pkt:
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        if spotify_ips:
            if src_ip not in spotify_ips and dst_ip not in spotify_ips:
                continue

        pkt_time = float(pkt.time)
        pkt_len = len(pkt)

        timestamps.append(pkt_time)
        pkt_sizes.append(pkt_len)

        if spotify_ips and dst_ip in spotify_ips:
            downlink_bytes += pkt_len
        else:
            uplink_bytes += pkt_len

        if TCP in pkt:
            flows.add((src_ip, dst_ip, pkt[TCP].sport, pkt[TCP].dport, "TCP"))
        elif UDP in pkt:
            flows.add((src_ip, dst_ip, pkt[UDP].sport, pkt[UDP].dport, "UDP"))

    if len(timestamps) < 2:
        return {}

    timestamps.sort()
    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return {}

    # ---------- Burst detection ----------
    bursts = []
    window = []

    for t in timestamps:
        window = [x for x in window if t - x <= BURST_WINDOW]
        window.append(t)
        bursts.append(len(window))

    # ---------- Packet size histogram ----------
    hist, _ = np.histogram(pkt_sizes, bins=HIST_BINS)
    hist_pct = hist / hist.sum() if hist.sum() else hist

    # ---------- Inter-arrival times ----------
    inter_times = np.diff(timestamps)

    total_bytes = sum(pkt_sizes)
    total_packets = len(pkt_sizes)

    features = {
        # ✅ RATE-BASED (FIX)
        "pkt_rate": total_packets / duration,
        "bytes_per_sec": total_bytes / duration,

        # ✅ Behavioral features
        "num_flows": len(flows),
        "avg_burst": np.mean(bursts),
        "max_burst": np.max(bursts),
        "std_burst": np.std(bursts),
        "mean_inter_pkt": np.mean(inter_times),
        "std_inter_pkt": np.std(inter_times),
        "downlink_ratio": downlink_bytes / total_bytes if total_bytes else 0
    }

    for i, v in enumerate(hist_pct):
        features[f"pkt_size_bin_{i}"] = v

    return features


def save_features(features, label, genre, outfile="dataset.csv"):
    if not features:
        print("[-] No valid features extracted")
        return

    exists = os.path.exists(outfile)

    with open(outfile, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(features.keys()) + ["label", "genre"]
        )
        if not exists:
            writer.writeheader()

        row = features.copy()
        row["label"] = label
        row["genre"] = genre if genre else "NA"
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ips", required=True, help="Spotify IP list")
    parser.add_argument("--label", required=True, choices=["music", "podcast"])
    parser.add_argument("--genre", help="Genre if music")
    parser.add_argument("--duration", type=int, default=60)

    args = parser.parse_args()

    spotify_ips = load_spotify_ips(args.ips)

    capture_traffic(args.duration)
    features = extract_features(CAPTURE_FILE, spotify_ips)

    save_features(features, args.label, args.genre)
    print("[+] Features saved to dataset.csv")


if __name__ == "__main__":
    main()

