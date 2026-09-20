#!/usr/bin/env python3
# scripts/test_esp8266_bridge.py
"""Test script for verifying physical ESP8266 honeypot connectivity and logging.

Usage:
    python scripts/test_esp8266_bridge.py [--ip 192.168.0.150]
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from securemesh_sce.environment.iot.hardware_bridge import PhysicalESP8266Bridge


def test_esp8266(device_ip: str):
    print(f"\n==================================================")
    print(f"  Testing ESP8266 Physical Honeypot at: {device_ip}")
    print(f"==================================================")

    bridge = PhysicalESP8266Bridge(device_ip=device_ip, port=80, timeout=3.0)

    # 1. Reachability Check
    print("\n[1/3] Checking if ESP8266 is reachable on network...")
    reachable = bridge.is_reachable()
    if reachable:
        print("  -> SUCCESS: ESP8266 web service is online!")
    else:
        print("  -> WARNING: Device not reachable at this IP.")
        print("     Ensure the ESP8266 is powered, connected to the same Wi-Fi,")
        print("     and check the Serial Monitor for the assigned IP address.")
        return

    # 2. Hardware Telemetry Check
    print("\n[2/3] Fetching device telemetry (/status)...")
    telemetry = bridge.get_hardware_telemetry()
    if telemetry and telemetry.get("reachable"):
        print(f"  -> Device      : {telemetry.get('device', 'ESP8266')}")
        print(f"  -> IP Address  : {telemetry.get('ip')}")
        print(f"  -> Free Heap   : {telemetry.get('free_heap')} bytes")
        print(f"  -> Uptime (ms) : {telemetry.get('uptime_ms')} ms")
        print(f"  -> Wi-Fi RSSI  : {telemetry.get('rssi')} dBm")
    else:
        print(f"  -> FAILED to parse telemetry: {telemetry}")

    # 3. Honeypot Attack Trap Probe
    print("\n[3/3] Sending simulated attack probe to /login trap...")
    probe_res = bridge.test_honeypot_login(username="admin", password="test_password_123")
    print(f"  -> Response HTTP Status: {probe_res.get('status_code')}")
    print(f"  -> Response Body       : {probe_res.get('response')}")
    print("\nCheck your FastAPI backend console to verify that the event was logged!")
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ESP8266 Honeypot Connectivity")
    parser.add_argument("--ip", type=str, default=None, help="ESP8266 IP address (defaults to ESP8266_DEVICE_IP in .env)")
    args = parser.parse_args()

    import os
    target_ip = args.ip or os.getenv("ESP8266_DEVICE_IP", "192.168.0.150")
    test_esp8266(target_ip)
