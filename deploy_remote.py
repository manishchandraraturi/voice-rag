"""Upload project payload directly to EC2 and start Docker container."""
import os
import subprocess
import tarfile
import time

IP = "13.201.137.78"
KEY = "voicerag-key.pem"

print("=" * 60)
print(" Packaging deployment bundle...")
print("=" * 60)

bundle_path = "deploy_payload.tar.gz"
if os.path.exists(bundle_path):
    os.remove(bundle_path)

with tarfile.open(bundle_path, "w:gz") as tar:
    for item in ["api", "core", "web", "ingest", "bench", "eval", "Dockerfile", "pyproject.toml", "uv.lock", ".env"]:
        if os.path.exists(item):
            tar.add(item)
    # Add data/index and data/reports
    if os.path.exists("data/index"):
        tar.add("data/index")
    if os.path.exists("data/reports"):
        tar.add("data/reports")

size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
print(f"[OK] Bundle created: {size_mb:.2f} MB")

print(f"\nUploading bundle to {IP} via SCP...")
subprocess.run(
    f'scp -o StrictHostKeyChecking=no -i "{KEY}" "{bundle_path}" ubuntu@{IP}:/home/ubuntu/{bundle_path}',
    shell=True,
    check=True,
)
print("[OK] Upload complete!")

remote_script = """#!/bin/bash
set -e
sudo mkdir -p /app
sudo chown -R ubuntu:ubuntu /app
cd /app
tar -xzf /home/ubuntu/deploy_payload.tar.gz
sudo docker stop voice-rag-container || true
sudo docker rm voice-rag-container || true
sudo docker build -t voice-rag-api .
sudo docker run -d --name voice-rag-container -p 8000:8000 --env-file .env -v /app/data:/data --restart always voice-rag-api
sleep 3
sudo docker ps
"""

with open("setup_ec2.sh", "w", newline="\n") as f:
    f.write(remote_script)

print("Uploading setup script to EC2...")
subprocess.run(
    f'scp -o StrictHostKeyChecking=no -i "{KEY}" setup_ec2.sh ubuntu@{IP}:/home/ubuntu/setup_ec2.sh',
    shell=True,
    check=True,
)

print("\nBuilding Docker container on EC2 instance (this takes ~1-2 mins)...")
subprocess.run(
    f'ssh -o StrictHostKeyChecking=no -i "{KEY}" ubuntu@{IP} "bash /home/ubuntu/setup_ec2.sh"',
    shell=True,
    check=True,
)

print("\n" + "=" * 60)
print(" Verifying remote health...")
print("=" * 60)

for i in range(15):
    time.sleep(5)
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://{IP}:8000/health", timeout=5)
        if r.status == 200:
            print("\n[SUCCESS] AWS EC2 Health check passed!")
            print(r.read().decode())
            break
    except Exception as e:
        print(f"Waiting for server to start ({i+1}/15)... {e}")

