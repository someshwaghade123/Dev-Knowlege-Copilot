import httpx
import sys

print("Testing local upload...")
try:
    with open("test.txt", "w") as f:
        f.write("This is a test document.")

    # Try port 8001 first, then 8000 (just in case)
    for port in [8001, 8000]:
        try:
            with open("test.txt", "rb") as f:
                files = {"file": ("test.txt", f, "text/plain")}
                res = httpx.post(f"http://localhost:{port}/api/v1/documents/upload", files=files, timeout=60)
                print(f"Port {port} Response:", res.status_code, res.text)
        except Exception as e:
            print(f"Port {port} failed:", e)
except Exception as main_e:
    print(f"Script failed: {main_e}")
