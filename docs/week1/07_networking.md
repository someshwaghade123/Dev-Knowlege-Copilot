# 🌐 Networking & Connection Guide

If your mobile app is stuck in a "Loading" state or says "Backend unreachable", follow these steps.

---

## 1. Bind Backend to your Local Network

By default, servers often run on `127.0.0.1` (localhost), which is **invisible** to your phone. You must tell it to listen on all interfaces.

**Run this command to start your backend:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
> The `--host 0.0.0.0` is the critical part.

---

## 2. Find your PC's Local IP Address

Your phone needs to know your PC's specific address on your Wi-Fi network.

1. Open a terminal on your PC.
2. Type `ipconfig` and press Enter.
3. Look for **IPv4 Address** under your Wi-Fi adapter (e.g., `192.168.1.5`).

---

## 3. Update the Mobile App

Open `mobile/services/api.ts` and update the `BASE_URL`:

```typescript
// Replace with YOUR PC's IP address found in Step 2
const BASE_URL = "http://192.168.1.5:8000/api/v1"; 
```

**Note**: Both your PC and your Phone **MUST** be on the same Wi-Fi network.

---

## 4. Fix a Corrupted Index (The "45-byte" Issue)

If your answer is always "I don't know" or the server crashes on query, your FAISS index might be corrupted. 

**Follow these steps to reset:**

1. Stop the backend server.
2. Delete the old index files:
   ```powershell
   Remove-Item data/faiss_index.bin
   Remove-Item data/metadata.db
   ```
3. Run the ingestion script again:
   ```powershell
   python scripts/ingest_docs.py --source data/sample_docs
   ```
4. Verify `data/faiss_index.bin` is now larger than 1KB.
5. Restart the backend.

---

## 5. Firewall Issues (Windows)

If it *still* won't connect, your Windows Firewall might be blocking port 8000.
1. Search for "Windows Defender Firewall with Advanced Security".
2. Create a new **Inbound Rule**.
3. Choose **Port** -> **TCP** -> **8000** -> **Allow the connection**.
