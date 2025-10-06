# Green O2 Exchange - Demo (Streamlit)

This is a small prototype for detecting partner branches and QR-coded cylinders in Jakarta.

Files:
- app.py : main Streamlit app
- requirements.txt : python deps
- data/branches.csv : sample branch locations in Jakarta
- data/qrcodes/* : sample QR images for demo tube IDs
- data/borrow_log.csv : log file where scans are appended

Run locally:
```
pip install -r requirements.txt
streamlit run app.py
```

To simulate scanning a QR code, open a browser to:
`http://localhost:8501/?tube=TUBE-Kuningan-001&branch=Kuningan`

For production, host the app on a public URL and encode that full URL in the QR code.
# Oksigen-QRCODE
# Oksigen-QRCODE
# Oksigen-QRCODE
# Oksigen-QRCODE
