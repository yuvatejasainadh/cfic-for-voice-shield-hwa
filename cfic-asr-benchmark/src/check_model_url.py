"""Check object store URL and download status."""

import urllib.request

url = "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
req = urllib.request.Request(url, method="HEAD")
try:
    with urllib.request.urlopen(req) as res:
        print("Status:", res.status)
        content_length = int(res.headers.get("Content-Length", 0))
        print("Content-Length (bytes):", content_length)
        print(f"Content-Length (GB): {content_length / (1024**3):.2f} GB")
except Exception as e:
    print(f"Error connecting to {url}: {e}")
