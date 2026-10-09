"""CI probe: can this runner reach realtor.com (HomeHarvest) and Redfin gis-csv? Uses a public landmark, never the subjects."""
import time, requests
from homeharvest import scrape_property
t = time.time()
try:
    d = scrape_property(location="Washington Square, San Francisco, CA 94133", listing_type="sold", radius=0.5,
                        past_days=365, extra_property_data=False, parallel=False, limit=500)
    print("homeharvest rows", len(d), round(time.time() - t, 1), "s")
except Exception as e:
    print("homeharvest FAIL", type(e).__name__, str(e)[:200])
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
poly = "-122.4160 37.7960,-122.4040 37.7960,-122.4040 37.8060,-122.4160 37.8060,-122.4160 37.7960"
try:
    r = requests.get("https://www.redfin.com/stingray/api/gis-csv", params={"al": 1, "num_homes": 350, "sold_within_days": 365,
                     "status": 9, "uipt": "2", "poly": poly, "v": 8}, headers={"User-Agent": UA, "Referer": "https://www.redfin.com/"}, timeout=60)
    print("redfin", r.status_code, r.text[:60].replace("\n", " | "), len(r.text.splitlines()), "lines")
except Exception as e:
    print("redfin FAIL", e)
for u in ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
          "https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012240&lang=PT"]:
    try:
        r = requests.get(u, timeout=60); print(u[:60], r.status_code, len(r.text))
    except Exception as e:
        print(u[:60], "FAIL", e)
