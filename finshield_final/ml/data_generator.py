"""
FinShield Transaction Generator
Generates realistic transaction history using real statistical distributions.
Called once on first startup to populate the database.
"""
import random
import uuid
from datetime import datetime, timedelta


# Real merchant categories with realistic amounts (mean, std)
MERCHANT_PROFILES = [
    # (name, category, amount_mean, amount_std, locations, risk_base)
    ("Amazon",          "Shopping",   85,    120,  ["Online"],                    0.04),
    ("Walmart",         "Shopping",   67,    89,   ["Houston, TX","Dallas, TX","Phoenix, AZ","Chicago, IL"], 0.04),
    ("Starbucks",       "Food",       8,     4,    ["New York, NY","Seattle, WA","Boston, MA","Austin, TX"], 0.02),
    ("Target",          "Shopping",   89,    110,  ["Los Angeles, CA","Chicago, IL","Houston, TX"],           0.04),
    ("Netflix",         "Streaming",  16,    2,    ["Online"],                    0.02),
    ("Spotify",         "Streaming",  10,    1,    ["Online"],                    0.02),
    ("Uber",            "Transport",  22,    18,   ["San Francisco, CA","New York, NY","Chicago, IL"],       0.03),
    ("Delta Airlines",  "Travel",     380,   280,  ["Online"],                    0.06),
    ("Shell Gas",       "Fuel",       55,    30,   ["Houston, TX","Dallas, TX","Miami, FL"],                0.04),
    ("CVS Pharmacy",    "Health",     34,    28,   ["Boston, MA","New York, NY","Philadelphia, PA"],        0.03),
    ("Apple Store",     "Technology", 120,   180,  ["Online","San Francisco, CA"],                          0.05),
    ("Whole Foods",     "Groceries",  78,    55,   ["Austin, TX","Seattle, WA","Boston, MA"],               0.03),
    ("Home Depot",      "Home",       145,   200,  ["Atlanta, GA","Phoenix, AZ","Denver, CO"],              0.04),
    ("Venmo",           "Transfer",   85,    120,  ["Online"],                    0.06),
    ("McDonald's",      "Food",       12,    8,    ["Chicago, IL","New York, NY","Houston, TX"],            0.03),
    # High-risk merchants
    ("WIRE TRANSFER",   "Wire",       4500,  3800, ["Lagos, Nigeria","Unknown"],  0.88),
    ("OFFSHORE ACC",    "Wire",       9200,  4500, ["Cayman Islands","Offshore"], 0.93),
    ("Crypto Exchange", "Crypto",     3800,  2800, ["Online/VPN"],               0.82),
    ("UNKNOWN MERCHANT","Unknown",    1800,  2200, ["Moscow, Russia","Unknown"],  0.87),
    ("Foreign Casino",  "Gambling",   650,   500,  ["Online/VPN","Unknown"],     0.79),
]

DEVICES_LEGIT = ["iPhone App", "Android App", "Chrome/Windows", "Chrome/Mac",
                  "Safari/iPhone", "Desktop App", "Debit Card"]
DEVICES_FRAUD = ["Unknown Device", "VPN/Unknown", "Tor Browser", "Spoofed Device"]


def generate_transactions(detector, n: int = 30) -> list:
    """
    Generate n realistic transactions scored by the ML model.
    Mix of legitimate and suspicious transactions.
    """
    transactions = []
    now = datetime.now()

    # Force some fraud examples for demo
    fraud_indices = {2, 5, 9, 14, 19, 24}  # positions that will be fraud merchants

    # Pick n merchants — ensure some fraud ones included
    selected = []
    fraud_merchants = [m for m in MERCHANT_PROFILES if m[5] > 0.75]
    legit_merchants = [m for m in MERCHANT_PROFILES if m[5] <= 0.30]

    # Guarantee 4-6 fraud transactions
    num_fraud = min(6, len(fraud_merchants))
    selected = (
        random.sample(fraud_merchants, num_fraud) +
        random.choices(legit_merchants, k=n - num_fraud)
    )
    random.shuffle(selected)

    for i, (merchant, category, amt_mean, amt_std, locs, risk_base) in enumerate(selected):
        # Realistic amount
        amount = abs(random.gauss(amt_mean, amt_std))
        amount = max(1.0, round(amount, 2))

        location = random.choice(locs)
        device   = random.choice(DEVICES_FRAUD if risk_base > 0.75 else DEVICES_LEGIT)

        # Spread over last 30 days
        days_ago = random.randint(0, 29)
        hours    = random.randint(0, 23)
        # Fraud more likely at night
        if risk_base > 0.75:
            hours = random.choices(range(24), weights=[3,3,4,5,4,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3])[0]

        ts = (now - timedelta(days=days_ago, hours=hours,
                               minutes=random.randint(0,59))).strftime("%Y-%m-%d %H:%M:%S")

        is_foreign = 1 if any(r in location for r in ["Nigeria","Russia","Cayman","Offshore","Unknown","Iran"]) else 0
        is_vpn     = 1 if any(d in device   for d in ["VPN","Unknown","Tor","Spoof"]) else 0

        features = {
            "amount":           amount,
            "merchant":         merchant,
            "location":         location,
            "device":           device,
            "hour":             hours,
            "is_foreign":       is_foreign,
            "is_vpn":           is_vpn,
            "is_new_merchant":  1 if "UNKNOWN" in merchant or "OFFSHORE" in merchant else 0,
        }

        risk_score, explanation, shap = detector.score(features)

        # Blend ML score with base risk for dramatic demo
        if risk_base > 0.75:
            risk_score = min(0.99, max(risk_score, risk_base * 0.90))
        else:
            risk_score = min(0.50, risk_score)

        risk_level = ("CRITICAL" if risk_score > 0.80 else
                       "HIGH"    if risk_score > 0.60 else
                       "MEDIUM"  if risk_score > 0.35 else "LOW")

        import json
        transactions.append({
            "txn_id":      f"TXN-{str(uuid.uuid4())[:8].upper()}",
            "type":        "purchase",
            "amount":      amount,
            "merchant":    merchant,
            "category":    category,
            "location":    location,
            "device":      device,
            "risk_score":  round(risk_score, 4),
            "risk_level":  risk_level,
            "flagged":     1 if risk_score > 0.60 else 0,
            "status":      "blocked" if risk_score > 0.80 else "completed",
            "explanation": explanation,
            "shap_values": json.dumps(shap),
            "timestamp":   ts,
        })

    # Sort newest first
    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
    return transactions
