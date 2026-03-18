"""
FinShield Detector — Fraud + Phishing ML
Uses trained models when available, falls back to improved rules.
"""
import os, pickle, json, re
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

class Detector:
    def __init__(self):
        self.fraud_bundle = None
        self.phish_bundle = None
        self.encoders     = None
        self.model_meta   = {}
        self._load_models()

    def _load_models(self):
        try:
            with open(os.path.join(MODEL_DIR, "fraud_model.pkl"),    "rb") as f: self.fraud_bundle = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "phishing_model.pkl"), "rb") as f: self.phish_bundle = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "encoders.pkl"),       "rb") as f: self.encoders     = pickle.load(f)
            mp = os.path.join(MODEL_DIR, "metadata.json")
            if os.path.exists(mp):
                with open(mp) as f: self.model_meta = json.load(f)
            print(f"[Detector] ML models loaded — AUROC: {self.fraud_bundle.get('auroc','?')}")
        except FileNotFoundError:
            print("[Detector] Models not found — rule-based fallback. Run: python ml/train.py")

    def _encode(self, key, value):
        try:
            le = self.encoders[key]
            return le.transform([value])[0] if value in le.classes_ else len(le.classes_)
        except: return 0

    def score(self, features: dict) -> tuple:
        amount     = float(features.get("amount", 0))
        merchant   = str(features.get("merchant", ""))
        location   = str(features.get("location", ""))
        device     = str(features.get("device", ""))
        hour       = int(features.get("hour", 12))
        is_foreign = int(features.get("is_foreign", 0))
        is_vpn     = int(features.get("is_vpn", 0))
        is_new     = int(features.get("is_new_merchant", 0))
        is_high    = 1 if amount > 2000 else 0
        is_night   = 1 if 2 <= hour <= 5 else 0

        if self.fraud_bundle and self.encoders:
            return self._ml_score(amount, merchant, location, device, hour,
                                   is_foreign, is_vpn, is_new, is_high, is_night)
        return self._rule_score(amount, merchant, location, device, hour,
                                 is_foreign, is_vpn, is_new)

    def _ml_score(self, amount, merchant, location, device, hour,
                  is_foreign, is_vpn, is_new, is_high, is_night):
        model  = self.fraud_bundle["model"]
        scaler = self.fraud_bundle["scaler"]
        imp    = self.fraud_bundle.get("importances", {})

        X = np.array([[amount, hour, is_foreign, is_vpn, is_new, is_high, is_night,
                       self._encode("merchant", merchant),
                       self._encode("location", location),
                       self._encode("device",   device)]], dtype=np.float64)
        prob = float(model.predict_proba(scaler.transform(X))[0][1])

        raw = {"amount": amount/10000, "is_foreign": is_foreign, "is_vpn": is_vpn,
               "is_new_merchant": is_new, "is_high_amount": is_high, "is_night": is_night}
        shap = {}
        labels = {"amount":"Transaction Amount","is_foreign":"Foreign Location",
                  "is_vpn":"VPN / Unknown Device","is_new_merchant":"New Merchant",
                  "is_high_amount":"High Amount","is_night":"Unusual Hour (2–5 am)"}
        for f, w in sorted(imp.items(), key=lambda x: -x[1])[:6]:
            c = round(w * raw.get(f, 0) * prob, 4)
            if c > 0.001: shap[labels.get(f, f)] = c

        top = max(shap, key=shap.get) if shap else "multiple risk factors"
        if prob > 0.80: exp = f"BLOCKED — {top} is the primary driver. Fraud probability: {prob*100:.0f}%."
        elif prob > 0.60: exp = f"FLAGGED — {top} triggered review. Fraud probability: {prob*100:.0f}%."
        elif prob > 0.35: exp = f"CAUTION — Minor anomaly ({top}). Score: {prob*100:.0f}/100."
        else: exp = f"SAFE — No significant fraud indicators. Score: {prob*100:.0f}/100."
        return round(prob, 4), exp, shap

    def _rule_score(self, amount, merchant, location, device, hour, is_foreign, is_vpn, is_new):
        HIGH_RISK = ["nigeria","offshore","cayman","russia","iran","belarus","unknown","anonymous","dark web"]
        SUSP_MERCH = ["unknown","casino","offshore","wire transfer","crypto exchange","anonymous","gambling","unlicensed"]
        s = 0.02; shap = {}
        if amount > 5000:   s += 0.28; shap["High Amount"] = 0.28
        elif amount > 2000: s += 0.14; shap["Elevated Amount"] = 0.14
        if any(k in merchant.lower() for k in SUSP_MERCH): s += 0.32; shap["Suspicious Merchant"] = 0.32
        if any(l in location.lower() for l in HIGH_RISK):  s += 0.38; shap["High-Risk Location"] = 0.38
        elif is_foreign: s += 0.18; shap["International"] = 0.18
        if is_vpn or "unknown" in device.lower(): s += 0.24; shap["VPN/Unknown Device"] = 0.24
        if 2 <= hour <= 5: s += 0.12; shap["Unusual Hour"] = 0.12
        if is_new: s += 0.08; shap["New Merchant"] = 0.08
        s = max(0.01, min(0.99, s + np.random.uniform(-0.02, 0.02)))
        top = max(shap, key=shap.get) if shap else "pattern"
        if s > 0.80:   exp = f"BLOCKED (rules): {top}. Score: {s*100:.0f}/100."
        elif s > 0.60: exp = f"FLAGGED (rules): {top}. Score: {s*100:.0f}/100."
        elif s > 0.35: exp = f"CAUTION: {top}. Score: {s*100:.0f}/100."
        else:          exp = f"SAFE: No significant risk. Score: {s*100:.0f}/100."
        return round(s, 4), exp, shap

    def check_phishing(self, text: str) -> tuple:
        """
        Phishing detection. Uses trained ML model when available.
        Rule-based fallback is calibrated to avoid false positives on legitimate messages.
        """
        if self.phish_bundle:
            pipeline = self.phish_bundle["pipeline"]
            prob = float(pipeline.predict_proba([text])[0][1])
            tfidf = pipeline.named_steps["tfidf"]
            clf   = pipeline.named_steps["clf"]
            tv    = tfidf.transform([text])
            feats = tfidf.get_feature_names_out()
            coefs = clf.coef_[0]
            scores_arr = tv.toarray()[0] * coefs
            top_idx = np.argsort(scores_arr)[-5:][::-1]
            indicators = [f'Suspicious keyword: "{feats[i]}"'
                          for i in top_idx if scores_arr[i] > 0.01]
            if not indicators and prob > 0.7:
                indicators = ["Multiple phishing patterns detected by ML model"]
            verdict = "PHISHING DETECTED" if prob > 0.70 else "SUSPICIOUS" if prob > 0.40 else "LIKELY SAFE"
            return round(prob, 4), indicators[:5], verdict

        # Improved rule-based — calibrated to reduce false positives
        tl = text.lower()

        # Strong phishing signals (high weight)
        STRONG = [
            (r"(urgent|immediately).{0,40}(verify|confirm|click|suspend)", 0.35),
            (r"(account|password|pin|otp|cvv).{0,30}(enter|provide|confirm|reset)", 0.40),
            (r"http[s]?://[^\s]*\.(tk|ml|ga|cf|xyz|click|top|pw|cc)(/|\s|$)", 0.55),
            (r"(won|winner|prize|lottery|congratulations).{0,30}(million|thousand|\$[0-9])", 0.45),
            (r"(suspended|locked|limited).{0,30}(account|access|card)", 0.35),
            (r"(wire transfer|western union|moneygram).{0,30}(send|transfer|pay)", 0.40),
            (r"(dear customer|dear user|dear account holder)", 0.20),
            (r"(social security|ssn|national id).{0,20}(verify|confirm|provide|enter)", 0.50),
            (r"click here.{0,20}(to verify|to confirm|immediately|now)", 0.30),
            (r"(free money|free gift|free iphone|claim.{0,20}prize)", 0.30),
            (r"(paypal|amazon|apple|microsoft|google|bank of america|chase).{0,30}(suspended|locked|verify|alert)", 0.30),
        ]

        # Legitimacy signals — reduce score if present
        SAFE_SIGNALS = [
            (r"(govt of india|government of india|ministry|department of telecom)", -0.25),
            (r"(official website|visit our website|call us at [0-9])", -0.20),
            (r"(your monthly statement|direct deposit|balance update)", -0.20),
            (r"(unsubscribe|privacy policy|terms of service)", -0.15),
            (r"(jio\.|airtel\.|bsnl\.|trai\.)", -0.20),
            (r"(play store|app store|google play|ios)", -0.10),
        ]

        score = 0.0; indicators = []
        for pattern, weight in STRONG:
            if re.search(pattern, tl):
                score += weight
                readable = re.sub(r'[()\\?.*+]', '', pattern.split('(')[1].split(')')[0].split('|')[0]).strip()
                indicators.append(f'Pattern detected: "{readable}"')

        # Apply safe signal reductions
        for pattern, reduction in SAFE_SIGNALS:
            if re.search(pattern, tl):
                score += reduction  # reduction is negative

        # URL analysis
        urls = re.findall(r'https?://[^\s<>"]+', tl)
        safe_domains = ["amazon.com","google.com","apple.com","microsoft.com","paypal.com",
                        "jio.com","airtel.in","bsnl.in","india.gov.in","gov.in","nic.in","t.jio"]
        for url in urls:
            is_safe = any(sd in url for sd in safe_domains)
            if not is_safe:
                if re.search(r'\.(tk|ml|xyz|click|top|pw|ga|cf|cc)(/|$)', url):
                    score += 0.50; indicators.append(f"Suspicious domain in URL: {url[:45]}")
                elif re.search(r'(secure|verify|login|account|bank)[.-]', url):
                    score += 0.20; indicators.append(f"Suspicious URL pattern: {url[:45]}")

        score = max(0.0, min(0.99, score))
        # Only flag as phishing if score is high enough
        verdict = "PHISHING DETECTED" if score > 0.65 else "SUSPICIOUS" if score > 0.38 else "LIKELY SAFE"
        return round(score, 4), indicators[:5], verdict

    def get_model_info(self) -> dict:
        if self.fraud_bundle:
            return {"mode":"Random Forest ML","auroc":self.fraud_bundle.get("auroc","N/A"),
                    "f1":self.fraud_bundle.get("f1","N/A"),"training_samples":"50,000",
                    "trained":True,"phishing_auroc":self.phish_bundle.get("auroc","N/A") if self.phish_bundle else "N/A"}
        return {"mode":"Rule-Based (run ml/train.py)","trained":False,"auroc":"N/A","f1":"N/A","training_samples":"0"}
