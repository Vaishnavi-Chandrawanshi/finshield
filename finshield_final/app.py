"""
FinShield — AI-Powered Financial Security Portal
Flask + SQLite + scikit-learn Random Forest
Run: pip install -r requirements.txt
     python ml/train.py   (first time only)
     python app.py
"""
import os, sqlite3, hashlib, uuid, json, random
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, session, redirect
from flask_cors import CORS
from ml.detector import Detector
from ml.data_generator import generate_transactions

app = Flask(__name__)
app.secret_key = "finshield_2025_secret"
CORS(app)
detector = Detector()

# ── DB ────────────────────────────────────────────────────────────────
def db():
    c = sqlite3.connect("finshield.db")
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE, password_hash TEXT,
            account_number TEXT UNIQUE, balance REAL DEFAULT 52400.0,
            security_score INTEGER DEFAULT 74,
            account_frozen INTEGER DEFAULT 0,
            two_fa_enabled INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, txn_id TEXT UNIQUE, type TEXT DEFAULT 'purchase',
            amount REAL, merchant TEXT, category TEXT, location TEXT, device TEXT,
            risk_score REAL DEFAULT 0.0, risk_level TEXT DEFAULT 'LOW',
            flagged INTEGER DEFAULT 0, status TEXT DEFAULT 'completed',
            explanation TEXT, shap_values TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, alert_type TEXT, title TEXT, message TEXT,
            severity TEXT DEFAULT 'medium', is_read INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, device_name TEXT, device_type TEXT,
            last_seen TEXT, is_trusted INTEGER DEFAULT 0, location TEXT
        );
    """)
    c.commit()

    pw = hashlib.sha256("demo123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (name,email,password_hash,account_number,balance,security_score) VALUES (?,?,?,?,?,?)",
              ("Alex Johnson","demo@finshield.com",pw,"ACC-4829-1023",52400.0,74))
    c.commit()

    u = c.execute("SELECT id FROM users WHERE email=?",("demo@finshield.com",)).fetchone()
    if u:
        uid = u["id"]
        if c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=?",(uid,)).fetchone()["n"] == 0:
            for t in generate_transactions(detector, 30):
                c.execute("""INSERT OR IGNORE INTO transactions
                    (user_id,txn_id,type,amount,merchant,category,location,device,
                     risk_score,risk_level,flagged,status,explanation,shap_values,timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (uid,t["txn_id"],t["type"],t["amount"],t["merchant"],t["category"],
                     t["location"],t["device"],t["risk_score"],t["risk_level"],t["flagged"],
                     t["status"],t["explanation"],t["shap_values"],t["timestamp"]))

            alerts_data = [
                ("fraud","🚨 Fraud Attempt Blocked","A high-risk wire to an offshore account was automatically blocked. Risk score: 93/100.","critical",1),
                ("login","⚠️ Unrecognised Device Login","Login attempted from unrecognised device in Moscow, Russia. Access was denied.","high",2),
                ("phishing","📧 Phishing Email Warning","A phishing email impersonating FinShield was detected. Do not click any links.","medium",3),
                ("tip","🔐 Enable 2FA","Two-factor authentication is not enabled. Enabling adds +20 points to your security score.","low",5),
            ]
            for atype,title,msg,sev,days in alerts_data:
                ts = (datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO alerts (user_id,alert_type,title,message,severity,created_at) VALUES (?,?,?,?,?,?)",
                          (uid,atype,title,msg,sev,ts))

            for name,dtype,loc,trusted in [("iPhone 14 Pro","mobile","New York, NY",1),("MacBook Air","laptop","New York, NY",1),("Unknown Android","mobile","Moscow, Russia",0)]:
                c.execute("INSERT INTO devices (user_id,device_name,device_type,last_seen,is_trusted,location) VALUES (?,?,?,?,?,?)",
                          (uid,name,dtype,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),trusted,loc))
            c.commit()
    c.close()

# ── Page routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect("/dashboard") if "user_id" in session else render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html") if "user_id" in session else redirect("/")

@app.route("/transactions")
def transactions_page():
    return render_template("transactions.html") if "user_id" in session else redirect("/")

@app.route("/alerts-page")
def alerts_page():
    return render_template("alerts.html") if "user_id" in session else redirect("/")

@app.route("/phishing")
def phishing_page():
    return render_template("phishing.html") if "user_id" in session else redirect("/")

@app.route("/devices")
def devices_page():
    return render_template("devices.html") if "user_id" in session else redirect("/")

@app.route("/education")
def education_page():
    return render_template("education.html") if "user_id" in session else redirect("/")

@app.route("/threat-intel")
def threat_intel_page():
    return render_template("threat_intel.html") if "user_id" in session else redirect("/")

@app.route("/security-report")
def security_report_page():
    return render_template("security_report.html") if "user_id" in session else redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ── Auth ──────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    d = request.json
    pw = hashlib.sha256(d.get("password","").encode()).hexdigest()
    c = db()
    u = c.execute("SELECT * FROM users WHERE email=? AND password_hash=?",(d.get("email",""),pw)).fetchone()
    c.close()
    if u:
        session["user_id"] = u["id"]
        session["name"]    = u["name"]
        return jsonify({"success":True,"redirect":"/dashboard"})
    return jsonify({"success":False,"message":"Invalid credentials"}), 401

# ── API ───────────────────────────────────────────────────────────────
@app.route("/api/user")
def api_user():
    if "user_id" not in session: return jsonify({}), 401
    c = db()
    u = dict(c.execute("SELECT id,name,email,account_number,balance,security_score,account_frozen,two_fa_enabled FROM users WHERE id=?",(session["user_id"],)).fetchone())
    u["unread_alerts"] = c.execute("SELECT COUNT(*) as n FROM alerts WHERE user_id=? AND is_read=0",(session["user_id"],)).fetchone()["n"]
    c.close()
    return jsonify(u)

@app.route("/api/transactions")
def api_transactions():
    if "user_id" not in session: return jsonify([]), 401
    c = db()
    rows = c.execute("SELECT * FROM transactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 60",(session["user_id"],)).fetchall()
    c.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/alerts")
def api_alerts():
    if "user_id" not in session: return jsonify([]), 401
    c = db()
    rows = c.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC",(session["user_id"],)).fetchall()
    c.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/alerts/read/<int:aid>", methods=["POST"])
def mark_read(aid):
    if "user_id" not in session: return jsonify({}), 401
    c = db()
    c.execute("UPDATE alerts SET is_read=1 WHERE id=? AND user_id=?",(aid,session["user_id"]))
    c.commit(); c.close()
    return jsonify({"ok":True})

@app.route("/api/devices")
def api_devices():
    if "user_id" not in session: return jsonify([]), 401
    c = db()
    rows = c.execute("SELECT * FROM devices WHERE user_id=?",(session["user_id"],)).fetchall()
    c.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/devices/trust/<int:did>", methods=["POST"])
def trust_device(did):
    if "user_id" not in session: return jsonify({}), 401
    trust = 1 if request.json.get("trust") else 0
    c = db()
    c.execute("UPDATE devices SET is_trusted=? WHERE id=? AND user_id=?",(trust,did,session["user_id"]))
    c.commit(); c.close()
    return jsonify({"ok":True})

@app.route("/api/stats")
def api_stats():
    if "user_id" not in session: return jsonify({}), 401
    c = db(); uid = session["user_id"]
    total   = c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=?",(uid,)).fetchone()["n"]
    flagged = c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND flagged=1",(uid,)).fetchone()["n"]
    blocked = c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND status='blocked'",(uid,)).fetchone()["n"]

    trend = []
    seeds = [18,7,24,35,12,88,15]
    for i in range(6,-1,-1):
        d = (datetime.now()-timedelta(days=i)).strftime("%Y-%m-%d")
        r = c.execute("SELECT COUNT(*) as cnt, AVG(risk_score) as avg FROM transactions WHERE user_id=? AND DATE(timestamp)=?",(uid,d)).fetchone()
        avg = round((r["avg"] or 0)*100, 1)
        if avg < 3: avg = seeds[i % 7]
        trend.append({"date":d,"count":r["cnt"],"avg_risk":max(avg, 10)})

    heatmap = {}
    for i in range(30):
        d = (datetime.now()-timedelta(days=i)).strftime("%Y-%m-%d")
        r = c.execute("SELECT COUNT(*) as cnt, MAX(risk_score) as mx FROM transactions WHERE user_id=? AND DATE(timestamp)=?",(uid,d)).fetchone()
        heatmap[d] = {"count":r["cnt"],"max_risk":round((r["mx"] or 0)*100,1)}

    breakdown = {
        "safe":     c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND risk_level='LOW'",(uid,)).fetchone()["n"],
        "medium":   c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND risk_level='MEDIUM'",(uid,)).fetchone()["n"],
        "high":     c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND risk_level='HIGH'",(uid,)).fetchone()["n"],
        "critical": c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND risk_level='CRITICAL'",(uid,)).fetchone()["n"],
    }
    c.close()
    return jsonify({"total":total,"flagged":flagged,"blocked":blocked,"trend":trend,"heatmap":heatmap,"breakdown":breakdown})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "user_id" not in session: return jsonify({}), 401
    d = request.json
    features = {
        "amount":          float(d.get("amount",0)),
        "merchant":        d.get("merchant",""),
        "location":        d.get("location",""),
        "device":          d.get("device",""),
        "hour":            datetime.now().hour,
        "is_foreign":      1 if any(w in d.get("location","") for w in ["Nigeria","Russia","Cayman","Offshore","Unknown","Iran"]) else 0,
        "is_vpn":          1 if "VPN" in d.get("device","") or "Unknown" in d.get("device","") else 0,
        "is_new_merchant": 1 if d.get("is_new_merchant") else 0,
    }
    rs, exp, shap = detector.score(features)
    rl = "CRITICAL" if rs>0.80 else "HIGH" if rs>0.60 else "MEDIUM" if rs>0.35 else "LOW"
    tid = f"TXN-{str(uuid.uuid4())[:8].upper()}"
    c = db()
    c.execute("""INSERT OR IGNORE INTO transactions
        (user_id,txn_id,type,amount,merchant,location,device,risk_score,risk_level,flagged,status,explanation,shap_values)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session["user_id"],tid,"purchase",features["amount"],features["merchant"],
         features["location"],features["device"],round(rs,4),rl,
         1 if rs>0.60 else 0,"blocked" if rs>0.80 else "completed",exp,json.dumps(shap)))
    c.commit(); c.close()
    color = "#f43f5e" if rs>0.80 else "#fb923c" if rs>0.60 else "#fbbf24" if rs>0.35 else "#34d399"
    return jsonify({"txn_id":tid,"risk_score":round(rs*100,1),"risk_level":rl,
                    "explanation":exp,"shap":shap,
                    "action":"BLOCKED" if rs>0.80 else "FLAGGED" if rs>0.60 else "APPROVED",
                    "color":color})

@app.route("/api/check-phishing", methods=["POST"])
def api_phishing():
    if "user_id" not in session: return jsonify({}), 401
    text = request.json.get("text","")
    score, indicators, verdict = detector.check_phishing(text)
    color = "#f43f5e" if score>0.65 else "#fb923c" if score>0.38 else "#34d399"
    return jsonify({"score":round(score*100,1),"verdict":verdict,"indicators":indicators,
                    "color":color,"safe":score<0.38})

@app.route("/api/freeze", methods=["POST"])
def api_freeze():
    if "user_id" not in session: return jsonify({}), 401
    freeze = 1 if request.json.get("freeze") else 0
    c = db()
    c.execute("UPDATE users SET account_frozen=? WHERE id=?",(freeze,session["user_id"]))
    c.commit(); c.close()
    return jsonify({"ok":True,"frozen":bool(freeze)})

@app.route("/api/security-score-breakdown")
def api_score():
    if "user_id" not in session: return jsonify({}), 401
    c = db(); uid = session["user_id"]
    u        = c.execute("SELECT security_score,two_fa_enabled FROM users WHERE id=?",(uid,)).fetchone()
    blocked  = c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND status='blocked'",(uid,)).fetchone()["n"]
    flagged  = c.execute("SELECT COUNT(*) as n FROM transactions WHERE user_id=? AND flagged=1",(uid,)).fetchone()["n"]
    untrusted= c.execute("SELECT COUNT(*) as n FROM devices WHERE user_id=? AND is_trusted=0",(uid,)).fetchone()["n"]
    c.close()
    factors = [
        {"name":"Two-Factor Auth",         "score":20 if u["two_fa_enabled"] else 0,        "max":20,"tip":"Enable 2FA for +20 pts"},
        {"name":"No Blocked Transactions", "score":max(0,25-blocked*5),                      "max":25,"tip":f"{blocked} blocked recently"},
        {"name":"Device Trust",            "score":15 if untrusted==0 else max(0,15-untrusted*5),"max":15,"tip":f"{untrusted} untrusted device(s)"},
        {"name":"Low Fraud Alerts",        "score":max(0,20-flagged*2),                      "max":20,"tip":f"{flagged} flagged recently"},
        {"name":"Account Activity",        "score":19,                                        "max":20,"tip":"Monitoring active"},
    ]
    return jsonify({"total":sum(f["score"] for f in factors),"factors":factors})

@app.route("/api/anomalies")
def api_anomalies():
    if "user_id" not in session: return jsonify([]), 401
    c = db(); uid = session["user_id"]
    rows = c.execute("SELECT category, AVG(amount) as avg, MAX(amount) as mx FROM transactions WHERE user_id=? AND status='completed' GROUP BY category",(uid,)).fetchall()
    c.close()
    return jsonify([{"category":r["category"],"avg_spend":round(r["avg"],2),"max_spend":round(r["mx"],2),
                     "deviation":round((r["mx"]/r["avg"]-1)*100,0),
                     "message":f"Unusual {r['category']} spend: ${r['mx']:.0f} vs avg ${r['avg']:.0f}"}
                    for r in rows if r["mx"] > r["avg"]*2.5 and r["mx"] > 200])

@app.route("/api/model-info")
def api_model_info():
    return jsonify(detector.get_model_info())

@app.route("/api/spending-insights")
def api_spending():
    if "user_id" not in session: return jsonify([]), 401
    c = db(); uid = session["user_id"]
    rows = c.execute("SELECT category, COUNT(*) as count, SUM(amount) as total, AVG(amount) as avg FROM transactions WHERE user_id=? AND status='completed' GROUP BY category ORDER BY total DESC",(uid,)).fetchall()
    c.close()
    total_spend = sum(r["total"] for r in rows) or 1
    return jsonify([{"category":r["category"],"count":r["count"],"total":round(r["total"],2),"avg":round(r["avg"],2),"pct":round(r["total"]/total_spend*100,1)} for r in rows])

@app.route("/api/threat-intel")
def api_threat_intel():
    return jsonify([
        {"type":"BEC","title":"CEO Wire Fraud Surge","desc":"500% increase in CEO impersonation wire fraud. Average loss: $137,000 per incident.","severity":"critical","source":"FinCEN 2025","date":"2025-03-15"},
        {"type":"Phishing","title":"Tax Season Phishing Campaign","desc":"IRS impersonation emails requesting SSN and bank details via fake refund forms.","severity":"high","source":"IRS Criminal Investigation","date":"2025-03-12"},
        {"type":"Malware","title":"Banking Trojan Grandoreiro","desc":"Grandoreiro malware targeting 1,500+ banks across 60 countries. Intercepts OTP codes.","severity":"critical","source":"IBM X-Force","date":"2025-03-10"},
        {"type":"Fraud","title":"Crypto Investment Scam Wave","desc":"Pig butchering scams via social media. Average victim loss exceeds $120,000.","severity":"high","source":"FTC","date":"2025-03-08"},
        {"type":"Deepfake","title":"AI Voice Cloning Fraud","desc":"Fraudsters using AI voice cloning to impersonate bank executives. 300% increase since 2024.","severity":"high","source":"Interpol","date":"2025-03-05"},
        {"type":"SIM Swap","title":"SIM Swap Attack Spike","desc":"Telecom employees bribed to swap SIM cards, bypassing 2FA on banking apps.","severity":"medium","source":"FBI IC3","date":"2025-03-01"},
    ])

@app.route("/api/toggle-2fa", methods=["POST"])
def api_toggle_2fa():
    if "user_id" not in session: return jsonify({}), 401
    c = db()
    cur = c.execute("SELECT two_fa_enabled FROM users WHERE id=?",(session["user_id"],)).fetchone()["two_fa_enabled"]
    new_val = 0 if cur else 1
    c.execute("UPDATE users SET two_fa_enabled=? WHERE id=?",(new_val,session["user_id"]))
    c.execute("UPDATE users SET security_score=MIN(100,MAX(0,security_score+?)) WHERE id=?",
              (20 if new_val else -20, session["user_id"]))
    c.commit(); c.close()
    return jsonify({"enabled":bool(new_val),"message":f"2FA {'enabled — security score +20' if new_val else 'disabled — security score -20'}."})

@app.route("/api/quiz-questions")
def api_quiz_questions():
    questions = [
        {"q":"A bank emails saying your account is suspended and asks you to click a link and enter your PIN. What should you do?","opts":["Click the link immediately","Call the number in the email","Go directly to the bank's official website or call their official number","Reply asking for verification"],"ans":2,"exp":"Always go directly to the bank's official website. Legitimate banks never ask for your PIN via email."},
        {"q":"You receive an alert for a $9,500 wire to an offshore account you don't recognise. Your first action?","opts":["Wait and see","Immediately freeze account and call your bank","Email your bank","Check next month"],"ans":1,"exp":"Freeze immediately, then contact your bank. Speed is critical — money can be recovered if you act fast."},
        {"q":"Which URL is most likely a phishing site?","opts":["https://chase.com/login","https://chase-secure-verify.tk/login","https://mobile.chase.com","https://online.chase.com"],"ans":1,"exp":".tk is a free domain used by scammers. Real Chase URLs always use chase.com — nothing else."},
        {"q":"What does a risk score of 87/100 mean?","opts":["Transaction is 87% complete","It is legitimate","AI detected strong fraud patterns","It took 87ms to process"],"ans":2,"exp":"Score above 80 = CRITICAL. Multiple high-risk signals detected. Transaction should be blocked."},
        {"q":"Which action increases your security score the most?","opts":["Changing password weekly","Enabling Two-Factor Authentication","Using mobile app","Having many transactions"],"ans":1,"exp":"2FA adds 20 points and prevents account takeover even if your password is stolen."},
        {"q":"An email says you won $1,000,000. Confirm bank details to receive payment. This is:","opts":["A real lottery win","Spam but harmless","An advance-fee scam — delete it","A government notification"],"ans":2,"exp":"Advance-fee scam. They request upfront fees then disappear. Never respond."},
        {"q":"What is behavioral biometrics in fraud detection?","opts":["Fingerprint scanning","Analysing how you type, scroll, and move your mouse to verify identity","Face recognition","Password strength analysis"],"ans":1,"exp":"Behavioral biometrics tracks unique interaction patterns. An attacker with stolen credentials behaves differently."},
        {"q":"You get an SMS: Your account is locked. Reply with OTP to unlock. What do you do?","opts":["Reply with OTP","Call your bank's official number directly","Ignore it","Forward to a friend"],"ans":1,"exp":"Never share OTPs via SMS. Call your bank's official number directly."},
        {"q":"What is XAI (Explainable AI) in fraud detection?","opts":["AI that explains why it flagged a transaction","AI that blocks all transactions","Machine learning only for banks","A government regulation"],"ans":0,"exp":"XAI shows exactly which factors caused a flag — amount, location, device — giving transparency into the AI decision."},
        {"q":"A caller says to transfer money to a safe account for protection. You should:","opts":["Transfer immediately","Ask for employee ID then transfer","Hang up and call your bank's official number","Transfer half to be safe"],"ans":2,"exp":"Safe account fraud is one of the most common scams. No legitimate bank will ever ask you to move money to protect it."},
        {"q":"What is SIM swapping?","opts":["Changing your SIM card yourself","Fraudsters convincing your carrier to transfer your number to their SIM to bypass 2FA","Sharing SIM with family","A phone upgrade process"],"ans":1,"exp":"SIM swap lets fraudsters receive your OTPs, bypassing 2FA on banking apps."},
        {"q":"Which is the safest way to receive 2FA codes?","opts":["SMS text message","Email","Authenticator app (Google Authenticator, Authy)","Phone call"],"ans":2,"exp":"Authenticator apps generate codes locally and cannot be intercepted by SIM swap attacks unlike SMS 2FA."},
    ]
    random.shuffle(questions)
    return jsonify(questions)

if __name__ == "__main__":
    init_db()
    info = detector.get_model_info()
    if not info["trained"]:
        print("\n" + "="*50)
        print("  Models not trained. Run: python ml/train.py")
        print("="*50 + "\n")
    app.run(debug=True, port=5000)
