# FinShield — AI Financial Security Portal

## Run in 3 steps:
```
pip install -r requirements.txt
python ml/train.py        ← Run ONCE to train ML models (~30 sec)
python app.py             ← Start the app
```
Open: http://localhost:5000
Login: demo@finshield.com / demo123

## Pages
- /dashboard       — Security overview, live transaction feed, ML model banner
- /transactions    — All transactions with XAI breakdown on click
- /alerts-page     — Security alerts
- /threat-intel    — Live threat feed, 2FA toggle, spending insights (NEW)
- /phishing        — Phishing email/URL scanner
- /devices         — Device trust manager
- /security-report — Printable security health report (NEW)
- /education       — Security quiz with 12 shuffled questions (NEW)
