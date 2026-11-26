# 📒 Kharcha Paani

Log expenses the way you say them. Type or speak "chai 20, auto 60, Swiggy 250" (Hinglish works too), or snap a bill or UPI screenshot, and Gemini sorts each item into **Food**, **Travel** or **Misc**. Set a monthly budget and get a warning at 80% and when you go over.

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt` and `README.md` to a GitHub repo.
2. Go to https://share.streamlit.io, click **Create app**, then pick the repo, branch `main` and file `app.py`.
3. Under **Advanced settings → Secrets**, paste: `GEMINI_API_KEY = "your-key"`
4. Click **Deploy**.

Get a free key at https://aistudio.google.com/apikey.

## Good to know
- Data is saved to `data/<khata-name>.json` on the server. Streamlit Cloud wipes it whenever the app restarts, so use **Download khata** in the sidebar for backups.
- Anyone who knows a khata name can open it. This setup suits personal or demo use.
- If Gemini is busy, the app retries and then falls back to `gemini-flash-lite-latest`.
