# CHATBOT
WORKING CHATBOT USING INTERNAL TEXT DATA

## Handling the GROQ API Key

- Do not commit `.env` to Git.
- Add `.env` to `.gitignore` so local secret files are not tracked.
- For Streamlit deployment, set `GROQ_API_KEY` in Streamlit Secrets rather than publishing the key.

Example `secrets.toml` on Streamlit Cloud:

```toml
GROQ_API_KEY = "your-production-key"
```

The app will read from:
- `GROQ_API_KEY` environment variable
- `st.secrets["GROQ_API_KEY"]`

This keeps your key private and safe for deployment.

## Streamlit UI

Run the app with:

```bash
streamlit run streamlit_app.py
```

Then ask questions against `data.txt` in the browser.

