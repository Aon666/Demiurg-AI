import streamlit as st
import requests

st.set_page_config(
    page_title="Demiurg-AI",
    page_icon="🗺️",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/ask"

st.title("🗺️ Demiurg-AI")
st.caption("Dein lokaler TTRPG-Worldbuilding-Assistent (RAG)")

with st.sidebar:
    st.header("⚙️ Einstellungen & Filter")
    st.markdown("Hier kannst du später dynamische Filter (z.B. Regionen, Tags) hinzufügen.")

    use_region_filter = st.checkbox("Nach Region filtern")
    region_value = None
    if use_region_filter:
        region_value = st.selectbox("Region wählen", ["North", "South", "East", "West"])#TODO Regionen sind zu generisch, müssen sich dem Datensatz anpassen

    top_k = st.slider("Anzahl Chunks (Top-K)", min_value=1, max_value=5, value=2)

user_question = st.text_input("Stelle eine Frage an deine Spielwelt:", placeholder="Z.B. Wo kann man Mithril abbauen?")

if st.button("Fragen", type="primary"):
    if not user_question.strip():
        st.warning("Bitte gib eine Frage ein.")
    else:
        payload = {
            "question": user_question,
            "top_k": top_k
        }

    if use_region_filter and region_value:
        payload["filters"] = {"region": region_value}

    with st.spinner("Demiurg-AI durchsucht die Archive..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()

                st.subheader("Antwort")
                st.write(data["answer"])

                with st.expander("Gefundene Quellen (Retrieval)"):
                    sources = data.get("sources", [])
                    if sources:
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. Datei:** `{src['file_path']}` `[{src['header_path']}]` *(Distanz: {src['distance']:.4f})*")
                    else:
                        st.info("Keine spezifische Quellen übergeben.")

            else:
                st.error(f"Fehler com Server: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Konnte keine Verbindung zum FastAPI-Backend herstellen. Läuft `phase4_api.py`?")#TODO app namen anpassen, kann nicht phase 4 heißen