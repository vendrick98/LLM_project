import google.generativeai as genai
from datetime import datetime
import spacy

# Inserisci qui la tua chiave API
genai.configure(api_key="AIzaSyDwmSWRSdTmxzmTtd67aT7xTo3TgZrSTPs")

# Definisci le impostazioni di sicurezza desiderate
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

# Inizializza il modello GenerativeModel con le impostazioni di sicurezza
model = genai.GenerativeModel(model_name='gemini-2.0-flash', safety_settings=safety_settings)

npc_context = """Sei un uomo sulla sessantina di nome Giovanni, un gioielliere che ha lavorato per 40 anni nel suo negozio "L'Arte dell'Oro" in una tranquilla via del centro di Milano. Quando rispondi, fornisci solo il tuo dialogo diretto come se stessi parlando con un cliente. Evita descrizioni dei tuoi pensieri, delle tue azioni non verbali o del tuo stato d'animo. Concentrati unicamente sulle tue battute verbali."""

# Inizializza la chat includendo il contesto come primo messaggio "user"
initial_user_message_for_context = f"Agisci come se fossi: {npc_context}"
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": [initial_user_message_for_context]
    }
])

# Nome del file
filename = f"conversazione_nlp_Giovanni_flash_2_0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Parametri per il riassunto
SUMMARY_FREQUENCY = 5
NUM_MESSAGES_TO_KEEP = 3

conversation_history_for_summary = []
summary = ""
important_keywords = set()

# Carica il modello italiano piccolo di spaCy
try:
    nlp = spacy.load("it_core_news_sm")
except OSError:
    print("Downloading spaCy Italian model...")
    spacy.cli.download("it_core_news_sm")
    nlp = spacy.load("it_core_news_sm")

def extract_important_keywords_nlp(text):
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]:
            if not token.is_stop:
                keywords.add(token.lemma_)
        if token.ent_type_:
            keywords.add(token.text)
    return list(keywords)

try:
    with open("ChatTestWithNLP/" + filename, "w", encoding="utf-8") as f:
        f.write("[CONTEXT]\n")
        f.write(npc_context + "\n")
        f.write("[/CONTEXT]\n\n")
        f.write("[INIZIO CONVERSAZIONE]\n")
        f.write(f"[INIZIO_CONTESTO_COME_USER]: {initial_user_message_for_context}\n")

        print(f"Conversazione salvata nel file: {filename}")
        print("Inizia a parlare con Giovanni (digita 'esci' per terminare):")

        while True:
            user_input = input("Tu: ")
            if user_input.lower() == "esci":
                f.write("[FINE CONVERSAZIONE]\n")
                break

            f.write(f"[USER]: {user_input}\n")
            conversation_history_for_summary.append({"role": "user", "content": user_input})
            inizioRichiesta = datetime.now()
            try:
                important_keywords_user = extract_important_keywords_nlp(user_input)
                important_keywords.update(important_keywords_user)

                response = chat.send_message(user_input)
                npc_response = response.text
                f.write(f"[NPC]: {npc_response}\n")
                conversation_history_for_summary.append({"role": "assistant", "content": npc_response})
                inizioExtraction = datetime.now()
                important_keywords_npc = extract_important_keywords_nlp(npc_response)
                print("Tempo di estrazione parole chiave:", datetime.now() - inizioExtraction)
                f.write(f"[ESTRAZIONE PAROLE CHIAVE]: {important_keywords_npc}\n")
                important_keywords.update(important_keywords_npc)

                if len(conversation_history_for_summary) > 0 and len(conversation_history_for_summary) % (SUMMARY_FREQUENCY * 2) == 0:
                    print("Generazione del riassunto...")
                    f.write("[GENERAZIONE RIASSUNTO]\n")
                    inizioRiassunto = datetime.now()
                    summary_prompt = f"""Riassumi la seguente conversazione tra un utente e un gioielliere di nome Giovanni, mantenendo le informazioni chiave, i dettagli importanti per il contesto futuro e  eventuali oggetti o preferenze discusse.
                    Parole chiave rilevanti finora: {list(important_keywords)}
                    Assicurati che il riassunto sia conciso ma completo e che non si perdano dettagli cruciali per la comprensione della conversazione."""
                    summary_response = model.generate_content(summary_prompt)
                    summary_text = summary_response.text
                    f.write (f"[Fine riassunto time]: {datetime.now () - inizioRiassunto}\n")
                    print (f"[Fine riassunto time]: {datetime.now () - inizioRiassunto}\n")
                    if summary_text:
                        summary = f"Riassunto della conversazione precedente: {summary_text}\nParole chiave rilevanti: {list(important_keywords)}\n"
                        f.write(f"[RIASSUNTO]:\n{summary}\n[/RIASSUNTO]\n")
                        new_history = [{"role": "user", "parts": [initial_user_message_for_context]}]
                        new_history.append({"role": "user", "parts": [summary]})

                        last_n_messages = conversation_history_for_summary[-(NUM_MESSAGES_TO_KEEP * 2):]
                        for msg in last_n_messages:
                            new_history.append({"role": msg['role'], "parts": [msg['content']]})

                        chat = model.start_chat(history=new_history)
                        conversation_history_for_summary = last_n_messages.copy()
                        important_keywords = set() # Resetta le parole chiave dopo il riassunto per concentrarsi sui nuovi temi
                        print("Riassunto generato e cronologia aggiornata.")
                        f.write("[CRONOLOGIA AGGIORNATA CON RIASSUNTO]\n")

                    else:
                        print("Errore nella generazione del riassunto.")
                        f.write("[ERRORE GENERAZIONE RIASSUNTO]\n")
                f.write (f"[Fine Richiesta]: {datetime.now () - inizioRichiesta}\n")
                print (f"[Fine Richiesta]: {datetime.now () - inizioRichiesta}\n")
                print (f"[NPC]: {npc_response}")

            except Exception as e:
                error_message = f"Si è verificato un errore: {e}\n"
                print(error_message)
                f.write(f"[ERROR]: {error_message}\n")
                break

except Exception as e:
    print(f"Errore durante la scrittura sul file: {e}")