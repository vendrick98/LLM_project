from datetime import datetime

import google.generativeai as genai
import spacy
from rouge_score import rouge_scorer

# Inserisci qui la tua chiave API
genai.configure (api_key="AIzaSyDwmSWRSdTmxzmTtd67aT7xTo3TgZrSTPs")
inizio = datetime.now ()
# Definisci le impostazioni di sicurezza desiderate
safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                   {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]

# Inizializza il modello GenerativeModel con le impostazioni di sicurezza
model = genai.GenerativeModel (model_name='gemini-2.0-flash', safety_settings=safety_settings)

npc_context = """Sei un uomo sulla sessantina di nome Giovanni, un gioielliere che ha lavorato per 40 anni nel suo negozio "L'Arte dell'Oro" in una tranquilla via del centro di Milano. Quando rispondi, fornisci solo il tuo dialogo diretto come se stessi parlando con un cliente. Evita descrizioni dei tuoi pensieri, delle tue azioni non verbali o del tuo stato d'animo. Concentrati unicamente sulle tue battute verbali."""

# Inizializza la chat includendo il contesto come primo messaggio "user"
initial_user_message_for_context = f"Agisci come se fossi: {npc_context}"
chat = model.start_chat (history=[{"role": "user", "parts": [initial_user_message_for_context]}])

# Nome del file
filename = f"conversazione_decay_Giovanni_flash_2_0_{datetime.now ().strftime ('%Y%m%d_%H%M%S')}.txt"

# Parametri per il riassunto
SUMMARY_FREQUENCY = 5
NUM_MESSAGES_TO_KEEP = 3

conversation_history_for_summary = []
summary = ""
keyword_scores = {}  # Dizionario per memorizzare le parole chiave e i loro punteggi
DECAY_RATE = 0.9
SCORE_THRESHOLD_TO_KEEP = 0.1  # Fattore di decadimento (es. 0.9 significa che il punteggio si riduce del 10% per interazione)

# Carica il modello italiano piccolo di spaCy
try:
    nlp = spacy.load ("it_core_news_sm")
except OSError:
    print ("Downloading spaCy Italian model...")
    spacy.cli.download ("it_core_news_sm")
    nlp = spacy.load ("it_core_news_sm")


def extract_important_keywords_nlp(text, keyword_scores):
    doc = nlp(text)
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space :  # Added check for punctuation
            weight = 1
            if token.pos_ in ["NOUN", "PROPN"]:
                weight = 3
            elif token.ent_type_:
                weight = 3
            lemma = token.lemma_ if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] else token.text
            keyword_scores[lemma] = keyword_scores.get(lemma, 0) + weight


try:
    with open ("ChatTestWithNLPAndDecay/" + filename, "w", encoding="utf-8") as f:
        f.write ("[CONTEXT (SYSTEM)]\n")
        f.write (npc_context + "\n")
        f.write ("[/CONTEXT]\n\n")
        f.write ("[INIZIO CONVERSAZIONE]\n")

        print (f"Conversazione salvata nel file: {filename}")
        print ("Inizia a parlare con Giovanni (digita 'esci' per terminare):")

        while True:

            user_input = input ("Tu: ")
            if user_input.lower () == "esci":
                f.write ("[FINE CONVERSAZIONE]\n")
                break

            conversation_history_for_summary.append ({"role": "user", "content": user_input})

            try:
                inizioRichiesta = datetime.now ()

                extract_important_keywords_nlp (user_input, keyword_scores)

                response = chat.send_message (user_input)
                npc_response = response.text

                f.write (f"[NPC]: {npc_response}\n")
                conversation_history_for_summary.append ({"role": "assistant", "content": npc_response})

                extract_important_keywords_nlp (npc_response, keyword_scores)

                # Applica il decay a tutti i punteggi delle parole chiave
                for keyword in list (keyword_scores.keys ()):
                    keyword_scores[keyword] *= DECAY_RATE
                    if keyword_scores[keyword] < SCORE_THRESHOLD_TO_KEEP:
                        del keyword_scores[keyword]

                if len (conversation_history_for_summary) > 0 and len (conversation_history_for_summary) % (
                        SUMMARY_FREQUENCY * 2) == 0:
                    print ("Generazione del riassunto...")
                    f.write ("[GENERAZIONE RIASSUNTO]\n")

                    sorted_keywords = sorted (keyword_scores.items (), key=lambda item: item[1], reverse=True)
                    top_keywords = [f"{keyword} ({score:.2f})" for keyword, score in sorted_keywords[:10]]

                    summary_prompt = f"""Riassumi la seguente conversazione:
                    {conversation_history_for_summary}

                    Prestando particolare attenzione a:
                    - Gli argomenti principali discussi e le entità chiave coinvolte (persone, oggetti, concetti).
                    - Qualsiasi preferenza, richiesta o necessità espressa dai partecipanti.
                    - Dettagli specifici o esempi concreti che sono stati menzionati per illustrare un punto.
                    - Eventuali azioni intraprese, proposte o discusse durante la conversazione.
                    - Le relazioni o connessioni che si sono sviluppate tra i diversi argomenti o partecipanti.

                    Parole chiave rilevanti (con punteggio di rilevanza): {top_keywords}
                    Considera i temi principali emersi dalle parole chiave più rilevanti.
                    Assicurati che il riassunto sia conciso ma completo e che non si perdano dettagli cruciali per la comprensione della conversazione."""
                    summary_response = model.generate_content (summary_prompt)
                    summary_text = summary_response.text
                    if summary_text:
                        summary = f"Riassunto della conversazione precedente: {summary_text}\nParole chiave rilevanti (con punteggio): {top_keywords}\n"
                        f.write (f"[RIASSUNTO]:\n{summary}\n[/RIASSUNTO]\n")

                        new_history = [{"role": "user", "parts": [npc_context]}, {"role": "user", "parts": [summary]}]

                        last_n_messages = conversation_history_for_summary[-(NUM_MESSAGES_TO_KEEP * 2):]
                        for msg in last_n_messages:
                            new_history.append ({"role": msg['role'], "parts": [msg['content']]})

                        chat = model.start_chat (history=new_history)
                        conversation_history_for_summary = last_n_messages.copy ()
                        print ("Riassunto generato e cronologia aggiornata.")
                        f.write ("[CRONOLOGIA AGGIORNATA CON RIASSUNTO]\n")


                    else:
                        print ("Errore nella generazione del riassunto.")
                        f.write ("[ERRORE GENERAZIONE RIASSUNTO]\n")
                f.write (f"[Fine Richiesta]: {datetime.now () - inizioRichiesta}\n")
                print (f"[Fine Richiesta]: {datetime.now () - inizioRichiesta}\n")
                print (f"[NPC]: {npc_response}")
                f.write (f"[NPC]: {npc_response}\n")
            except Exception as e:
                error_message = f"Si è verificato un errore: {e}\n"
                print (error_message)

                f.write (f"[ERROR]: {error_message}\n")
                break

except Exception as e:
    print (f"Errore durante la scrittura sul file: {e}")
