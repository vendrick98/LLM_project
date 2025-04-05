import time
from datetime import datetime

import google.generativeai as genai
import spacy

def extract_important_keywords_nlp (text, keyword_scores):
    doc = nlp (text)
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            weight = 1
            if token.pos_ in ["NOUN", "PROPN"]:
                weight = 2
            elif token.ent_type_:
                weight = 3
            lemma = token.lemma_ if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] else token.text
            keyword_scores[lemma] = keyword_scores.get (lemma, 0) + weight


def get_token_count_gemini (model, text):
    """Conta il numero di token in un testo usando il modello Gemini."""
    response = model.count_tokens (text)
    return response.total_tokens




# Inserisci qui la tua chiave API
genai.configure (api_key="AIzaSyDwmSWRSdTmxzmTtd67aT7xTo3TgZrSTPs")

# Definisci le impostazioni di sicurezza desiderate
safety_settings =[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]


npc_context = """Sei un uomo sulla sessantina di nome Giovanni, un gioielliere che ha lavorato per 40 anni nel suo negozio "L'Arte dell'Oro" in una tranquilla via del centro di Milano. Quando rispondi, fornisci solo il tuo dialogo diretto come se stessi parlando con un cliente. Evita descrizioni dei tuoi pensieri, delle tue azioni non verbali o del tuo stato d'animo. Concentrati unicamente sulle tue battute verbali."""

# Inizializza il modello GenerativeModel con le impostazioni di sicurezza
modelChat = genai.GenerativeModel (model_name='gemini-2.0-flash', safety_settings=safety_settings, system_instruction=npc_context)
modelSummary = genai.GenerativeModel (model_name='gemini-2.0-flash', safety_settings=safety_settings)



chat = modelChat.start_chat()

# Parametri per il riassunto
SUMMARY_FREQUENCY = 5
USE_KEYWORDS_FOR_SUMMARY = True  # Variabile per scegliere se usare le keyword
MAX_CHAT_HISTORY_TOKENS = 6000  # Limite approssimativo di token per la cronologia

conversation_history = []  # Manteniamo tutta la cronologia
keyword_scores = {}
DECAY_RATE = 0.9
SCORE_THRESHOLD_TO_KEEP = 0.1

# Carica il modello italiano piccolo di spaCy
try:
    nlp = spacy.load ("it_core_news_sm")
except OSError:
    print ("Downloading spaCy Italian model...")
    spacy.cli.download ("it_core_news_sm")
    nlp = spacy.load ("it_core_news_sm")


try:
    keywords_status = "con_keywords" if USE_KEYWORDS_FOR_SUMMARY else "senza_keywords"
    filename =f"conversazione_intera_chat_{keywords_status}_Giovanni_flash_2_0_{datetime.now ().strftime ('%Y%m%d_%H%M%S')}.txt"

    with open ("ChatFullSummary/" + filename, "w", encoding="utf-8") as f:
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

            f.write (f"[USER]: {user_input}\n")
            conversation_history.append ({"role": "user", "content": user_input})
            inizio_scambio = time.time ()
            try:
                # Estrazione delle parole chiave dall'input dell'utente
                inizio_estrazione = time.time ()
                extract_important_keywords_nlp (user_input, keyword_scores)
                fine_estrazione = time.time ()
                tempo_estrazione = fine_estrazione - inizio_estrazione

                start_time = time.time ()

                response = chat.send_message (user_input)
                end_time = time.time ()
                request_time = end_time - start_time

                npc_response = response.text
                f.write (f"[NPC]: {npc_response}\n")
                f.write (f"[TEMPO_RISPOSTA]: {request_time:.2f} secondi\n")
                f.write (f"[TEMPO_ESTRAZIONE_KEYWORD]: {tempo_estrazione:.4f} secondi\n")
                conversation_history.append ({"role": "assistant", "content": npc_response})

                extract_important_keywords_nlp (npc_response, keyword_scores)

                for keyword in list (keyword_scores.keys ()):
                    keyword_scores[keyword] *= DECAY_RATE
                    if keyword_scores[keyword] < SCORE_THRESHOLD_TO_KEEP:
                        del keyword_scores[keyword]

                if len (conversation_history) > 0 and len (conversation_history) % SUMMARY_FREQUENCY == 0:
                    print ("Generazione del riassunto dell'intera chat...")
                    f.write ("[GENERAZIONE RIASSUNTO (INTERA CHAT)]\n")
                    summary_start_time = time.time ()

                    conversation_text_for_summary = "\n".join (
                        [f"{msg['role']}: {msg['content']}" for msg in conversation_history])

                    summary_prompt = f"""Riassumi la seguente conversazione completa:
                    {conversation_text_for_summary}

                    Prestando particolare attenzione a:
                    - Gli argomenti principali discussi e le entità chiave coinvolte (persone, oggetti, concetti,Nomi).
                    - Qualsiasi preferenza, richiesta o necessità espressa dai partecipanti.
                    - Dettagli specifici o esempi concreti che sono stati menzionati per illustrare un punto.
                    - Eventuali azioni intraprese, proposte o discusse durante la conversazione.
                    - Le relazioni o connessioni che si sono sviluppate tra i diversi argomenti o partecipanti.
                    """

                    if USE_KEYWORDS_FOR_SUMMARY:
                        sorted_keywords = sorted (keyword_scores.items (), key=lambda item: item[1], reverse=True)
                        top_keywords = [f"{keyword} ({score:.2f})" for keyword, score in sorted_keywords[:10]]
                        summary_prompt += f"\nParole chiave rilevanti (con punteggio di rilevanza): {top_keywords}"
                        summary_prompt += "\nConsidera i temi principali emersi dalle parole chiave più rilevanti."

                    summary_prompt += "\nAssicurati che il riassunto sia **estremamente conciso** ma completo e che non si perdano dettagli cruciali per la comprensione della conversazione, formattato come testo."

                    prompt_token_count = get_token_count_gemini (modelSummary, summary_prompt)
                    print (f"Lunghezza stimata del prompt (token): {prompt_token_count}")
                    f.write (f"[TOKEN_PROMPT_RIASSUNTO]: {prompt_token_count}\n")

                    if prompt_token_count > MAX_CHAT_HISTORY_TOKENS:
                        print (
                            f"ATTENZIONE: La cronologia della chat è troppo lunga ({prompt_token_count} token). Potrebbe superare i limiti del modello. Considerare di troncare la cronologia.")
                        f.write (f"[ATTENZIONE]: Cronologia chat troppo lunga ({prompt_token_count} token).\n")

                    summary_response = modelSummary.generate_content (summary_prompt)
                    summary_end_time = time.time ()
                    summary_time = summary_end_time - summary_start_time

                    summary_text = summary_response.text
                    if summary_text:

                        print (f"Riassunto generato (Tempo: {summary_time:.2f} secondi):\n{summary_text}")
                        f.write (f"[RIASSUNTO]:\n{summary_text}\n[/RIASSUNTO]\n")
                        f.write (f"[TEMPO_RIASSUNTO]: {summary_time:.2f} secondi\n")
                        summary_token_length = get_token_count_gemini (modelSummary, summary_text)
                        print (f"Lunghezza del riassunto (token stimati): {summary_token_length}")
                        # Riavvia la chat con il contesto iniziale e il riassunto
                        new_history = [
                            {"role": "user", "parts": [f"Riassunto della conversazione precedente:\n{summary_text}"]}]
                        chat = modelChat.start_chat (history=new_history)

                    else:
                        print ("Errore nella generazione del riassunto.")
                        f.write ("[ERRORE GENERAZIONE RIASSUNTO]\n")

                # Stampa della risposta dell'NPC separata e come ultima informazione
                print(f"Fine della iterazione: {time.time() - inizio_scambio:.2f} secondi")
                f.write(f"[FINE ITERAZIONE]: {time.time() - inizio_scambio:.2f} secondi\n")
                print (f"Giovanni: {npc_response}")

            except Exception as e:
                error_message = f"Si è verificato un errore: {e}\n"
                print (error_message)
                f.write (f"[ERROR]: {error_message}\n")
                break

except Exception as e:
    print (f"Errore durante la scrittura sul file: {e}")
