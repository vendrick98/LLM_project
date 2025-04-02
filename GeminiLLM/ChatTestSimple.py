
import google.generativeai as genai
from datetime import datetime, time

# Inserisci qui la tua chiave API
genai.configure(api_key="AIzaSyDwmSWRSdTmxzmTtd67aT7xTo3TgZrSTPs")
inizio = datetime.now()
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

npc_context = """npc_context = Sei un uomo sulla sessantina di nome Giovanni, un gioielliere che ha lavorato per 40 anni nel suo negozio "L'Arte dell'Oro" in una tranquilla via del centro di Milano. Quando rispondi, fornisci solo il tuo dialogo diretto come se stessi parlando con un cliente. Evita descrizioni dei tuoi pensieri, delle tue azioni non verbali o del tuo stato d'animo. Concentrati unicamente sulle tue battute verbali."""

# Inizializza la chat includendo il contesto come primo messaggio "user"
initial_user_message_for_context = f"Agisci come se fossi: {npc_context}"
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": [initial_user_message_for_context]
    }
])

# Nome del file in cui salvare la conversazione
filename = f"conversazione_giovanni_flash_2_0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Imposta la lunghezza massima della cronologia SENZA contare il messaggio di contesto iniziale
MAX_HISTORY_LENGTH_WITHOUT_CONTEXT = 6  # Esempio: Mantieni al massimo 5 scambi utente-NPC

try:
    with open("ChatSimpleGiovanni/" + filename, "w", encoding="utf-8") as f:
        f.write("[CONTEXT]\n")
        f.write(npc_context + "\n")
        f.write("[/CONTEXT]\n\n")
        f.write("[INIZIO CONVERSAZIONE]\n")
        f.write(f"[INIZIO_CONTESTO_COME_USER]: {initial_user_message_for_context}\n")

        print(f"Conversazione salvata nel file: {filename}")
        print("Inizia a parlare con Giovanni (digita 'esci' per terminare):")
        inizio = datetime.now()
        while True:

            user_input = input("Tu: ")
            if user_input.lower() == "esci":
                f.write("[FINE CONVERSAZIONE]\n")
                break

            f.write(f"[USER]: {user_input}\n")

            try:
                inizioRisposta = datetime.now ()
                response = chat.send_message(user_input)
                npc_response = response.text
                f.write(f"[NPC]: {npc_response}\n")

                # Aggiungi l'input dell'utente e la risposta dell'NPC alla cronologia
                chat.history.append({"role": "user", "parts": [user_input]})
                chat.history.append({"role": "assistant", "parts": [npc_response]})

                # Implementa il troncamento della cronologia mantenendo il primo messaggio (contesto)
                if len(chat.history) > 1 + (MAX_HISTORY_LENGTH_WITHOUT_CONTEXT * 2):
                    chat.history = [chat.history[0]] + chat.history[3:]
                f.write ("tempo generazione: " + str(datetime.now() - inizioRisposta) + "\n")
                print (str (datetime.now () - inizioRisposta) + "\n")
            except Exception as e:
                error_message = f"Si è verificato un errore: {e}\n"
                print(error_message)
                f.write(f"[ERROR]: {error_message}\n")
                break
        f.write("[FINE CONVERSAZIONE]\n")
        print(str(datetime.now() - inizio)+"\n")
        f.write(str(datetime.now() - inizio) + "\n")
        print (f"[NPC]: {npc_response}")

except Exception as e:
    print(f"Errore durante la scrittura sul file: {e}")

"""
chat = model.start_chat(history=[
    {
        "role": "system",
        "parts": [npc_context]
    }
])
genai.configure(api_key="AIzaSyDwmSWRSdTmxzmTtd67aT7xTo3TgZrSTPs")
"""
