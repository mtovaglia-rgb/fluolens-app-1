# FluoLens Compare - MVP Streamlit

Applicazione Streamlit per confrontare una immagine reference di fluoresceina con una immagine campione.

## Funzioni incluse
- Doppio upload immagini
- Selezione manuale di centro e raggio della lente
- Suddivisione in 4 zone concentriche
- Estrazione del segnale di fluoresceina
- Confronto delle intensità per zona
- Giudizio qualitativo del clearance centrale
- Stima approssimativa del clearance in micron
- Profilo radiale reference vs campione

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note
Questo MVP è pensato come base tecnica. La stima in micron è solo orientativa e richiede una futura calibrazione su immagini acquisite con protocollo standardizzato.
