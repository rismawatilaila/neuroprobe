# Neuroprobe

Skrining awal senyawa neuroprotektif: perkiraan toksisitas (ikan zebra), respons stres sel (Tox21),
target neuro yang mirip (ChEMBL), dan sifat untuk menembus otak.

## Menjalankan di komputer sendiri
```
pip install -r requirements.txt
streamlit run app.py
```

## Struktur
- `app.py`: aplikasi Streamlit
- `data/model.json`: koefisien model
- `data/reference.json`: senyawa acuan (LC50 terukur dan bioaktivitas ChEMBL)
- `.streamlit/config.toml`: tema
- `requirements.txt`: kebutuhan paket

## Sumber data dan batasan
- LC50 ikan zebra: 63 senyawa dari berkas proyek (belum diverifikasi ke publikasi sumber).
- Bioaktivitas: ChEMBL.
- Label stres sel: Tox21 (MoleculeNet). Cantumkan sumber dan periksa ketentuan penggunaannya.
- Hasil hanya untuk skrining awal dan pembelajaran, bukan pengganti uji laboratorium.
