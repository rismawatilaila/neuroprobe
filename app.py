"""Neuroprobe: skrining awal toksisitas dan target neuro untuk senyawa neuroprotektif."""
import json
import math
from pathlib import Path

import numpy as np
import streamlit as st
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
DATA = Path(__file__).parent / "data"
METALS = {3, 4, 11, 12, 13, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 47, 48, 50, 56, 78, 79, 80, 82, 83}
COLORS = {"cool": "#4fc3f7", "mid": "#b9a9ff", "hot": "#ff5c8a"}
MARK = {"cool": "●", "mid": "◆", "hot": "▲"}
EXAMPLES = {
    "Curcumin": "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Berberine": "COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2",
    "Donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "Ginkgolide B": "CC1C(=O)OC2C(O)C34C5CC(C(C)(C)C)C36C(OC(=O)C6O)OC4(C(=O)O5)C12O",
    "Chlorpromazine": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
}

st.set_page_config(page_title="Neuroprobe", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Menyiapkan indeks kemiripan…")
def load():
    model = json.loads((DATA / "model.json").read_text())
    ref = json.loads((DATA / "reference.json").read_text())
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    names = {n["s"]: n["n"] for n in ref["names"]}
    labs = {l["s"]: l for l in ref["lab"]}
    for l in ref["lab"]:
        names[l["s"]] = l["n"]
    acts = {}
    for a in ref["act"]:
        acts.setdefault(a["s"], []).append(a)
    index = {}
    for s in set(labs) | set(acts):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        e = index.setdefault(c, {"canon": c, "fp": gen.GetFingerprint(m), "name": None, "lab": None, "acts": []})
        e["name"] = e["name"] or names.get(s)
        e["lab"] = e["lab"] or labs.get(s)
        e["acts"].extend(acts.get(s, []))
    entries = list(index.values())
    return model, ref, gen, entries, [e["fp"] for e in entries]


def featurize(smiles, gen):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    at = [a.GetAtomicNum() for a in m.GetAtoms()]
    f = dict(
        MW=Descriptors.MolWt(m), LogP=Crippen.MolLogP(m), TPSA=rdMolDescriptors.CalcTPSA(m),
        HBD=rdMolDescriptors.CalcNumHBD(m), HBA=rdMolDescriptors.CalcNumHBA(m),
        RotB=rdMolDescriptors.CalcNumRotatableBonds(m), Rings=rdMolDescriptors.CalcNumRings(m),
        Arom=rdMolDescriptors.CalcNumAromaticRings(m), Fsp3=rdMolDescriptors.CalcFractionCSP3(m),
        Hetero=rdMolDescriptors.CalcNumHeteroatoms(m), Halogen=sum(a in (9, 17, 35, 53) for a in at),
        Heavy=m.GetNumHeavyAtoms(),
    )
    fp = gen.GetFingerprint(m)
    bits = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return dict(f=f, fp=fp, bits=bits, canon=Chem.MolToSmiles(m), metal=any(a in METALS for a in at))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict_zebrafish(model, f):
    z = model["intercept"]
    for i, c in enumerate(model["cols"]):
        z += model["coef"][i] * (f[c] - model["mean"][i]) / model["scale"][i]
    return sigmoid(z)


def predict_tox21(model, f, bits):
    t = model["t21"]
    z = t["intercept"]
    for i, c in enumerate(t["dcols"]):
        z += t["coef"][i] * (f[c] - t["mean"][i]) / t["scale"][i]
    z += float(np.dot(np.array(t["coef"][12:]), bits))
    return sigmoid(z)


def out_of_range(model, f):
    out = []
    for c in model["cols"]:
        lo, hi = model["ranges"][c]
        span = max(hi - lo, 1e-9)
        if f[c] < lo - 0.25 * span or f[c] > hi + 0.25 * span:
            out.append(c)
    return out


def tag(tier, text):
    return f"<span style='color:{COLORS[tier]};font-weight:700'>{MARK[tier]} {text}</span>"


def gauge(p):
    cx, cy, r = 180, 170, 140

    def pt(v, rr=r):
        a = math.pi * (1 - v)
        return cx + rr * math.cos(a), cy - rr * math.sin(a)

    def arc(v0, v1, col):
        x0, y0 = pt(v0); x1, y1 = pt(v1)
        return f'<path d="M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}" stroke="{col}" stroke-width="16" fill="none"/>'

    nx, ny = pt(p, r - 26)
    return (
        f'<svg viewBox="0 0 360 200" width="100%" style="max-width:360px">'
        f'{arc(0, .35, COLORS["hot"])}{arc(.35, .65, COLORS["mid"])}{arc(.65, 1, COLORS["cool"])}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#e8ecfa" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="7" fill="#e8ecfa"/>'
        f'<text x="{cx-30}" y="{cy+24}" fill="#e8ecfa" font-size="16" font-family="monospace">{p*100:.0f}%</text></svg>'
    )


model, ref, gen, entries, fps = load()

st.markdown("# Neuroprobe")
st.caption("Skrining awal senyawa neuroprotektif: toksisitas, respons stres sel, target neuro mirip, dan sifat untuk menembus otak.")
st.markdown(
    f"<div style='display:grid;grid-template-columns:35fr 30fr 35fr;height:8px'>"
    f"<i style='background:{COLORS['hot']}'></i><i style='background:{COLORS['mid']}'></i><i style='background:{COLORS['cool']}'></i></div>"
    f"<div style='display:flex;justify-content:space-between;font-size:12px;opacity:.75'><span>▲ cenderung toksik</span><span>◆ tidak pasti</span><span>● tidak terdeteksi toksik</span></div>",
    unsafe_allow_html=True,
)

tab_uji, tab_data = st.tabs(["Uji senyawa", "Data dan model"])

with tab_uji:
    if "smi" not in st.session_state:
        st.session_state.smi = EXAMPLES["Curcumin"]
    c1, c2 = st.columns([3, 2])
    with c2:
        pick = st.selectbox("Contoh senyawa", ["(pilih)"] + list(EXAMPLES))
        if pick != "(pilih)":
            st.session_state.smi = EXAMPLES[pick]
        by_name = st.selectbox("Atau cari nama di dataset", [""] + sorted({n["n"] for n in ref["names"]}), index=0)
        if by_name:
            st.session_state.smi = next(n["s"] for n in ref["names"] if n["n"] == by_name)
    with c1:
        smiles = st.text_input("SMILES senyawa", key="smi").strip()

    r = featurize(smiles, gen) if smiles else None
    if not smiles:
        st.info("Masukkan SMILES atau pilih contoh.")
    elif r is None:
        st.error("SMILES tidak valid. Periksa penulisannya atau pilih salah satu contoh.")
    else:
        f = r["f"]
        p = predict_zebrafish(model, f)
        p21 = predict_tox21(model, f, r["bits"])
        sims = DataStructs.BulkTanimotoSimilarity(r["fp"], fps)
        rows = list(zip(entries, sims))
        lab_n = sorted([x for x in rows if x[0]["lab"]], key=lambda x: -x[1])[:5]
        exact = next((e for e, s in rows if s >= 0.999 and e["canon"] == r["canon"]), None)
        near = [(e, s) for e, s in rows if s >= 0.4 and e["acts"]]
        groups = {}
        for e, s in near:
            for a in e["acts"]:
                g = groups.setdefault(a["t"], {"t": a["t"], "n": set(), "sim": 0, "best": float("inf")})
                g["n"].add(e["canon"]); g["sim"] = max(g["sim"], s); g["best"] = min(g["best"], a["v"])
        act_g = sorted(groups.values(), key=lambda g: (-g["sim"], g["best"]))

        if r["metal"]:
            st.warning("Senyawa mengandung logam atau garam anorganik. Model dilatih terutama dari molekul organik, jadi hasilnya kurang bisa dipercaya.")
        if exact:
            lab = exact["lab"]
            st.info(f"Senyawa ini ada di dataset sebagai **{exact['name'] or '(tanpa nama)'}**"
                    + (f", LC50 terukur {lab['lc']:g} µM ({lab['cl']})" if lab else "")
                    + ". Gunakan nilai terukur itu, bukan prediksi.")

        top_sim = lab_n[0][1] if lab_n else 0
        flags = []
        oor = out_of_range(model, f)
        if oor:
            flags.append("Nilai " + ", ".join(oor) + " di luar rentang data latih.")
        if top_sim < 0.3:
            flags.append(f"Tidak ada senyawa uji toksisitas yang mirip (kemiripan tertinggi {top_sim:.2f}).")
        if p <= 0.35:
            tier, verdict, desc = "hot", "Cenderung toksik", "Model memperkirakan LC50 ikan zebra di bawah 100 µM."
        elif p >= 0.65:
            tier, verdict, desc = "cool", "Tidak terdeteksi toksik", "Model memperkirakan LC50 ikan zebra 100 µM atau lebih. Ini bukan bukti keamanan."
        else:
            tier, verdict, desc = "mid", "Tidak pasti", "Probabilitas berada di zona tengah. Butuh data eksperimen."

        left, right = st.columns([5, 7], gap="large")
        with left:
            st.caption("PROBABILITAS TIDAK TOKSIK")
            st.markdown(gauge(p), unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:28px;font-weight:700'>{tag(tier, verdict)}</div>", unsafe_allow_html=True)
            st.write(desc)
            for x in flags:
                st.caption("⚠ " + x)
        with right:
            if act_g:
                a = act_g[0]
                t_act = "cool" if a["sim"] >= 0.6 else "mid"
                kelas = "kuat" if a["best"] <= 100 else "sedang" if a["best"] <= 1000 else "lemah"
                st.markdown(f"**Kemiripan target neuro**<br>{tag(t_act, a['t'])}<br>"
                            f"<span style='opacity:.75'>Kemiripan {a['sim']:.2f}, potensi terbaik {a['best']:,.1f} nM, kelas {kelas}.</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"**Kemiripan target neuro**<br>{tag('hot', 'Tidak ada yang mirip')}", unsafe_allow_html=True)
            t21 = "hot" if p21 >= 0.5 else "mid" if p21 >= 0.25 else "cool"
            st.markdown(f"**Stres sel (Tox21)**<br>{tag(t21, f'{p21*100:.0f}% peluang aktif')}<br>"
                        f"<span style='opacity:.75'>Aktif pada uji respons stres sel (ARE, MMP, HSE, p53, ATAD5). Dasar: 25% dari {model['t21']['n']:,} senyawa Tox21 aktif.</span>",
                        unsafe_allow_html=True)
            checks = [("MW ≤ 450", f["MW"] <= 450), ("TPSA ≤ 90 Å²", f["TPSA"] <= 90), ("HBD ≤ 3", f["HBD"] <= 3), ("LogP 1–4", 1 <= f["LogP"] <= 4)]
            n = sum(ok for _, ok in checks)
            tc = "cool" if n >= 4 else "mid" if n >= 3 else "hot"
            st.markdown(f"**Sifat untuk sistem saraf pusat**<br>{tag(tc, f'{n} dari 4 terpenuhi')}<br>"
                        f"<span style='opacity:.75'>{' · '.join(('✓ ' if ok else '✗ ') + t for t, ok in checks)}</span>",
                        unsafe_allow_html=True)
            st.markdown("**Deskriptor**")
            st.dataframe({"MW": [round(f["MW"], 1)], "LogP": [round(f["LogP"], 2)], "TPSA": [round(f["TPSA"], 1)],
                          "HBD": [f["HBD"]], "HBA": [f["HBA"]], "Cincin": [f["Rings"]], "Aromatik": [f["Arom"]],
                          "Fsp3": [round(f["Fsp3"], 2)], "Halogen": [f["Halogen"]]}, hide_index=True)

        a_col, b_col = st.columns(2, gap="large")
        with a_col:
            st.subheader("Senyawa uji toksisitas terdekat")
            st.caption(f"Dari {len(ref['lab'])} senyawa dengan LC50 terukur.")
            st.dataframe([{"Senyawa": e["lab"]["n"], "Kemiripan": round(s, 2), "LC50 (µM)": e["lab"]["lc"]} for e, s in lab_n], hide_index=True)
        with b_col:
            st.subheader("Target neuro dari senyawa mirip")
            st.caption(f"Dari {len(near)} senyawa bioaktif ChEMBL dengan kemiripan ≥ 0,40. Potensi adalah IC50, Ki atau EC50 terbaik, bukan bukti efek neuroprotektif.")
            if act_g:
                st.dataframe([{"Target": g["t"], "Jumlah": len(g["n"]), "Mirip maks": round(g["sim"], 2), "nM terbaik": round(g["best"], 1)} for g in act_g], hide_index=True)
            else:
                st.write("Tidak ada.")
        st.caption("Hasil ini untuk skrining awal dan tugas pembelajaran. Tidak menggantikan uji laboratorium atau penilaian toksikolog.")

with tab_data:
    cv = model["cv"]; t21 = model["t21"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baris di berkas", f"{ref['counts']['rows']:,}")
    k2.metric("LC50 ikan zebra", cv["n"])
    k3.metric("Senyawa Tox21 berlabel", f"{t21['n']:,}")
    k4.metric("AUC ikan zebra / Tox21", f"{cv['auc']:.2f} / {t21['auc']:.2f}")
    st.markdown(f"""
**Cara kerja**
- Model ikan zebra: regresi logistik dari 9 deskriptor. Label tidak toksik berarti LC50 96 jam ≥ 100 µM ({cv['n_safe']} dari {cv['n']} senyawa).
- Model Tox21: regresi logistik dari 12 deskriptor dan sidik jari Morgan 2048 bit pada {t21['n']:,} senyawa. Aktif berarti aktif pada minimal satu dari lima uji respons stres sel.
- Kemiripan: sidik jari Morgan radius 2 dengan koefisien Tanimoto.

**Seberapa bisa dipercaya**
- AUC ikan zebra {cv['auc']:.2f} (validasi silang) dan {cv['loo_auc']:.2f} (leave-one-out) pada hanya {cv['n']} senyawa. Model baik menandai senyawa toksik (88% benar), lemah memastikan aman (27% dari yang diprediksi aman ternyata toksik).
- AUC Tox21 {t21['auc']:.2f} dengan validasi yang memisahkan kerangka molekul. Dari prediksi ≥ 50%, 61% benar aktif.
- Tox21 adalah uji sel in vitro, bukan hewan utuh dan bukan sel saraf. Potensi ChEMBL mengukur penghambatan target, bukan efek neuroprotektif.

**Catatan data**
- Nilai LC50 benchmark ikan zebra belum diverifikasi ke publikasi sumbernya. Nilai rotenone (15 µM) tampak terlalu tinggi dan perlu diperiksa.
- Label Tox21 berasal dari data publik MoleculeNet/Tox21.
""")
