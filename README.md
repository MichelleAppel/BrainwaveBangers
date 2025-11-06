# EEG-driven Music Optimization 🎧🧠

This project connects **EEG data** (Unicorn Hybrid Black or mock) to **Ableton Live** via OSC.
A real-time optimizer (SPSA or Bayesian Optimization) adjusts musical parameters
— `drums`, `pad`, `tempo`, `grain` — to reach emotional targets in **arousal** and **valence** space.

---

## 🚀 Run

Use one of the following commands:

```bash
python main.py --optimizer spsa
```

```bash
python main.py --optimizer bo
```

If no EEG device is found, a mock source simulates signals.

---

## ⚙️ Structure

File | Purpose
---- | --------
main.py | Core loop (EEG → features → optimizer → OSC)
features.py | Extract arousal/valence from EEG bands
sources.py | Real or mock EEG source
optimizers/ | SPSA + Bayesian optimizers
plotting.py | Live A/V + parameter visualization
config.py | Settings & emotional targets

---

## 🧠 Tuning

- Smooth A/V → increase `EMA_TAU` in `config.py`
- Phase length → edit `epoch_hops` in `main.py`
- Learning rate → adjust `alpha0` in `spsa.py`
- Exploration (BO) → change `xi` or `sigma` in `bo.py`

