"""Synthetic hospital-network simulator.

Every record produced here is SYNTHETIC. The simulator exists because no openly
licensed, de-identified dataset combines dated encounters, facility identifiers,
laboratory series and medication records (see docs/DATASET_ASSESSMENT.md).

Design principles
-----------------
* Outcomes are generated *forward in time*: whether a patient returns within 30 days is
  drawn from a documented hazard that depends on that encounter's attributes, and the
  return admission then exists as a real row in the encounters table. The modelling label
  is later re-derived from the encounter table by `readmit.features.cohort`, so the label
  builder is tested against how the data were actually produced.
* A latent "frailty" term drives risk but is never observed; observed proxies (coded
  comorbidities, prior utilisation, albumin) only partially capture it.
* Realistic imperfections: incomplete diagnosis coding, informative lab ordering,
  out-of-network readmissions that are invisible in the data, and a competing risk of death.
* The true generating probability is saved in a separate truth table that is never a
  model feature; it gives an oracle ceiling for interpreting model performance.

The coefficients below are plausible *design assumptions*, not clinical estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..clinical import egfr_ckdepi_2021
from ..config import SynthConfig

FACILITIES = [
    ("F01", "Harbor General Hospital", "Academic", 620),
    ("F02", "Northgate Community Hospital", "Community", 240),
    ("F03", "Riverbend Regional Medical Center", "Regional", 410),
    ("F04", "Summit Valley Hospital", "Community", 280),
    ("F05", "Lakeshore University Hospital", "Academic", 780),
    ("F06", "Prairie County Hospital", "Rural", 90),
    ("F07", "Eastbrook Medical Center", "Regional", 360),
    ("F08", "Westfield Community Hospital", "Community", 200),
    ("F09", "Cedar Ridge Hospital", "Rural", 110),
    ("F10", "St. Marlow Medical Center", "Regional", 450),
]

# principal diagnosis groups: key, ICD-10, volume weight, logit offset, log-mean LOS days, ICU prob
DX = [
    ("HF", "I50.9", 0.080, 0.55, 1.55, 0.08),
    ("COPD", "J44.1", 0.060, 0.45, 1.40, 0.06),
    ("PNEUMONIA", "J18.9", 0.080, 0.20, 1.45, 0.07),
    ("AMI", "I21.4", 0.050, 0.10, 1.30, 0.30),
    ("SEPSIS", "A41.9", 0.070, 0.35, 1.90, 0.45),
    ("STROKE", "I63.9", 0.050, 0.00, 1.70, 0.20),
    ("DIABETES", "E11.65", 0.050, 0.25, 1.30, 0.05),
    ("AKI", "N17.9", 0.050, 0.40, 1.50, 0.07),
    ("GI_BLEED", "K92.2", 0.040, 0.15, 1.30, 0.10),
    ("UTI", "N39.0", 0.060, 0.00, 1.20, 0.02),
    ("JOINT_REPLACEMENT", "M16.11", 0.070, -1.30, 0.90, 0.01),
    ("SURGICAL_GI", "K80.20", 0.080, -0.60, 1.10, 0.05),
    ("OTHER", "R07.9", 0.260, -0.10, 0.90, 0.02),
]
DX_KEYS = [d[0] for d in DX]
DX_ICD = np.array([d[1] for d in DX])
DX_W = np.array([d[2] for d in DX])
DX_OFF = np.array([d[3] for d in DX])
DX_LOS = np.array([d[4] for d in DX])
DX_ICU = np.array([d[5] for d in DX])
I_HF, I_COPD, I_PNA, I_AMI, I_SEP, I_STR, I_DM, I_AKI, I_GIB, I_UTI, I_JNT, I_SRG, I_OTH = range(13)

# chronic conditions: key, ICD-10, Charlson-style weight, base logit, age slope (per 10y), frailty slope
CHRONIC = [
    ("HTN", "I10", 0, 0.2, 0.5, 0.2),
    ("DM", "E11.9", 1, -1.1, 0.25, 0.35),
    ("CHF", "I50.22", 1, -2.2, 0.55, 0.65),
    ("COPD", "J44.9", 1, -2.0, 0.35, 0.55),
    ("CKD", "N18.30", 2, -2.2, 0.55, 0.50),
    ("CANCER", "C34.90", 2, -2.3, 0.50, 0.25),
    ("DEMENTIA", "F03.90", 1, -3.3, 0.90, 0.50),
    ("LIVER", "K74.60", 1, -3.1, 0.00, 0.40),
    ("DEPRESSION", "F32.9", 0, -1.6, -0.15, 0.30),
    ("AFIB", "I48.91", 0, -2.3, 0.70, 0.30),
    ("CAD", "I25.10", 1, -1.6, 0.50, 0.40),
    ("STROKE_HX", "Z86.73", 1, -2.8, 0.50, 0.30),
]
C_KEYS = [c[0] for c in CHRONIC]
C_ICD = [c[1] for c in CHRONIC]
C_W = np.array([c[2] for c in CHRONIC])
(C_HTN, C_DM, C_CHF, C_COPD, C_CKD, C_CANCER, C_DEM, C_LIVER, C_DEP, C_AFIB, C_CAD, C_STRK) = range(12)

# log-multipliers on principal dx weights given chronic conditions (dx x chronic)
_M = np.zeros((13, 12))
_M[I_HF, [C_CHF, C_CKD, C_HTN]] = [2.0, 0.3, 0.3]
_M[I_COPD, C_COPD] = 2.0
_M[I_PNA, [C_COPD, C_DEM]] = [0.5, 0.6]
_M[I_AMI, [C_CAD, C_DM]] = [1.2, 0.3]
_M[I_SEP, [C_DEM, C_CKD, C_CANCER]] = [0.4, 0.3, 0.3]
_M[I_STR, [C_AFIB, C_STRK, C_HTN]] = [1.0, 1.0, 0.3]
_M[I_DM, C_DM] = 1.6
_M[I_AKI, C_CKD] = 1.4
_M[I_GIB, [C_LIVER, C_AFIB]] = [0.8, 0.3]
_M[I_UTI, [C_DEM, C_DM]] = [0.6, 0.3]

ACUTE_SECONDARY_ICD = ["E87.1", "N17.9", "D64.9", "E44.0"]  # hyponatremia, AKI, anemia, malnutrition
DISPOSITIONS = ["HOME", "HOME_HEALTH", "SNF", "REHAB", "AMA", "TRANSFER_ACUTE", "HOSPICE", "EXPIRED"]
(D_HOME, D_HH, D_SNF, D_REHAB, D_AMA, D_XFER, D_HOSP, D_EXP) = range(8)
MED_CLASSES_ORDERED = [
    "ANTICOAGULANT", "INSULIN", "OPIOID", "DIURETIC", "ANTIPSYCHOTIC",
    "STEROID", "BETA_BLOCKER", "STATIN", "ANTIBIOTIC", "ACE_ARB",
]
INSURANCE = ["MEDICARE", "MEDICAID", "COMMERCIAL", "SELF_PAY"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sample_rows(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Sample one category per row from an (n, k) probability matrix."""
    p = probs / probs.sum(axis=1, keepdims=True)
    u = rng.random(probs.shape[0])[:, None]
    return (u > np.cumsum(p, axis=1)).sum(axis=1).clip(0, probs.shape[1] - 1)


def _trunc_exp_days(rng: np.random.Generator, n: int, scale: float = 9.0, hi: float = 30.0) -> np.ndarray:
    u = rng.random(n)
    return -scale * np.log(1.0 - u * (1.0 - np.exp(-hi / scale))) + 0.05


def _ts(start: np.datetime64, days: np.ndarray) -> np.ndarray:
    return start + np.round(days * 86400.0).astype("timedelta64[s]")


def facility_effects(cfg: SynthConfig) -> pd.Series:
    """The planted facility log-odds effects (first random draw of `simulate`). Used only to
    validate that facility monitoring recovers known structure; never a model input."""
    eff = np.random.default_rng(cfg.seed).normal(0.0, 0.15, len(FACILITIES))
    return pd.Series(eff, index=[f[0] for f in FACILITIES])


def simulate(cfg: SynthConfig = SynthConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    start = np.datetime64(cfg.window_start, "s")
    T = float((np.datetime64(cfg.window_end, "D") - np.datetime64(cfg.window_start, "D")).astype(int)) + 1.0
    P = cfg.n_patients
    start_dow0 = int((start.astype("datetime64[D]").astype(int) + 3) % 7)  # Monday=0

    # ---------------------------------------------------------------- facilities
    fac_ids = np.array([f[0] for f in FACILITIES])
    fac_w = np.array([f[3] for f in FACILITIES], dtype=float)
    fac_w /= fac_w.sum()
    fac_effect = rng.normal(0.0, 0.15, len(FACILITIES))
    drift_idx = int(np.where(fac_ids == cfg.drift_facility)[0][0])
    drift_t0 = float((np.datetime64(cfg.drift_start, "D") - np.datetime64(cfg.window_start, "D")).astype(int))

    # ---------------------------------------------------------------- patients
    ses = rng.integers(1, 6, P)  # 1 = least deprived ... 5 = most deprived (synthetic ADI-like)
    older = rng.random(P) < 0.52
    age0 = np.where(older, 65 + rng.gamma(2.2, 4.5, P), 18 + rng.beta(2.2, 1.6, P) * 47)
    age0 = np.clip(age0, 18, 97)
    z = 0.35 * (age0 - 60) / 15 + np.sqrt(1 - 0.35**2) * rng.normal(size=P) + 0.12 * (ses - 3)
    z = (z - z.mean()) / z.std()
    female = rng.random(P) < 0.52
    ins = np.empty(P, dtype=int)
    p_mcaid = _sigmoid(-1.0 + 0.6 * (ses - 3))
    u = rng.random(P)
    ins[:] = np.where(
        age0 >= 65,
        np.where(u < 0.88, 0, np.where(u < 0.94, 1, 2)),
        np.where(u < p_mcaid * 0.75, 1, np.where(u < 0.95, 2, 3)),
    )
    chronic = np.zeros((P, 12), dtype=bool)
    for j, (_, _, _, b, a_slope, f_slope) in enumerate(CHRONIC):
        chronic[:, j] = rng.random(P) < _sigmoid(b + a_slope * (age0 - 65) / 10 + f_slope * z)
    charlson_true = chronic.astype(float) @ C_W
    n_chronic = chronic.sum(1)
    home_fac = rng.choice(len(FACILITIES), size=P, p=fac_w)

    # ---------------------------------------------------------------- trajectory simulation
    K = 10  # ring buffer of last admission times
    buf = np.full((P, K), -1e9)
    cnt = np.zeros(P, dtype=int)
    last_disch = np.full(P, -1e9)
    death_t = np.full(P, np.nan)

    cur_pat = np.arange(P)
    cur_t = rng.uniform(0.0, T, P)
    cur_readm = np.zeros(P, dtype=bool)
    cur_planned = np.zeros(P, dtype=bool)
    cur_prevdx = np.full(P, -1)

    enc_parts: list[dict] = []
    diag_parts: list[dict] = []
    med_parts: list[dict] = []
    lab_parts: list[dict] = []
    truth_parts: list[dict] = []
    enc_counter = 0

    while cur_pat.size:
        n = cur_pat.size
        zz = z[cur_pat]
        age = age0[cur_pat] + cur_t / 365.25
        fem = female[cur_pat]
        chron = chronic[cur_pat]
        insc = ins[cur_pat]
        sesc = ses[cur_pat]
        # facility
        fac = np.where(rng.random(n) < 0.82, home_fac[cur_pat], rng.choice(len(FACILITIES), n, p=fac_w))

        # admission type / principal dx
        elective = cur_planned | (~cur_readm & (rng.random(n) < 0.12))
        w = DX_W[None, :] * np.exp(chron.astype(float) @ _M.T)
        w_el = np.zeros_like(w)
        w_el[:, [I_JNT, I_SRG, I_OTH]] = np.array([0.55, 0.40, 0.05])
        w[:, I_JNT] *= 0.08
        w[:, I_SRG] *= 0.6
        w = np.where(elective[:, None], w_el, w)
        dx = _sample_rows(rng, w)
        same = cur_readm & ~cur_planned & (cur_prevdx >= 0) & (rng.random(n) < 0.5)
        dx = np.where(same, cur_prevdx, dx)
        urgent = ~elective & (rng.random(n) < 0.20)
        # admission source: 0 ED, 1 CLINIC, 2 TRANSFER
        u = rng.random(n)
        src = np.where(
            elective, 1,
            np.where(urgent, np.where(u < 0.55, 1, np.where(u < 0.90, 0, 2)),
                     np.where(u < 0.80, 0, np.where(u < 0.90, 2, 1))),
        )

        # prior utilisation (recorded admissions only)
        prior_dt = cur_t[:, None] - buf[cur_pat]
        prior12 = (prior_dt <= 365.0).sum(1)
        is_readm_of_prior = (cur_t - last_disch[cur_pat]) <= 30.0

        # stay
        icu = rng.random(n) < np.clip(DX_ICU[dx] * np.exp(0.2 * zz), 0, 0.9)
        los_mean = np.exp(DX_LOS[dx]) * np.exp(0.012 * (age - 65) + 0.10 * zz) * np.where(icu, 1.5, 1.0)
        los = np.clip(los_mean * np.exp(rng.normal(0, 0.45, n)), 0.6, 30.0)
        los = np.round(los * 24) / 24
        t_disch = cur_t + los
        dow = (np.floor(t_disch).astype(int) + start_dow0) % 7
        weekend = dow >= 5
        n_proc = rng.poisson(0.3 + 1.6 * np.isin(dx, [I_JNT, I_SRG]) + 0.8 * (dx == I_AMI) + 0.5 * icu)

        # disposition
        is_stroke, is_joint = dx == I_STR, dx == I_JNT
        d = np.zeros((n, 8))
        d[:, D_HOME] = np.exp(2.2 - 0.02 * (age - 65) - 0.3 * zz - 0.05 * los + 0.6 * elective)
        d[:, D_HH] = np.exp(0.4 + 0.02 * (age - 65) + 0.2 * zz + 0.6 * is_joint)
        d[:, D_SNF] = np.exp(-0.8 + 0.045 * (age - 65) + 0.4 * zz + 0.08 * los + 0.7 * is_stroke + 0.6 * is_joint)
        d[:, D_REHAB] = np.exp(-2.0 + 1.5 * is_stroke + 1.2 * is_joint)
        d[:, D_AMA] = np.exp(-4.2 + 0.5 * (insc == 3))
        d[:, D_XFER] = np.exp(-4.6 + 0.8 * (dx == I_AMI) + 0.5 * icu)
        d[:, D_HOSP] = np.exp(-4.5 + 0.9 * zz + 0.03 * (age - 65) + 1.0 * chron[:, C_CANCER] + 0.8 * chron[:, C_DEM] + 0.8 * (dx == I_SEP))
        d[:, D_EXP] = np.exp(-4.8 + 1.0 * (dx == I_SEP) + 0.8 * icu + 0.7 * zz + 0.03 * (age - 65))
        disp = _sample_rows(rng, d)

        # medications at discharge
        nconds = chron.sum(1)
        rates = np.zeros((n, len(MED_CLASSES_ORDERED)))
        rates[:, 0] = 0.05 + 0.6 * chron[:, C_AFIB] + 0.2 * chron[:, C_STRK] + 0.15 * (dx == I_AMI)
        rates[:, 1] = 0.02 + 0.45 * chron[:, C_DM]
        rates[:, 2] = 0.10 + 0.6 * np.isin(dx, [I_JNT, I_SRG])
        rates[:, 3] = 0.10 + 0.85 * (dx == I_HF) + 0.4 * chron[:, C_CHF] + 0.25 * chron[:, C_HTN]
        rates[:, 4] = 0.02 + 0.30 * chron[:, C_DEM] + 0.05 * chron[:, C_DEP]
        rates[:, 5] = 0.04 + 0.50 * (dx == I_COPD) + 0.20 * chron[:, C_COPD]
        rates[:, 6] = 0.05 + 0.60 * np.isin(dx, [I_HF, I_AMI]) + 0.3 * chron[:, C_AFIB] + 0.2 * chron[:, C_CAD]
        rates[:, 7] = 0.05 + 0.6 * chron[:, C_CAD] + 0.3 * chron[:, C_DM] + 0.3 * (dx == I_STR)
        rates[:, 8] = 0.05 + 0.85 * np.isin(dx, [I_SEP, I_PNA, I_UTI])
        rates[:, 9] = 0.05 + 0.35 * chron[:, C_HTN] + 0.5 * (dx == I_HF) + 0.2 * chron[:, C_DM]
        med_flags = rng.random(rates.shape) < np.clip(rates, 0, 0.97)
        n_other = np.clip(rng.poisson(np.clip(1.5 + 0.55 * nconds + 0.9 * (age > 75) + 0.7 * zz, 0.2, None)), 0, 20)
        n_meds = med_flags.sum(1) + n_other

        # latent lab truth (admission and discharge)
        ckd, liver, cancer, chf = chron[:, C_CKD], chron[:, C_LIVER], chron[:, C_CANCER], chron[:, C_CHF]
        infect = np.isin(dx, [I_SEP, I_PNA, I_UTI])
        na_base = 139.5 - 1.2 * chf - 0.8 * liver - 0.7 * zz
        na_adm = na_base - 2.2 * np.isin(dx, [I_HF, I_PNA, I_SEP]) + rng.normal(0, 3.2, n)
        na_dis = na_base + 0.5 * (na_adm - na_base) + rng.normal(0, 2.0, n)
        cr_base = np.exp(rng.normal(np.log(0.95), 0.18, n)) * np.where(fem, 0.85, 1.1) * (1 + 0.9 * ckd + 0.25 * (age > 75))
        acute = np.clip(np.exp(rng.normal(-0.5, 0.5, n)) - 0.4, 0, None) + 0.9 * (dx == I_AKI) + 0.25 * (dx == I_SEP)
        cr_adm = cr_base * (1 + acute)
        cr_dis = cr_base * (1 + 0.45 * acute) * np.exp(rng.normal(0, 0.08, n))
        hb_base = 13.5 - 1.5 * fem - 1.0 * ckd - 0.8 * cancer - 0.4 * zz
        hb_adm = hb_base - 1.6 * (dx == I_GIB) - 0.3 * infect + rng.normal(0, 1.0, n)
        hb_dis = hb_adm - 0.3 * np.isin(dx, [I_JNT, I_SRG]) + rng.normal(0, 0.5, n)
        wbc_base = 7.5 * np.exp(rng.normal(0, 0.22, n))
        wbc_adm = wbc_base * (1 + 0.8 * infect + rng.normal(0, 0.2, n).clip(-0.4, None))
        wbc_dis = wbc_base * (1 + 0.2 * infect) * np.exp(rng.normal(0, 0.20, n))
        al_base = 4.0 - 0.35 * zz - 0.5 * liver - 0.3 * cancer - 0.2 * ckd + rng.normal(0, 0.3, n)
        al_adm = al_base - 0.35 * (infect | (dx == I_SEP)) + rng.normal(0, 0.15, n)
        al_dis = al_adm + rng.normal(0, 0.2, n)
        egfr_dis = egfr_ckdepi_2021(cr_dis, age, fem)

        # --------------------------------------------------------- readmission hazard (TRUE)
        sex_male = ~fem
        fac_shift = fac_effect[fac] + np.where((fac == drift_idx) & (t_disch >= drift_t0), cfg.drift_logit_shift, 0.0)
        signal = (
            DX_OFF[dx]
            + 0.50 * zz
            + 0.010 * (age - 65)
            + 0.11 * (chron.astype(float) @ C_W)
            + 0.20 * np.minimum(prior12, 6)
            + 0.30 * is_readm_of_prior
            + 0.030 * np.minimum(los, 14) + 0.25 * (los >= 7)
            + 0.15 * icu
            + 0.22 * (disp == D_SNF) + 0.05 * (disp == D_HH) - 0.10 * (disp == D_REHAB) + 0.55 * (disp == D_AMA)
            + 0.28 * (na_dis < 135) + 0.35 * (egfr_dis < 30) + 0.15 * ((egfr_dis >= 30) & (egfr_dis < 45))
            + 0.30 * (hb_dis < 10) + 0.30 * (al_dis < 3.0) + 0.15 * (wbc_dis > 12)
            + 0.18 * (n_meds >= 10) + 0.12 * med_flags[:, 1] + 0.05 * med_flags[:, 0] + 0.10 * med_flags[:, 2] + 0.15 * med_flags[:, 4]
            + 0.25 * (dx == I_HF) * ckd
            + 0.25 * (age > 80) * chron[:, C_DEM]
            + 0.12 * zz * (prior12 >= 3)
            + 0.20 * (insc == 1) + 0.10 * (insc == 3) - 0.05 * (insc == 2)
            + 0.07 * (sesc - 3)
            + 0.06 * sex_male
            + 0.08 * weekend
            + 0.30 * (dx == I_HF) * (los < 3)
            + 0.25 * (dx == I_HF) * (na_dis < 135)
            + 0.20 * (dx == I_COPD) * med_flags[:, 5]
            + 0.20 * (age > 75) * (n_meds >= 10)
            + 0.25 * (disp == D_SNF) * chron[:, C_DEM]
            + 0.07 * np.cos(2 * np.pi * (t_disch % 365.25 - 15) / 365.25)
        )
        # Facility effects are NOT scaled so that the injected facility signal stays detectable.
        logit_true = cfg.base_logit + cfg.effect_scale * signal + fac_shift
        # `true_risk` (saved to the truth table) is the signal including unobservable frailty but
        # excluding the hidden encounter-level noise below; no model can observe either.
        p_true = _sigmoid(logit_true)
        p_draw = _sigmoid(logit_true + rng.normal(0.0, cfg.hidden_noise_sd, n))

        # --------------------------------------------------------- outcome draw
        exits = np.isin(disp, [D_EXP, D_HOSP, D_XFER])
        p_death = np.clip(0.006 * np.exp(0.5 * zz + 0.02 * (age - 65)), 0, 0.08)
        u = rng.random(n)
        dies = ~exits & (u < p_death)
        readm = ~exits & ~dies & (u < p_death + p_draw * (1 - p_death))
        hidden = readm & (rng.random(n) < cfg.out_of_network_rate)
        recorded_readm = readm & ~hidden
        planned_next = ~exits & ~dies & ~readm & (rng.random(n) < 0.02)
        p_more = _sigmoid(-0.55 + 0.55 * zz + 0.10 * n_chronic[cur_pat] + 0.012 * (age - 65))
        far = ~exits & ~dies & ~readm & ~planned_next & (rng.random(n) < p_more)
        hidden_far = hidden & (rng.random(n) < 0.5)

        gap_readm = _trunc_exp_days(rng, n)
        gap_plan = rng.uniform(5.0, 30.0, n)
        gap_far = 31.0 + rng.exponential(190.0 * np.exp(-0.30 * zz))
        next_t = np.full(n, np.nan)
        next_t = np.where(recorded_readm, t_disch + gap_readm, next_t)
        next_t = np.where(planned_next, t_disch + gap_plan, next_t)
        next_t = np.where(far | hidden_far, t_disch + gap_far + np.where(hidden_far, 30.0, 0.0), next_t)
        death_when = t_disch + rng.uniform(1.0, 30.0, n)
        death_t[cur_pat[dies]] = death_when[dies]
        # a hospice discharge usually ends the observable record
        # (expired in-hospital deaths also recorded)
        in_hosp_death = disp == D_EXP
        death_t[cur_pat[in_hosp_death]] = t_disch[in_hosp_death]

        # --------------------------------------------------------- build tables for this round
        eid = np.arange(enc_counter, enc_counter + n)
        enc_counter += n
        adm_type = np.where(elective, "ELECTIVE", np.where(urgent, "URGENT", "EMERGENCY"))
        src_lbl = np.array(["ED", "CLINIC_REFERRAL", "TRANSFER_IN"])[src]
        enc_parts.append(dict(
            _eid=eid, _pat=cur_pat, facility_idx=fac,
            admit_days=cur_t, discharge_days=t_disch,
            admission_type=adm_type, admit_source=src_lbl,
            discharge_disposition=np.array(DISPOSITIONS)[disp],
            principal_dx=DX_ICD[dx], icu_stay=icu.astype(int), n_procedures=n_proc,
            is_planned_admission=cur_planned.astype(int),
        ))
        truth_parts.append(dict(
            _eid=eid, true_risk=p_true,
            true_event_any=(readm).astype(int),
            true_event_recorded=recorded_readm.astype(int),
        ))

        # diagnoses (secondary): documented chronic conditions + acute findings
        doc = chron & (rng.random(chron.shape) < 0.88)
        acute_flags = np.stack([
            (na_dis < 135) & (rng.random(n) < 0.6),
            ((cr_dis / np.maximum(cr_base, 1e-3)) > 1.5) & (rng.random(n) < 0.7),
            (hb_dis < 10) & (rng.random(n) < 0.6),
            (al_dis < 2.8) & (rng.random(n) < 0.3),
        ], axis=1)
        all_sec = np.concatenate([doc, acute_flags], axis=1)
        sec_codes = np.array(C_ICD + ACUTE_SECONDARY_ICD)
        ii, jj = np.nonzero(all_sec)
        first = np.r_[True, ii[1:] != ii[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(ii)), 0))
        diag_parts.append(dict(_eid=eid[ii], seq=np.arange(len(ii)) - grp_start + 2, icd10=sec_codes[jj]))
        diag_parts.append(dict(_eid=eid, seq=np.ones(n, dtype=int), icd10=DX_ICD[dx]))

        # medications
        mi, mj = np.nonzero(med_flags)
        classes = np.array(MED_CLASSES_ORDERED)
        med_parts.append(dict(_eid=eid[mi], drug_class=classes[mj], n_drugs=np.ones(len(mi), dtype=int)))
        o = n_other > 0
        med_parts.append(dict(_eid=eid[o], drug_class=np.full(o.sum(), "OTHER"), n_drugs=n_other[o]))

        # labs: [lab][timepoint]; admission panel mostly ordered, discharge panel informative
        order_adm = rng.random(n) < 0.93
        order_dis = rng.random(n) < np.clip(0.55 + 0.08 * np.minimum(los, 5) + 0.04 * icu, 0, 0.95)
        order_alb = rng.random(n) < 0.45
        labs_true = {
            "SODIUM": (na_adm, na_dis, 1.0, False),
            "CREATININE": (cr_adm, cr_dis, 0.03, True),
            "HEMOGLOBIN": (hb_adm, hb_dis, 0.3, False),
            "WBC": (wbc_adm, wbc_dis, 0.03, True),
            "ALBUMIN": (al_adm, al_dis, 0.1, False),
        }
        for name, (v_adm, v_dis, noise, rel) in labs_true.items():
            for tp, vals, mask in (("ADM", v_adm, order_adm), ("DIS", v_dis, order_dis)):
                if name == "ALBUMIN":
                    mask = mask & order_alb
                obs = vals * (1 + rng.normal(0, noise, n)) if rel else vals + rng.normal(0, noise, n)
                if tp == "ADM":
                    t_col = cur_t + rng.uniform(0.0, 0.25, n).clip(0, los * 0.9)
                else:
                    t_col = t_disch - rng.uniform(0.02, 0.5, n).clip(0, los * 0.9)
                idx = np.nonzero(mask)[0]
                lab_parts.append(dict(_eid=eid[idx], lab_name=np.full(len(idx), name), collected_days=t_col[idx], value=obs[idx]))

        # update history buffers for the *next* round
        slot = cnt[cur_pat] % K
        buf[cur_pat, slot] = cur_t
        cnt[cur_pat] += 1
        last_disch[cur_pat] = t_disch

        # advance
        keep = ~np.isnan(next_t) & (next_t <= T)
        cur_pat = cur_pat[keep]
        cur_t = next_t[keep]
        cur_readm = recorded_readm[keep]
        cur_planned = planned_next[keep]
        cur_prevdx = dx[keep]

    # ---------------------------------------------------------------- assemble tables
    def cat(parts):
        keys = parts[0].keys()
        return pd.DataFrame({k: np.concatenate([p[k] for p in parts]) for k in keys})

    enc = cat(enc_parts)
    # shuffle the integer ids so identifiers carry no temporal ordering
    perm_p = rng.permutation(P)
    enc["patient_id"] = np.char.add("P", (100000 + perm_p[enc["_pat"].to_numpy()]).astype(str))
    enc = enc.sort_values("admit_days").reset_index(drop=True)
    eid_map = {old: f"E{1000000 + i}" for i, old in enumerate(enc["_eid"].to_numpy())}
    enc["encounter_id"] = enc["_eid"].map(eid_map)
    enc["facility_id"] = fac_ids[enc["facility_idx"].to_numpy()]
    enc["admit_ts"] = _ts(start, enc["admit_days"].to_numpy())
    enc["discharge_ts"] = _ts(start, enc["discharge_days"].to_numpy())
    encounters = enc[[
        "encounter_id", "patient_id", "facility_id", "admit_ts", "discharge_ts", "admission_type",
        "admit_source", "discharge_disposition", "principal_dx", "icu_stay", "n_procedures", "is_planned_admission",
    ]].copy()

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["encounter_id"] = df["_eid"].map(eid_map)
        return df.drop(columns="_eid")

    diagnoses = remap(cat(diag_parts))[["encounter_id", "seq", "icd10"]].sort_values(["encounter_id", "seq"]).reset_index(drop=True)
    medications = remap(cat(med_parts))[["encounter_id", "drug_class", "n_drugs"]].sort_values("encounter_id").reset_index(drop=True)
    labs = remap(cat(lab_parts))
    labs["collected_ts"] = _ts(start, labs["collected_days"].to_numpy())
    pat_of_enc = encounters.set_index("encounter_id")[["patient_id", "facility_id"]]
    labs = labs.join(pat_of_enc, on="encounter_id")
    labs["unit"] = labs["lab_name"].map({"SODIUM": "mmol/L", "CREATININE": "mg/dL", "HEMOGLOBIN": "g/dL", "WBC": "10^9/L", "ALBUMIN": "g/dL"})
    labs = labs[["encounter_id", "patient_id", "facility_id", "lab_name", "collected_ts", "value", "unit"]].reset_index(drop=True)
    truth = remap(cat(truth_parts))

    patients = pd.DataFrame({
        "patient_id": np.char.add("P", (100000 + perm_p).astype(str)),
        "birth_year": (int(cfg.window_start[:4]) - np.round(age0)).astype(int),
        "sex": np.where(female, "F", "M"),
        "insurance": np.array(INSURANCE)[ins],
        "ses_quintile": ses,
        "death_ts": np.where(np.isnan(death_t), np.datetime64("NaT"), _ts(start, np.nan_to_num(death_t))),
    })
    patients = patients[patients["patient_id"].isin(encounters["patient_id"].unique())].reset_index(drop=True)

    # ED visits that did not result in admission (history signal)
    years = T / 365.25
    lam = 0.30 * np.exp(0.65 * z + 0.01 * (age0 - 65) + 0.10 * (ses - 3)) * years
    k = rng.poisson(lam)
    pat_rep = np.repeat(np.arange(P), k)
    ed_days = rng.uniform(0.0, T, len(pat_rep))
    dt = death_t[pat_rep]
    ok = np.isnan(dt) | (ed_days < dt)
    pat_rep, ed_days = pat_rep[ok], ed_days[ok]
    ed = pd.DataFrame({
        "patient_id": np.char.add("P", (100000 + perm_p[pat_rep]).astype(str)),
        "facility_id": fac_ids[np.where(rng.random(len(pat_rep)) < 0.82, home_fac[pat_rep], rng.choice(len(FACILITIES), len(pat_rep), p=fac_w))],
        "visit_ts": _ts(start, ed_days),
    })
    ed = ed[ed["patient_id"].isin(patients["patient_id"])].sort_values("visit_ts").reset_index(drop=True)
    ed.insert(0, "ed_visit_id", [f"V{2000000 + i}" for i in range(len(ed))])

    facilities = pd.DataFrame(FACILITIES, columns=["facility_id", "facility_name", "facility_type", "licensed_beds"])
    return dict(
        facilities=facilities, patients=patients, encounters=encounters, diagnoses=diagnoses,
        medications=medications, labs=labs, ed_visits=ed, _truth=truth,
    )


# ====================================================================== DQ defect injection
def inject_data_quality_issues(tables: dict[str, pd.DataFrame], cfg: SynthConfig = SynthConfig()) -> tuple[dict[str, pd.DataFrame], dict]:
    """Inject documented raw-data defects so the DQ layer has something real to find.

    Returns the corrupted tables and a ledger of what was injected (used by tests to
    verify the monitor actually detects and repairs the defects).
    """
    rng = np.random.default_rng(cfg.seed + 1)
    out = {k: v.copy() for k, v in tables.items()}
    ledger: dict = {}

    enc = out["encounters"]
    n_dup = int(len(enc) * cfg.dq_duplicate_rate)
    dup = enc.sample(n_dup, random_state=cfg.seed)
    out["encounters"] = pd.concat([enc, dup], ignore_index=True)
    ledger["duplicate_encounter_rows"] = n_dup

    enc = out["encounters"]
    n_bad = int(len(enc) * cfg.dq_bad_dates_rate)
    bad_idx = rng.choice(len(enc), n_bad, replace=False)
    a = enc.loc[bad_idx, "admit_ts"].to_numpy()
    d = enc.loc[bad_idx, "discharge_ts"].to_numpy()
    enc.loc[bad_idx, "admit_ts"] = d
    enc.loc[bad_idx, "discharge_ts"] = a
    ledger["discharge_before_admit_rows"] = n_bad
    ledger["_bad_date_encounter_ids"] = enc.loc[bad_idx, "encounter_id"].tolist()

    unk = rng.random(len(enc)) < 0.004
    enc.loc[unk, "discharge_disposition"] = "UNKNOWN"
    ledger["unknown_disposition_rows"] = int(unk.sum())
    out["encounters"] = enc

    labs = out["labs"]
    n_imp = int(len(labs) * cfg.dq_impossible_lab_rate)
    idx = rng.choice(len(labs), n_imp, replace=False)
    labs.loc[idx, "value"] = rng.choice([-1.0, 0.0, 9999.0, 99999.0], n_imp)
    ledger["impossible_lab_values"] = n_imp

    # unit change: creatinine silently reported in umol/L (x88.4) with the unit label NOT updated
    m = (
        (labs["facility_id"] == cfg.dq_unit_change_facility)
        & (labs["lab_name"] == "CREATININE")
        & (labs["collected_ts"] >= pd.Timestamp(cfg.dq_unit_change_date))
        & (labs["value"] > 0) & (labs["value"] < 30)
    )
    labs.loc[m, "value"] = labs.loc[m, "value"] * 88.4
    ledger["creatinine_unit_shift_rows"] = int(m.sum())
    ledger["creatinine_unit_shift_facility"] = cfg.dq_unit_change_facility
    ledger["creatinine_unit_shift_from"] = cfg.dq_unit_change_date
    out["labs"] = labs
    return out, ledger
