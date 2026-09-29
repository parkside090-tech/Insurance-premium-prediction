"""
generate_dataset.py
--------------------
Generates a clean, synthetic healthwise.csv that mirrors the schema and
relationships of the original notebook's dataset (1203 rows, 17 columns).

This is a stand-in dataset, not real customer data — useful so the
dashboard has something to train on out of the box. Swap it for your
real healthwise.csv at any time; the schema just needs to match.

Column roles (matches the source notebook's own feature audit):
  Modeling features : age, bmi, children, exercise_freq, sex, smoker, region
  Target columns     : annual_charge (regression), risk_tier (classification)
  Redundant           : weight_kg (~bmi), date_of_birth (~age)
  Irrelevant/noise    : favorite_color, zodiac_sign, lucky_number,
                        preferred_contact, marketing_opt_in
  Identifier          : customer_id

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 1203
CURRENT_YEAR = 2026


def generate(n_rows: int = N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 66, n_rows)
    sex = rng.choice(["male", "female"], n_rows)
    bmi = np.clip(rng.normal(28, 5.5, n_rows), 16, 48).round(1)
    height_m = np.clip(rng.normal(1.70, 0.09, n_rows), 1.5, 2.0)
    weight_kg = (bmi * height_m**2).round(1)
    children = rng.choice([0, 1, 2, 3, 4, 5], n_rows, p=[0.32, 0.26, 0.22, 0.12, 0.06, 0.02])
    smoker = rng.choice(["yes", "no"], n_rows, p=[0.22, 0.78])
    region = rng.choice(["north", "south", "east", "west"], n_rows)
    exercise_freq = np.clip(rng.normal(3, 1.8, n_rows), 0, 7).round(0).astype(int)

    birth_year = CURRENT_YEAR - age
    birth_month = rng.integers(1, 13, n_rows)
    birth_day = rng.integers(1, 29, n_rows)
    date_of_birth = [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(birth_year, birth_month, birth_day)]

    favorite_color = rng.choice(["red", "blue", "green", "yellow", "purple", "orange"], n_rows)
    zodiac_sign = rng.choice(
        ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"],
        n_rows,
    )
    lucky_number = rng.integers(1, 100, n_rows)
    preferred_contact = rng.choice(["sms", "phone", "email"], n_rows)
    marketing_opt_in = rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45])

    # --- annual_charge: realistic-ish formula driven by the true modeling
    # features, with smoking and age as the dominant drivers, exercise as a
    # modest discount, and noise on top. ---
    noise = rng.normal(0, 1800, n_rows)
    annual_charge = (
        1200
        + age * 210
        + np.clip(bmi - 25, 0, None) * 260
        + children * 380
        + (smoker == "yes") * 21000
        - exercise_freq * 260
        + noise
    )
    annual_charge = np.clip(annual_charge, 1500, None).round(2)

    # --- risk_tier: bucketed off annual_charge (mirrors how the tiers were
    # evidently derived in the source dataset) ---
    q1, q2 = np.quantile(annual_charge, [0.45, 0.80])
    risk_tier = np.where(annual_charge <= q1, "Low", np.where(annual_charge <= q2, "Medium", "High"))

    customer_id = [f"HW{1000 + i}" for i in range(n_rows)]

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "weight_kg": weight_kg,
            "children": children,
            "smoker": smoker,
            "region": region,
            "exercise_freq": exercise_freq.astype(float),
            "date_of_birth": date_of_birth,
            "favorite_color": favorite_color,
            "zodiac_sign": zodiac_sign,
            "lucky_number": lucky_number,
            "preferred_contact": preferred_contact,
            "marketing_opt_in": marketing_opt_in,
            "annual_charge": annual_charge,
            "risk_tier": risk_tier,
        }
    )

    # Already clean: no duplicates, no missing values
    assert df.duplicated().sum() == 0
    assert df.isna().sum().sum() == 0
    return df


def main():
    df = generate()
    df.to_csv("healthwise.csv", index=False)
    print(f"Wrote healthwise.csv — shape {df.shape}")
    print(df["risk_tier"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()
