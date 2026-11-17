"""Small synthetic files in the AMLworld layout, for tests only (not for results)."""
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
          "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering")
TYPES = ["FAN-OUT", "FAN-IN", "CYCLE:  Max 4 hops", "STACK"]


def write_synthetic(raw_dir: Path, variant: str = "HI-Small", n: int = 5000,
                    seed: int = 0, days: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-09-01")
    minutes = np.sort(rng.integers(0, days * 24 * 60, n))
    ts = (start + pd.to_timedelta(minutes, unit="m")).strftime("%Y/%m/%d %H:%M")
    accts = np.array([f"80{i:07X}" for i in range(400)])
    banks = rng.integers(1, 60, size=(n, 2))
    amount = rng.lognormal(7, 2, n).round(2)
    df = pd.DataFrame({
        "Timestamp": ts,
        "From Bank": [f"{b:03d}" for b in banks[:, 0]],
        "src": rng.choice(accts, n), "To Bank": [f"{b:03d}" for b in banks[:, 1]],
        "dst": rng.choice(accts, n),
        "Amount Received": amount, "Receiving Currency": "US Dollar",
        "Amount Paid": amount, "Payment Currency": "US Dollar",
        "Payment Format": rng.choice(["ACH", "Cheque", "Credit Card", "Wire"], n),
        "Is Laundering": 0,
    })
    pos = rng.choice(n, size=max(8, n // 500), replace=False)
    df.loc[pos, "Is Laundering"] = 1

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f"{variant}_Trans.csv", "w") as fh:
        fh.write(HEADER + "\n")
        df.to_csv(fh, header=False, index=False)

    # Patterns file: bank ids padded to four digits, unlike the transaction file.
    lines, chunks = [], np.array_split(np.sort(pos), 4)
    for k, idx in enumerate(chunks):
        t = TYPES[k % len(TYPES)]
        lines.append(f"BEGIN LAUNDERING ATTEMPT - {t}")
        for i in idx:
            r = df.loc[i].copy()
            r["From Bank"] = f"{int(r['From Bank']):04d}"
            r["To Bank"] = f"{int(r['To Bank']):04d}"
            lines.append(",".join(str(v) for v in r.values))
        lines.append(f"END LAUNDERING ATTEMPT - {t.split(':')[0]}")
        lines.append("")
    (raw_dir / f"{variant}_Patterns.txt").write_text("\n".join(lines))
    return df
