"""Single source of fixed experimental parameters (Sections 4.2, 4.5, 4.9, 4.10).

Nothing here is tuned. Any change after the first run is a deviation from the
specification and must be recorded in Appendix A.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"          # Kaggle files, unmodified
DATA_INTERIM = ROOT / "data" / "interim"  # typed, sorted, split frames (Parquet)
RESULTS = ROOT / "results"                # integrity reports, split boundaries, metrics

VARIANTS = ("HI-Small", "LI-Small")

# Header as distributed: the account column name is repeated (Section 4.4).
RAW_HEADER = ["Timestamp", "From Bank", "Account", "To Bank", "Account",
              "Amount Received", "Receiving Currency", "Amount Paid",
              "Payment Currency", "Payment Format", "Is Laundering"]

# Names applied by position after the raw header has been verified (Listing 4.1).
COLUMNS = ["Timestamp", "From Bank", "src_account", "To Bank", "dst_account",
           "Amount Received", "Receiving Currency", "Amount Paid",
           "Payment Currency", "Payment Format", "Is Laundering"]

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

# Published composition (Table 4.1; Altman et al., 2023, Appendix Table 4).
# Counts are rounded in the source, so they are compared with a tolerance.
PUBLISHED = {
    "HI-Small": {"transactions": 5_000_000, "laundering": 3_600, "ratio_1_in": 981},
    "LI-Small": {"transactions": 7_000_000, "laundering": 4_000, "ratio_1_in": 1_942},
}
COUNT_TOLERANCE = 0.05   # rounded published counts ("5M", "3.6K")
RATIO_TOLERANCE = 0.01   # published 1-in-N ratio is given to the unit

SPLIT_QUANTILES = (0.70, 0.85)   # train | validation | test (Section 4.5)
SEEDS = (0, 1, 2, 3, 4)          # Section 4.9

# Columns that may never enter a feature matrix (Sections 4.4 and 4.6).
# Raw identifiers and the raw timestamp are inputs to feature construction, not features.
NON_FEATURE_COLS = ["Is Laundering", "typology", "attempt_id", "block", "file_row",
                    "Timestamp", "src_account", "dst_account", "From Bank", "To Bank"]

# Kaggle dataset (Section 4.2). The version is resolved on the first download and then
# pinned in results/DATA_VERSION.json.
DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
