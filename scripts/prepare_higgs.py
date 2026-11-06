import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SRC = Path("data/HIGGS.csv.gz")
DST = Path("data/higgs_full.parquet")

# note to self: column order from the paper, label first, then 21 low-level and 7 high-level features
COLS = ["label"] + [
    "lepton_pt", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]
assert len(COLS) == 29


def main():
    t0 = time.time()
    # note to self: float32 for everything, the label gets turned into an int when the loader needs it
    schema = pa.schema([(c, pa.float32()) for c in COLS])
    reader = pacsv.open_csv(
        pa.input_stream(str(SRC), compression="gzip"),
        read_options=pacsv.ReadOptions(column_names=COLS, block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=schema),
    )

    n_rows = 0
    with pq.ParquetWriter(DST, schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            n_rows += batch.num_rows
            print(f"{n_rows:>10,} rows  {time.time() - t0:5.0f}s", flush=True)

    # note to self: quick sanity check on what actually got written
    meta = pq.ParquetFile(DST).metadata
    label = pq.read_table(DST, columns=["label"]).column("label").to_numpy()
    print(f"\nrows {meta.num_rows:,}   columns {meta.num_columns}   "
          f"file {DST.stat().st_size / 1e9:.2f} GB   label mean {label.mean():.4f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()