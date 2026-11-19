"""Load the processed feature table into BigQuery."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import settings


def load_table(
    parquet_path: Path,
    project: str | None = None,
    dataset: str | None = None,
    table: str = "features",
) -> str:
    """Upload a Parquet file to BigQuery and return the fully qualified table id."""
    from google.cloud import bigquery

    project = project or settings.project
    dataset = dataset or settings.dataset
    if not project:
        raise SystemExit("Set GCP_PROJECT (or pass --project) before loading.")

    client = bigquery.Client(project=project, location=settings.location)
    dataset_id = f"{project}.{dataset}"
    dataset_ref = bigquery.Dataset(dataset_id)
    dataset_ref.location = settings.location
    client.create_dataset(dataset_ref, exists_ok=True)

    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    # BigQuery needs a DATE column, not a Python object column.
    frame["date"] = pd.to_datetime(frame["date"]).dt.date

    table_id = f"{dataset_id}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=bigquery.TimePartitioning(field="date"),
        clustering_fields=["ticker"],
    )
    job = client.load_table_from_dataframe(frame, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(frame):,} rows into {table_id}")
    return table_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/features.parquet"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)

    load_table(args.input, args.project, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
