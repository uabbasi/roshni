"""
Base ETL (Extract → Transform → Load) for health data pipelines.

Subclass this for each data source.  The pipeline calls ``etl()``
which runs extract → transform → load → schema enforcement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .models import ColumnType


class BaseETL(ABC):
    """Abstract base class for health-data ETL processors.

    Each ETL class should return a consistently-typed DataFrame with:
    - Standardized column names (prefixed per source)
    - Proper data types (int, float, str, None)
    - ISO 8601 dates (YYYY-MM-DD)
    - All canonical columns present (even if NaN)

    Requires ``pandas`` (installed via ``roshni[health]``).
    """

    def __init__(self, source_path: str | None = None):
        self.source_path = source_path

    @abstractmethod
    def extract(self) -> "pd.DataFrame":
        """Extract raw data from the source."""

    @abstractmethod
    def transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Transform raw data into a cleaned / normalised DataFrame."""

    @abstractmethod
    def load(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Prepare DataFrame for the main pipeline (typing, renaming)."""

    @abstractmethod
    def get_schema(self) -> dict[str, str]:
        """Return {column_name: ColumnType.xxx} for this source."""

    # ── helpers ──────────────────────────────────────────────────────

    def _ensure_canonical_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Add missing columns from the schema as NaN / None."""
        import pandas as pd

        schema = self.get_schema()
        for col_name, col_type in schema.items():
            if col_name not in df.columns:
                df[col_name] = pd.NA if col_type in (ColumnType.INTEGER, ColumnType.FLOAT) else None
        return df

    def _ensure_proper_types(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Coerce columns to the types declared in ``get_schema()``."""
        import pandas as pd

        schema = self.get_schema()
        for col_name, col_type in schema.items():
            if col_name not in df.columns:
                continue
            if col_type == ColumnType.INTEGER:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(int)
            elif col_type == ColumnType.FLOAT:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0).astype(float)
            elif col_type == ColumnType.DATE:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce").dt.strftime("%Y-%m-%d")
            elif col_type == ColumnType.DATETIME:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            elif col_type == ColumnType.TEXT:
                df[col_name] = df[col_name].astype(str).where(df[col_name].notna(), None)
        return df

    def etl(self) -> "pd.DataFrame":
        """Full pipeline: extract → transform → load → enforce schema."""
        df = self.extract()
        df = self.transform(df)
        df = self.load(df)
        df = self._ensure_canonical_columns(df)
        df = self._ensure_proper_types(df)
        return df
