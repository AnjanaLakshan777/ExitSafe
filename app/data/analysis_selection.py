"""Choose which canonical market-data rows reach the quantitative analytics.

AI-sourced rows (e.g. Gemini prices) are secondary data: they stay in the
imported data with their source, but are left out of the analysis unless the
caller explicitly allows them.
"""

from dataclasses import dataclass

import pandas as pd

from app.data.source_catalog import find_source

AI_SOURCED_WARNING = ("Analysis includes secondary AI-sourced prices (Gemini). They are not "
                      "official exchange data, so results are not based on CSE data alone.")
AI_EXCLUDED_NOTE = ("Secondary AI-sourced prices (Gemini) are not treated as canonical exchange "
                    "data and are left out of quantitative analysis by default.")


@dataclass(frozen=True)
class AnalysisSelection:
    data: pd.DataFrame            # rows the analytics will use
    excluded: pd.DataFrame        # AI-sourced rows left out (empty when they're allowed)
    ai_sourced_rows: int          # AI-sourced rows in the input, used or not
    allow_ai_sourced: bool

    @property
    def uses_ai_sourced(self):
        return self.allow_ai_sourced and self.ai_sourced_rows > 0

    @property
    def message(self):
        """What the user should be told about AI-sourced rows, or None if there were none."""
        if not self.ai_sourced_rows:
            return None
        return AI_SOURCED_WARNING if self.allow_ai_sourced else AI_EXCLUDED_NOTE


def is_ai_sourced(sources):
    """Boolean mask of rows whose source is registered as AI-generated."""
    def check(name):
        source = find_source(name) if isinstance(name, str) else None
        return bool(source and source.is_ai_generated)
    return sources.map(check).astype(bool)


def select_analysis_data(data, allow_ai_sourced=False):
    """Split canonical data into the rows for analysis and the AI-sourced rows left out.

    The input is not changed. Rows without a source column, or with any non-AI
    source, are always kept.
    """
    if "source" not in data.columns:
        return AnalysisSelection(data, data.iloc[0:0], 0, allow_ai_sourced)
    ai = is_ai_sourced(data["source"])
    if allow_ai_sourced:
        return AnalysisSelection(data, data.iloc[0:0], int(ai.sum()), True)
    return AnalysisSelection(data[~ai].reset_index(drop=True), data[ai].reset_index(drop=True),
                             int(ai.sum()), False)
