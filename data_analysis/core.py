from __future__ import annotations
from typing import Optional, List
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder,
    MinMaxScaler, StandardScaler, RobustScaler
)


# ─────────────────────────────────────────────
#  Helper – Cramér's V
# ─────────────────────────────────────────────
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V statistic for two categorical columns."""
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.values.sum()
    r, k = confusion.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else 0.0


# ─────────────────────────────────────────────
#  Helper – Eta (ANOVA-based)
# ─────────────────────────────────────────────
def _eta_squared(cat: pd.Series, num: pd.Series) -> float:
    """Compute Eta-squared between a categorical and numeric column."""
    groups = [num[cat == c].dropna().values for c in cat.unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    grand_mean = num.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((num - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total != 0 else 0.0


# ═══════════════════════════════════════════════════════════════
#  DataInspector
# ═══════════════════════════════════════════════════════════════
class DataInspector:
    """
    A comprehensive data cleaning and exploration tool for Google Colab.
    Provides interactive visualizations using Plotly and robust data sanitization.
    """

    # Strings treated as missing
    _GARBAGE = {"?", "n/a", "na", "null", "none", "", " ", "nan", "N/A", "NULL", "None"}

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.numeric_df: Optional[pd.DataFrame] = None
        self.categorical_df: Optional[pd.DataFrame] = None
        self._normalized_num: Optional[pd.DataFrame] = None
        self._normalized_cat: Optional[pd.DataFrame] = None

    # ── 1. DATA INGESTION ──────────────────────────────────────

    def upload_data(self) -> None:
        """
        Upload a CSV file from your local machine inside Google Colab.
        Automatically handles garbage strings and attempts type correction.
        """
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return
        filename = list(uploaded.keys())[0]
        content = uploaded[filename]
        self.df = pd.read_csv(
            io.BytesIO(content),
            na_values=list(self._GARBAGE),
            keep_default_na=True
        )
        self._auto_type_correction()
        self._split_column_types()
        print(f"✅ Loaded '{filename}': {self.df.shape[0]} rows × {self.df.shape[1]} columns")

    def load_from_path(self, path: str) -> None:
        """
        Load a CSV directly from a file path (e.g. a Colab sample dataset).
        Automatically handles garbage strings and type correction.
        """
        self.df = pd.read_csv(
            path,
            na_values=list(self._GARBAGE),
            keep_default_na=True
        )
        self._auto_type_correction()
        self._split_column_types()
        print(f"✅ Loaded '{path}': {self.df.shape[0]} rows × {self.df.shape[1]} columns")

    def _auto_type_correction(self) -> None:
        """Force-convert columns to numeric where possible without going all-null."""
        for col in self.df.columns:
            if self.df[col].dtype == object:
                converted = pd.to_numeric(self.df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    self.df[col] = converted

    def _split_column_types(self) -> None:
        """Separate numeric and categorical columns into dedicated DataFrames."""
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        self.numeric_df = self.df[num_cols].copy()
        self.categorical_df = self.df[cat_cols].copy()

    # ── 2. STRUCTURAL ANALYSIS ─────────────────────────────────

    def data_summary(self) -> None:
        """
        Display:
        - Row / column counts
        - First 20 rows
        - Numeric vs categorical column breakdown
        - Missing value counts
        """
        if self.df is None:
            print("No data loaded."); return
        print("=" * 60)
        print(f"Rows: {self.df.shape[0]}   Columns: {self.df.shape[1]}")
        print(f"Numeric columns  ({len(self.numeric_df.columns)}): {list(self.numeric_df.columns)}")
        print(f"Categorical cols ({len(self.categorical_df.columns)}): {list(self.categorical_df.columns)}")
        print("\nMissing values per column:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None ✅")
        print("=" * 60)
        print("\nFirst 20 rows:")
        try:
            from IPython.display import display
            display(self.df.head(20))
        except Exception:
            print(self.df.head(20))

    # ── 3. CLEANING ────────────────────────────────────────────

    def handle_missing_values(
        self,
        strategy: str = "mean",
        fill_value=None,
        columns: Optional[List[str]] = None
    ) -> None:
        """
        Impute missing values.

        Parameters
        ----------
        strategy  : 'mean' | 'median' | 'mode' | 'constant'
        fill_value: used only when strategy='constant'
        columns   : list of column names to target; None = all columns
        """
        if self.df is None:
            print("No data loaded."); return
        cols = columns or self.df.columns.tolist()
        for col in cols:
            if self.df[col].isnull().sum() == 0:
                continue
            if strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == "constant":
                self.df[col].fillna(fill_value, inplace=True)
            else:
                print(f"  Skipping '{col}' (strategy '{strategy}' not applicable).")
        self._split_column_types()
        print(f"✅ Missing values handled with strategy='{strategy}'")

    def remove_duplicates(self) -> None:
        """Remove exact duplicate rows from the dataset."""
        if self.df is None:
            print("No data loaded."); return
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._split_column_types()
        print(f"✅ Removed {before - len(self.df)} duplicate rows. Remaining: {len(self.df)}")

    def handle_outliers(
        self,
        columns: Optional[List[str]] = None,
        action: str = "flag"
    ) -> None:
        """
        Detect outliers using IQR method.

        Parameters
        ----------
        columns : numeric columns to check; None = all numeric columns
        action  : 'flag'   → adds a boolean column '<col>_outlier'
                  'remove' → drops outlier rows
        """
        if self.df is None:
            print("No data loaded."); return
        cols = columns or self.numeric_df.columns.tolist()
        mask = pd.Series([False] * len(self.df), index=self.df.index)
        for col in cols:
            if col not in self.df.columns:
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            col_mask = (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
            if action == "flag":
                self.df[f"{col}_outlier"] = col_mask
            mask |= col_mask
        if action == "remove":
            before = len(self.df)
            self.df = self.df[~mask].reset_index(drop=True)
            print(f"✅ Removed {before - len(self.df)} outlier rows.")
        elif action == "flag":
            print(f"✅ Outlier flag columns added for: {cols}")
        self._split_column_types()

    def delete_rows(self) -> None:
        """Interactively delete rows by index (comma-separated input)."""
        if self.df is None:
            print("No data loaded."); return
        raw = input("Enter row indices to delete (comma-separated): ")
        indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        self.df.drop(index=indices, inplace=True, errors="ignore")
        self.df.reset_index(drop=True, inplace=True)
        self._split_column_types()
        print(f"✅ Deleted rows: {indices}")

    def delete_columns(self) -> None:
        """Interactively delete columns by name (comma-separated input)."""
        if self.df is None:
            print("No data loaded."); return
        raw = input("Enter column names to delete (comma-separated): ")
        cols = [c.strip() for c in raw.split(",")]
        self.df.drop(columns=cols, inplace=True, errors="ignore")
        self._split_column_types()
        print(f"✅ Deleted columns: {cols}")

    # ── 4. NORMALIZATION ───────────────────────────────────────

    def extract_normalized_numeric_data(self, method: str = "minmax") -> pd.DataFrame:
        """
        Scale numeric columns.

        Parameters
        ----------
        method : 'minmax' | 'standard' | 'robust'

        Returns
        -------
        pd.DataFrame with scaled values
        """
        if self.numeric_df is None or self.numeric_df.empty:
            print("No numeric data."); return pd.DataFrame()
        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler()
        }
        scaler = scalers.get(method, MinMaxScaler())
        data = self.numeric_df.dropna()
        scaled = scaler.fit_transform(data)
        self._normalized_num = pd.DataFrame(scaled, columns=data.columns, index=data.index)
        print(f"✅ Numeric data normalized using '{method}' scaling.")
        return self._normalized_num

    def extract_normalized_categorical_data(self, method: str = "onehot") -> pd.DataFrame:
        """
        Encode categorical columns.

        Parameters
        ----------
        method : 'onehot' | 'ordinal' | 'uniform'

        Returns
        -------
        pd.DataFrame with encoded values
        """
        if self.categorical_df is None or self.categorical_df.empty:
            print("No categorical data."); return pd.DataFrame()
        data = self.categorical_df.dropna()
        if method == "onehot":
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = enc.fit_transform(data)
            col_names = enc.get_feature_names_out(data.columns)
            self._normalized_cat = pd.DataFrame(encoded, columns=col_names, index=data.index)
        elif method == "ordinal":
            enc = OrdinalEncoder()
            encoded = enc.fit_transform(data)
            self._normalized_cat = pd.DataFrame(encoded, columns=data.columns, index=data.index)
        elif method == "uniform":
            result = data.copy()
            for col in result.columns:
                codes = pd.Categorical(result[col]).codes.astype(float)
                result[col] = (codes - codes.min()) / (codes.max() - codes.min() + 1e-9)
            self._normalized_cat = result
        else:
            print(f"Unknown method '{method}'."); return pd.DataFrame()
        print(f"✅ Categorical data encoded using '{method}'.")
        return self._normalized_cat

    def merge_normalized_data(self) -> pd.DataFrame:
        """Combine normalized numeric and encoded categorical into one DataFrame."""
        parts = []
        if self._normalized_num is not None and not self._normalized_num.empty:
            parts.append(self._normalized_num)
        if self._normalized_cat is not None and not self._normalized_cat.empty:
            parts.append(self._normalized_cat)
        if not parts:
            print("Run extract_normalized_numeric_data and/or extract_normalized_categorical_data first.")
            return pd.DataFrame()
        merged = pd.concat(parts, axis=1)
        print(f"✅ Merged dataset shape: {merged.shape}")
        return merged

    # ── 5. UNIVARIATE VISUALIZATION ────────────────────────────

    def plot_numeric_distribution(self, column: str) -> None:
        """
        3-panel subplot for a single numeric column:
        - Horizontal Violin/Box
        - Scatter (index vs value)
        - Histogram
        """
        if self.df is None or column not in self.df.columns:
            print(f"Column '{column}' not found."); return
        series = self.df[column].dropna()
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Violin + Box", "Scatter (Index vs Value)", "Histogram"]
        )
        # Panel 1 – Violin
        fig.add_trace(go.Violin(x=series, name=column, box_visible=True,
                                meanline_visible=True, orientation="h"), row=1, col=1)
        # Panel 2 – Scatter
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                 mode="markers", name=column,
                                 marker=dict(size=4, opacity=0.6)), row=1, col=2)
        # Panel 3 – Histogram
        fig.add_trace(go.Histogram(x=series, name=column, nbinsx=30), row=1, col=3)
        fig.update_layout(title=f"Distribution: {column}", showlegend=False, height=400)
        fig.show()

    def plot_all_numeric_distributions(self) -> None:
        """Plot univariate distributions for every numeric column."""
        if self.numeric_df is None:
            print("No numeric data."); return
        for col in self.numeric_df.columns:
            self.plot_numeric_distribution(col)

    # ── 6. RELATIONSHIP VISUALIZATION ─────────────────────────

    def plot_relationship(self, col1: str, col2: str) -> None:
        """
        Auto-detect column types and draw the appropriate chart:
        - Num + Num  → Scatter with OLS trendline
        - Cat + Num  → Box plot with individual points
        - Cat + Cat  → Grouped bar chart
        """
        if self.df is None:
            print("No data loaded."); return
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])

        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols",
                             title=f"{col1} vs {col2} (Scatter + OLS)")
        elif not is_num1 and is_num2:
            fig = px.box(self.df, x=col1, y=col2, points="all",
                         title=f"{col2} by {col1}")
        elif is_num1 and not is_num2:
            fig = px.box(self.df, x=col2, y=col1, points="all",
                         title=f"{col1} by {col2}")
        else:
            counts = self.df.groupby([col1, col2]).size().reset_index(name="count")
            fig = px.bar(counts, x=col1, y="count", color=col2, barmode="group",
                         title=f"{col1} vs {col2} (Grouped Bar)")
        fig.show()

    # ── 7. CATEGORICAL FREQUENCY ───────────────────────────────

    def plot_categorical_frequency(self, column: str) -> None:
        """
        Bar chart of category counts with percentage labels for a categorical column.
        """
        if self.df is None or column not in self.df.columns:
            print(f"Column '{column}' not found."); return
        counts = self.df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
        counts["label"] = counts.apply(lambda r: f"{r['count']} ({r['pct']}%)", axis=1)
        fig = px.bar(counts, x=column, y="count", text="label",
                     title=f"Frequency: {column}")
        fig.update_traces(textposition="outside")
        fig.show()

    # ── 8. ASSOCIATION HEATMAP ─────────────────────────────────

    def plot_all_associations_heatmap(self) -> None:
        """
        Unified heatmap of pairwise associations across all column types:
        - Num–Num  : Pearson r
        - Cat–Cat  : Cramér's V
        - Num–Cat  : Eta-squared (ANOVA)
        """
        if self.df is None:
            print("No data loaded."); return
        cols = self.df.columns.tolist()
        n = len(cols)
        matrix = np.zeros((n, n))

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                num1 = pd.api.types.is_numeric_dtype(self.df[c1])
                num2 = pd.api.types.is_numeric_dtype(self.df[c2])
                try:
                    if num1 and num2:
                        matrix[i, j] = self.df[[c1, c2]].dropna().corr().iloc[0, 1]
                    elif not num1 and not num2:
                        matrix[i, j] = _cramers_v(self.df[c1].dropna(), self.df[c2].dropna())
                    else:
                        cat_col, num_col = (c1, c2) if not num1 else (c2, c1)
                        matrix[i, j] = _eta_squared(self.df[cat_col], self.df[num_col])
                except Exception:
                    matrix[i, j] = 0.0

        fig = go.Figure(go.Heatmap(
            z=matrix, x=cols, y=cols,
            colorscale="RdBu", zmid=0,
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Association")
        ))
        fig.update_layout(title="Association Heatmap (All Column Types)", height=600)
        fig.show()


# ═══════════════════════════════════════════════════════════════
#  PlottingMethods
# ═══════════════════════════════════════════════════════════════
class PlottingMethods:
    """
    Granular chart-generation utilities.
    Each method returns an HTML string of the Plotly figure for flexible embedding.
    """

    @staticmethod
    def bar_chart(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "Bar Chart",
        color: Optional[str] = None
    ) -> str:
        """
        Generate a bar chart and return its HTML representation.

        Parameters
        ----------
        data  : source DataFrame
        x     : column for x-axis categories
        y     : column for bar heights
        title : chart title
        color : optional column for color grouping
        """
        if data is None or data.empty:
            return "<p>No data provided.</p>"
        fig = px.bar(data, x=x, y=y, title=title, color=color, barmode="group")
        return fig.to_html(full_html=False)

    @staticmethod
    def pie_chart(
        data: pd.DataFrame,
        names: str,
        values: str,
        title: str = "Pie Chart"
    ) -> str:
        """
        Generate a pie chart and return its HTML representation.

        Parameters
        ----------
        data   : source DataFrame
        names  : column with slice labels
        values : column with slice sizes
        title  : chart title
        """
        if data is None or data.empty:
            return "<p>No data provided.</p>"
        fig = px.pie(data, names=names, values=values, title=title)
        return fig.to_html(full_html=False)

    @staticmethod
    def histogram(
        data: pd.DataFrame,
        column: str,
        nbins: int = 30,
        title: str = "Histogram"
    ) -> str:
        """
        Generate a histogram and return its HTML representation.

        Parameters
        ----------
        data   : source DataFrame
        column : numeric column to plot
        nbins  : number of histogram bins
        title  : chart title
        """
        if data is None or data.empty:
            return "<p>No data provided.</p>"
        fig = px.histogram(data, x=column, nbins=nbins, title=title)
        return fig.to_html(full_html=False)
