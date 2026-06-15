import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

# Put this class into regime_discovery.py.
# If regime_discovery.py already has `import common`, keep only one import.
import common


class LabelRatioCurveAnalyzer:
    """
    Plot-only analyzer for label-ratio curves.

    Purpose
    -------
    Similar to LabelRegimeAnalyzer, but focused on one-dimensional parameter sweeps:
        1) para.para_type == 'volatility': x-axis is vol_multiplier
        2) para.para_type == 'horizon'   : x-axis is predict_num / horizon

    For each parameter value, it directly calls the provided label function, normally:
        common.attach_label(temp_df, para=para, label_col='label')

    It does not patch, wrap, or replace FTHL/TBM labeling logic.
    """

    CLASS_CONFIG = {
        "positive": {
            "col": "p_long",
            "gaussian_col": "g_positive",
            "title": "Positive Label Ratio",
            "ylabel": "Positive ratio",
            "color": "green",
        },
        "negative": {
            "col": "p_short",
            "gaussian_col": "g_negative",
            "title": "Negative Label Ratio",
            "ylabel": "Negative ratio",
            "color": "red",
        },
        "neutral": {
            "col": "p_neutral",
            "gaussian_col": "g_neutral",
            "title": "Neutral Label Ratio",
            "ylabel": "Neutral ratio",
            "color": "gray",
        },
    }

    def __init__(self, df, interval_ms, para=common.BaseDefine(), output_dir=common.OUTPUT_DIR):
        self.df = df
        self.interval_ms = interval_ms
        self.para = para
        self.symbol = para.symbol
        self.interval = para.interval
        self.label_type = str(para.label_type).lower()
        self.para_type = str(para.para_type).lower()

        self.output_dir = os.path.join(
            output_dir,
            "regime_discovery_output",
            f"{para.symbol}_{para.interval}",
            f"{self.label_type}_{self.para_type}_label_ratio_curves",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.results_df = None

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    def run_parameter_sweep(
        self,
        parameter_range,
        stop_range=None,
        fun=common.attach_label,
        label_col="label",
    ):
        """
        Scan one parameter and record label ratios.

        Parameters
        ----------
        parameter_range : iterable
            If para.para_type == 'volatility', values are vol_multiplier.
            If para.para_type == 'horizon', values are predict_num / horizon.
        stop_range : iterable or None
            Kept for compatibility with LabelRegimeAnalyzer. Default: [np.inf].
        fun : callable
            Label function, normally common.attach_label.
        label_col : str
            Temporary label column name used in each sweep iteration.
        """
        if stop_range is None:
            stop_range = [np.inf]

        parameter_values = list(parameter_range)
        sweep_data = []

        x_name = self._x_name()
        print(
            f"🚀 Scanning label ratio curves | "
            f"symbol={self.symbol}, interval={self.interval}, "
            f"label_type={self.label_type}, para_type={self.para_type}"
        )

        for x in tqdm(parameter_values, desc=f"{x_name} steps"):
            for stop in stop_range:
                temp_df = self.df.copy()
                para = copy.deepcopy(self.para)

                self._apply_parameter(para, x)
                para.stop_multiplier_rate_long = stop
                para.stop_multiplier_rate_short = stop

                temp_df = fun(temp_df, para=para, label_col=label_col)

                if label_col not in temp_df.columns:
                    raise ValueError(
                        f"Label column '{label_col}' not found after calling label function."
                    )

                valid_df = temp_df[temp_df[label_col] != common.Signal.INVALID]

                if len(valid_df) > 0:
                    counts = valid_df[label_col].value_counts(normalize=True).to_dict()
                else:
                    counts = {}

                row = {
                    "parameter": x_name,
                    "x": float(x),
                    "stop_rate": stop,
                    "label_type": para.label_type,
                    "para_type": para.para_type,
                    "vol_multiplier": float(para.vol_multiplier_long),
                    "horizon": int(para.predict_num),
                    "valid_count": int(len(valid_df)),
                    "p_short": float(counts.get(common.Signal.NEGATIVE, 0.0)),
                    "p_neutral": float(counts.get(common.Signal.NEUTRAL, 0.0)),
                    "p_long": float(counts.get(common.Signal.POSITIVE, 0.0)),
                }

                g = self._gaussian_reference(np.array([float(x)]), para)
                row.update(
                    {
                        "g_negative": float(g["negative"][0]),
                        "g_neutral": float(g["neutral"][0]),
                        "g_positive": float(g["positive"][0]),
                    }
                )
                sweep_data.append(row)

        self.results_df = pd.DataFrame(sweep_data)
        out_csv = os.path.join(self.output_dir, "label_ratio_sweep_results.csv")
        self.results_df.to_csv(out_csv, index=False)
        print(f"✅ Sweep completed: {len(self.results_df)} samples")
        print(f"✅ Results saved: {out_csv}")
        return self.results_df

    def _x_name(self):
        if self.para_type == "volatility":
            return "vol_multiplier"
        if self.para_type == "horizon":
            return "horizon"
        raise ValueError(f"Unsupported para_type: {self.para.para_type}")

    def _x_label(self):
        if self.para_type == "volatility":
            return "Volatility Multiplier"
        if self.para_type == "horizon":
            return "Horizon / predict_num"
        return "Parameter"

    def _apply_parameter(self, para, x):
        if self.para_type == "volatility":
            x = round(float(x), 6)
            para.vol_multiplier_long = x
            para.vol_multiplier_short = x
        elif self.para_type == "horizon":
            para.predict_num = int(round(float(x)))
        else:
            raise ValueError(f"Unsupported para_type: {para.para_type}")

    # ------------------------------------------------------------------
    # Gaussian null
    # ------------------------------------------------------------------
    def _gaussian_reference(self, x, para):
        """
        Zero-mean Gaussian null model.

        volatility sweep:
            z = vol_multiplier

        horizon sweep:
            z = fixed_vol_multiplier / sqrt(horizon)

        Then:
            P(pos) = 1 - Phi(z)
            P(neg) = Phi(-z) = 1 - Phi(z)
            P(neu) = Phi(z) - Phi(-z)
        """
        x = np.asarray(x, dtype=float)

        if self.para_type == "volatility":
            z = x
        elif self.para_type == "horizon":
            horizon = np.maximum(x, 1.0)
            z = float(para.vol_multiplier_long) / np.sqrt(horizon)
        else:
            raise ValueError(f"Unsupported para_type: {self.para_type}")

        p_positive = 1.0 - norm.cdf(z)
        p_negative = norm.cdf(-z)
        p_neutral = 1.0 - p_positive - p_negative

        return {
            "positive": p_positive,
            "negative": p_negative,
            "neutral": p_neutral,
        }

    # ------------------------------------------------------------------
    # Derivative helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth(y, smooth_window):
        y = pd.Series(y, dtype="float64")
        if smooth_window is not None and smooth_window > 1:
            return (
                y.rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
        return y.to_numpy(dtype=float)

    def _curve_data(self, df_plot, empirical_col, gaussian_col, smooth_window):
        x = df_plot["x"].to_numpy(dtype=float)
        empirical = self._smooth(df_plot[empirical_col].to_numpy(dtype=float), smooth_window)
        gaussian = df_plot[gaussian_col].to_numpy(dtype=float)

        if len(x) < 2:
            d1_emp = np.full_like(empirical, np.nan, dtype=float)
            d1_gau = np.full_like(gaussian, np.nan, dtype=float)
        else:
            d1_emp = np.gradient(empirical, x)
            d1_gau = np.gradient(gaussian, x)

        if len(x) < 3:
            d2_emp = np.full_like(empirical, np.nan, dtype=float)
            d2_gau = np.full_like(gaussian, np.nan, dtype=float)
        else:
            d2_emp = np.gradient(d1_emp, x)
            d2_gau = np.gradient(d1_gau, x)

        return x, empirical, gaussian, d1_emp, d1_gau, d2_emp, d2_gau

    def _single_stop_subset(self, stop_rate):
        if self.results_df is None:
            raise RuntimeError("Please run run_parameter_sweep() first.")

        df_plot = self.results_df[np.isclose(self.results_df["stop_rate"], stop_rate)].copy()
        if df_plot.empty:
            raise ValueError(f"No rows found for stop_rate={stop_rate}")

        df_plot = df_plot.sort_values("x").reset_index(drop=True)
        return df_plot

    # ------------------------------------------------------------------
    # Plot public API
    # ------------------------------------------------------------------
    def plot_all_label_ratio_curves(
        self,
        stop_rate=np.inf,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
    ):
        """
        Generate 9 plots:
            positive / negative / neutral ratio curves
            positive / negative / neutral first derivative curves
            positive / negative / neutral second derivative curves

        Each plot contains empirical and Gaussian curves.
        """
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        df_plot = self._single_stop_subset(stop_rate)
        saved_paths = {}

        for class_name in ["positive", "negative", "neutral"]:
            saved_paths[class_name] = self.plot_single_label_ratio_curve(
                class_name=class_name,
                stop_rate=stop_rate,
                output_dir=output_dir,
                smooth_window=smooth_window,
                include_gaussian=include_gaussian,
            )

        overview_path = self.plot_distribution_overview(
            stop_rate=stop_rate,
            output_dir=output_dir,
            smooth_window=smooth_window,
            include_gaussian=include_gaussian,
        )
        saved_paths["overview"] = overview_path

        print(f"✅ All label-ratio plots saved to: {output_dir}")
        return saved_paths

    def plot_single_label_ratio_curve(
        self,
        class_name,
        stop_rate=np.inf,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
    ):
        """Plot ratio, first derivative, and second derivative for one class."""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if class_name not in self.CLASS_CONFIG:
            raise ValueError(f"class_name must be one of {list(self.CLASS_CONFIG)}")

        cfg = self.CLASS_CONFIG[class_name]
        df_plot = self._single_stop_subset(stop_rate)

        x, empirical, gaussian, d1_emp, d1_gau, d2_emp, d2_gau = self._curve_data(
            df_plot=df_plot,
            empirical_col=cfg["col"],
            gaussian_col=cfg["gaussian_col"],
            smooth_window=smooth_window,
        )

        prefix = f"{self.label_type}_{self.para_type}_{class_name}"
        saved = {}

        saved["ratio"] = self._plot_two_curves(
            x=x,
            y_emp=empirical,
            y_gau=gaussian,
            title=f"{cfg['title']} vs {self._x_label()}",
            ylabel=cfg["ylabel"],
            empirical_label="Empirical ratio",
            gaussian_label="Gaussian ratio",
            output_path=os.path.join(output_dir, f"{prefix}_ratio.png"),
            color=cfg["color"],
            include_gaussian=include_gaussian,
            y_lim=(-0.05, 1.05),
        )

        saved["first_derivative"] = self._plot_two_curves(
            x=x,
            y_emp=d1_emp,
            y_gau=d1_gau,
            title=f"First Derivative of {cfg['title']}",
            ylabel="First derivative",
            empirical_label="Empirical dP/dx",
            gaussian_label="Gaussian dP/dx",
            output_path=os.path.join(output_dir, f"{prefix}_d1.png"),
            color=cfg["color"],
            include_gaussian=include_gaussian,
            add_zero_line=True,
        )

        saved["second_derivative"] = self._plot_two_curves(
            x=x,
            y_emp=d2_emp,
            y_gau=d2_gau,
            title=f"Second Derivative of {cfg['title']}",
            ylabel="Second derivative",
            empirical_label="Empirical d²P/dx²",
            gaussian_label="Gaussian d²P/dx²",
            output_path=os.path.join(output_dir, f"{prefix}_d2.png"),
            color=cfg["color"],
            include_gaussian=include_gaussian,
            add_zero_line=True,
        )

        return saved

    def plot_distribution_overview(
        self,
        stop_rate=np.inf,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
    ):
        """One overview figure with positive / negative / neutral ratios."""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        df_plot = self._single_stop_subset(stop_rate)
        x = df_plot["x"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(12, 7))

        for class_name in ["positive", "negative", "neutral"]:
            cfg = self.CLASS_CONFIG[class_name]
            y_emp = self._smooth(df_plot[cfg["col"]].to_numpy(dtype=float), smooth_window)
            ax.plot(
                x,
                y_emp,
                marker="o",
                linewidth=2.2,
                color=cfg["color"],
                label=f"{class_name.capitalize()} empirical",
            )
            if include_gaussian:
                ax.plot(
                    x,
                    df_plot[cfg["gaussian_col"]].to_numpy(dtype=float),
                    linestyle=":",
                    linewidth=1.8,
                    color=cfg["color"],
                    label=f"{class_name.capitalize()} Gaussian",
                )

        ax.set_xlabel(self._x_label(), fontsize=12)
        ax.set_ylabel("Label ratio", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Label Distribution vs {self._x_label()} "
            f"({self.label_type.upper()} / {self.para_type})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9, frameon=True)
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"{self.label_type}_{self.para_type}_distribution_overview.png",
        )
        plt.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"✅ Overview plot saved: {output_path}")
        return output_path

    def _plot_two_curves(
        self,
        x,
        y_emp,
        y_gau,
        title,
        ylabel,
        empirical_label,
        gaussian_label,
        output_path,
        color,
        include_gaussian=True,
        y_lim=None,
        add_zero_line=False,
    ):
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(
            x,
            y_emp,
            marker="o",
            linewidth=2.5,
            color=color,
            label=empirical_label,
        )

        if include_gaussian:
            ax.plot(
                x,
                y_gau,
                linestyle="--",
                linewidth=2.0,
                color="black",
                alpha=0.8,
                label=gaussian_label,
            )

        if add_zero_line:
            ax.axhline(0.0, color="black", linestyle=":", alpha=0.5)

        ax.set_xlabel(self._x_label(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{title}\n{self.label_type.upper()} / {self.para_type}",
            fontsize=14,
            fontweight="bold",
        )
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=True, shadow=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved: {output_path}")
        return output_path
