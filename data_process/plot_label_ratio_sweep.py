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
            para.label_type,
            para.para_type,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.results_df = None

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    def run_parameter_sweep(
        self,
        parameter_range,
        fun=common.attach_label,
        label_col="label",
    ):
        """
        Scan one parameter (volatility multiplier or horizon) and record
        label ratios.

        Parameters
        ----------
        parameter_range : iterable
            If para.para_type == 'volatility', values are vol_multiplier.
            If para.para_type == 'horizon', values are predict_num / horizon.
        fun : callable
            Label function, normally common.attach_label.
        label_col : str
            Temporary label column name used in each sweep iteration.

        Notes
        -----
        stop_multiplier_rate_long / stop_multiplier_rate_short are FIXED
        for the entire sweep: whatever value `self.para` carries at
        construction time is what every sweep point uses (deep-copied,
        never overwritten here). This class does not sweep over
        stop-loss configurations -- to compare different
        stop_multiplier_rate settings, construct a separate
        LabelRatioCurveAnalyzer (with its own `para`) per setting.

        Same-bar double-barrier-touch samples (Signal.AMBIGUOUS) are
        excluded from the "hard invalid" bucket and instead split 50/50
        into the positive/negative counts for this diagnostic ratio
        curve, since the true direction is unrecoverable from OHLC data
        but the underlying touch event itself is real and, for a
        symmetric/driftless barrier, equally likely to have resolved
        either way. This is purely a plotting-time adjustment; the
        underlying label column retains the AMBIGUOUS value untouched,
        so downstream training code (which only selects exact
        POSITIVE/NEGATIVE/NEUTRAL matches) still excludes these samples
        entirely.
        """
        parameter_values = list(parameter_range)
        sweep_data = []

        x_name = self._x_name()
        print(
            f"🚀 Scanning label ratio curves | "
            f"symbol={self.symbol}, interval={self.interval}, "
            f"label_type={self.label_type}, para_type={self.para_type}, "
            f"stop_multiplier_rate_long={self.para.stop_multiplier_rate_long}, "
            f"stop_multiplier_rate_short={self.para.stop_multiplier_rate_short}"
        )

        for x in tqdm(parameter_values, desc=f"{x_name} steps"):
            temp_df = self.df.copy()
            para = copy.deepcopy(self.para)

            self._apply_parameter(para, x)
            # stop_multiplier_rate_long/short are intentionally left
            # untouched here -- they come from self.para as a fixed
            # setting, not from a swept list.

            temp_df = fun(temp_df, para=para, label_col=label_col)

            if label_col not in temp_df.columns:
                raise ValueError(
                    f"Label column '{label_col}' not found after calling label function."
                )

            label_vals = temp_df[label_col].to_numpy()

            # "Hard" invalid: insufficient future data, no direction
            # information at all -- always excluded, no redistribution
            # possible.
            hard_invalid_mask = (label_vals == common.Signal.INVALID)
            usable_vals = label_vals[~hard_invalid_mask]
            n_usable = len(usable_vals)

            n_amb = 0
            if n_usable > 0:
                n_pos = float(np.sum(usable_vals == common.Signal.POSITIVE))
                n_neg = float(np.sum(usable_vals == common.Signal.NEGATIVE))
                n_neu = float(np.sum(usable_vals == common.Signal.NEUTRAL))
                n_amb = float(np.sum(usable_vals == common.Signal.AMBIGUOUS))

                n_pos_adj = n_pos + 0.5 * n_amb
                n_neg_adj = n_neg + 0.5 * n_amb

                p_long = n_pos_adj / n_usable
                p_short = n_neg_adj / n_usable
                p_neutral = n_neu / n_usable
            else:
                p_long = p_short = p_neutral = 0.0

            row = {
                "parameter": x_name,
                "x": float(x),
                "stop_multiplier_rate_long": para.stop_multiplier_rate_long,
                "stop_multiplier_rate_short": para.stop_multiplier_rate_short,
                "label_type": para.label_type,
                "para_type": para.para_type,
                "vol_multiplier": float(para.vol_multiplier_long),
                "horizon": int(para.predict_num),
                "valid_count": int(n_usable),
                "ambiguous_count": int(n_amb),
                "p_short": float(p_short),
                "p_neutral": float(p_neutral),
                "p_long": float(p_long),
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

        self.results_df = pd.DataFrame(sweep_data).sort_values("x").reset_index(drop=True)
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
        Dispatch to the appropriate null model depending on the labeling
        method being swept:
          - FTHL: terminal-return distribution (endpoint-only, matches the
            fixed-time-horizon label definition).
          - TBM : path-dependent first-passage (touch-before-horizon)
            distribution (matches the triple-barrier label definition).
        These are genuinely different null models -- reusing the FTHL
        formula for TBM compares path-dependent labels against a
        reference derived for an endpoint-only rule, understating the
        true null-model touch probability (by roughly a factor of ~2 in
        the one-sided case, via the reflection principle).
        """
        x = np.asarray(x, dtype=float)
        if self.label_type == "tbm":
            return self._tbm_gaussian_reference(x, para)
        return self._fthl_gaussian_reference(x, para)

    def _fthl_gaussian_reference(self, x, para):
        """
        Zero-mean Gaussian null model for FTHL: terminal return
        r_{t,h} ~ N(0, h * sigma_t^2).

        volatility sweep:
            z = vol_multiplier

        horizon sweep:
            z = fixed_vol_multiplier / sqrt(horizon)

        Then:
            P(pos) = 1 - Phi(z)
            P(neg) = Phi(-z) = 1 - Phi(z)
            P(neu) = Phi(z) - Phi(-z)
        """
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

    def _tbm_gaussian_reference(self, x, para):
        """
        Zero-mean Gaussian null model for a SYMMETRIC TBM: probability
        that a driftless random walk with per-step std sigma_t touches
        either of two symmetric barriers at +-tau_t = +-k*sigma_t within
        h steps.

        This is the classical two-sided first-passage ("Brownian motion
        escaping a symmetric strip") problem. With z = tau_t / sigma_t = k
        (the barrier distance in units of per-step volatility, independent
        of h), the probability that NEITHER barrier is touched within h
        steps is given by the eigenfunction-expansion solution of the
        heat equation with Dirichlet boundary conditions at +-k:

            P_neutral(k, h) = (4/pi) * sum_{n=0}^inf [(-1)^n / (2n+1)]
                                * exp( -(2n+1)^2 * pi^2 * h / (8 * k^2) )

        By symmetry (no drift, symmetric barriers), the probability of
        touching the upper barrier first equals the probability of
        touching the lower barrier first, each = (1 - P_neutral) / 2.

        This closed form was numerically validated against a
        fine-grained Monte Carlo simulation of the underlying random
        walk (sub-stepping to approximate continuous monitoring) before
        being adopted here; a coarse single-step-per-bar simulation
        under-samples path extrema and will NOT match this formula
        (known discrete-monitoring bias in first-passage problems).

        Validity: this closed form assumes a single pair of symmetric
        barriers, i.e. vol_multiplier_long == vol_multiplier_short AND
        stop_multiplier_rate_long == stop_multiplier_rate_short == 1.0.
        A mismatch triggers a warning; the curve is then only an
        approximation and should not be used for crossing-point analysis
        without noting this caveat.
        """
        stop_long = getattr(para, "stop_multiplier_rate_long", None)
        stop_short = getattr(para, "stop_multiplier_rate_short", None)
        symmetric = (
            stop_long == 1.0
            and stop_short == 1.0
            and np.isclose(para.vol_multiplier_long, para.vol_multiplier_short)
        )
        if not symmetric:
            import warnings
            warnings.warn(
                "TBM Gaussian null model assumes symmetric barriers "
                "(vol_multiplier_long == vol_multiplier_short and "
                "stop_multiplier_rate_long == stop_multiplier_rate_short == 1.0); "
                "current parameters do not satisfy this, so the null curve "
                "is only an approximation.",
                stacklevel=2,
            )

        if self.para_type == "volatility":
            k = x
            h = np.full_like(x, float(para.predict_num))
        elif self.para_type == "horizon":
            k = np.full_like(x, float(para.vol_multiplier_long))
            h = x
        else:
            raise ValueError(f"Unsupported para_type: {self.para_type}")

        p_neutral = self._tbm_symmetric_survival_prob(k, h)
        p_positive = (1.0 - p_neutral) / 2.0
        p_negative = (1.0 - p_neutral) / 2.0

        return {
            "positive": p_positive,
            "negative": p_negative,
            "neutral": p_neutral,
        }

    @staticmethod
    def _tbm_symmetric_survival_prob(k, h, n_terms=200):
        """
        Closed-form no-touch probability for a driftless random walk with
        symmetric absorbing barriers at +-k (in units of per-step std),
        observed over h steps. See `_tbm_gaussian_reference` docstring
        for the formula and its derivation.
        """
        k = np.asarray(k, dtype=float)
        h = np.asarray(h, dtype=float)

        k_safe = np.where(k > 1e-8, k, 1e-8)
        h_safe = np.where(h > 0, h, 1e-8)

        n = np.arange(n_terms)
        k_b = k_safe[..., None]
        h_b = h_safe[..., None]
        coeff = ((-1.0) ** n) / (2 * n + 1)
        exponent = -((2 * n + 1) ** 2) * (np.pi ** 2) * h_b / (8.0 * k_b ** 2)
        terms = coeff * np.exp(exponent)
        survival = (4.0 / np.pi) * np.sum(terms, axis=-1)
        return np.clip(survival, 0.0, 1.0)

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

    @staticmethod
    def _auto_ylim(*ys, lower=0.0, padding_ratio=0.10, min_upper=0.02, max_upper=None):
        valid_values = []

        for y in ys:
            if y is None:
                continue

            arr = np.asarray(y, dtype=float)
            arr = arr[np.isfinite(arr)]

            if arr.size > 0:
                valid_values.append(arr)

        if not valid_values:
            return None

        values = np.concatenate(valid_values)
        y_max = np.max(values)

        upper = max(y_max * (1.0 + padding_ratio), min_upper)

        if max_upper is not None:
            upper = min(upper, max_upper)

        return lower, upper

    @staticmethod
    def _find_crossings(x, y_a, y_b, max_points=2):
        """
        Find where curve y_a crosses curve y_b, walking left to right along x.

        Uses linear interpolation between the two samples that bracket a sign
        change of (y_a - y_b), so the reported x_cross is not restricted to
        the discrete sample grid. Also handles the (rare) case where a sample
        lands exactly on the crossing (diff == 0).

        Only the first `max_points` crossings are returned (ordered by x),
        since higher-order crossings on noisy curves (e.g. a second
        derivative) tend to be numerical noise rather than meaningful
        structure.

        Returns
        -------
        list of dict, each with keys: x, y, index_left (the sample index
        immediately before the crossing, useful for debugging/reporting).
        """
        x = np.asarray(x, dtype=float)
        y_a = np.asarray(y_a, dtype=float)
        y_b = np.asarray(y_b, dtype=float)

        diff = y_a - y_b
        finite = np.isfinite(diff) & np.isfinite(x)

        crossings = []
        idx = np.where(finite)[0]

        for k in range(len(idx) - 1):
            i0, i1 = idx[k], idx[k + 1]
            d0, d1 = diff[i0], diff[i1]

            if d0 == 0.0:
                crossings.append({"x": float(x[i0]), "y": float(y_a[i0]), "index_left": int(i0)})
            elif d0 * d1 < 0.0:
                t = d0 / (d0 - d1)  # fraction of the way from i0 to i1 where diff hits 0
                x_c = x[i0] + t * (x[i1] - x[i0])
                y_c = y_a[i0] + t * (y_a[i1] - y_a[i0])
                crossings.append({"x": float(x_c), "y": float(y_c), "index_left": int(i0)})

            if len(crossings) >= max_points:
                break

        return crossings[:max_points]

    def _annotate_crossings(self, ax, crossings, color="black", prefix="", direction="up"):
        """
        Draw a marker + vertical guide line + text label for each crossing
        point found by `_find_crossings`. Only meant to decorate up to 2
        points, so labels are kept short ("1st crossing" / "2nd crossing").

        `direction` fixes which side of the point the label sits on
        ("up" or "down") for the WHOLE series passed in this call. This
        must be tied to series identity (e.g. positive vs negative),
        not to crossing order (1st vs 2nd) -- otherwise two different
        series' 1st crossings both default to the same offset direction
        and their labels overlap whenever the two crossing points are
        close together in (x, y), which happens routinely for
        positive/negative label-ratio curves crossing a shared Gaussian
        reference. Within one direction, the 1st and 2nd crossing of the
        SAME series are still separated via increasing horizontal/
        vertical offset so they don't stack on each other either.
        """
        ordinal_labels = ["1st crossing", "2nd crossing"]
        sign = 1 if direction == "up" else -1
        offsets = [(12, sign * 14), (28, sign * 26)]

        for i, c in enumerate(crossings[:2]):
            label = f"{prefix}{ordinal_labels[i]}\nx={c['x']:.3f}"
            ax.axvline(c["x"], color=color, linestyle=":", linewidth=1.1, alpha=0.6)
            ax.scatter(
                [c["x"]], [c["y"]],
                s=70, color=color, edgecolor="black", linewidth=0.8, zorder=5,
            )
            ax.annotate(
                label,
                xy=(c["x"], c["y"]),
                xytext=offsets[i],
                textcoords="offset points",
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.8),
            )

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

    def _sweep_df(self):
        """
        Return the sweep results, sorted by x. Replaces the old
        `_single_stop_subset(stop_rate)` -- since stop_multiplier_rate
        is now a single fixed value per analyzer instance (not swept),
        there is nothing left to filter by; results_df already contains
        exactly one row per swept x value.
        """
        if self.results_df is None:
            raise RuntimeError("Please run run_parameter_sweep() first.")
        return self.results_df.sort_values("x").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plot public API
    # ------------------------------------------------------------------
    def plot_all_label_ratio_curves(
        self,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
        annotate_crossings=True,
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

        saved_paths = {}

        # for class_name in ["positive", "negative", "neutral"]:
        #     saved_paths[class_name] = self.plot_single_label_ratio_curve(
        #         class_name=class_name,
        #         output_dir=output_dir,
        #         smooth_window=smooth_window,
        #         include_gaussian=include_gaussian,
        #     )

        saved_paths["positive_negative"] = self.plot_positive_negative_ratio_curves(
            output_dir=output_dir,
            smooth_window=smooth_window,
            include_gaussian=include_gaussian,
            annotate_crossings=annotate_crossings,
        )

        saved_paths["neutral"] = self.plot_single_label_ratio_curve(
            class_name="neutral",
            output_dir=output_dir,
            smooth_window=smooth_window,
            include_gaussian=include_gaussian,
            annotate_crossings=annotate_crossings,
        )

        overview_path = self.plot_distribution_overview(
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
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
        annotate_crossings=True,
    ):
        """Plot ratio, first derivative, and second derivative for one class."""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if class_name not in self.CLASS_CONFIG:
            raise ValueError(f"class_name must be one of {list(self.CLASS_CONFIG)}")

        cfg = self.CLASS_CONFIG[class_name]
        df_plot = self._sweep_df()

        x, empirical, gaussian, d1_emp, d1_gau, d2_emp, d2_gau = self._curve_data(
            df_plot=df_plot,
            empirical_col=cfg["col"],
            gaussian_col=cfg["gaussian_col"],
            smooth_window=smooth_window,
        )

        prefix = f"{class_name}"
        saved = {}
        crossings_summary = []

        ratio_y_lim = self._auto_ylim(
            empirical,
            gaussian if include_gaussian else None,
            lower=0.0,
            padding_ratio=0.10,
            min_upper=0.02,
            max_upper=1.05,
        )

        saved["ratio"], crossings = self._plot_two_curves(
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
            y_lim=ratio_y_lim,
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "ratio", "crossings": crossings})

        saved["first_derivative"], crossings = self._plot_two_curves(
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
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "first_derivative", "crossings": crossings})

        saved["second_derivative"], crossings = self._plot_two_curves(
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
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "second_derivative", "crossings": crossings})

        self._save_crossings_csv(
            crossings_summary, output_dir, prefix=f"{prefix}_crossings"
        )

        return saved

    def plot_positive_negative_ratio_curves(
        self,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
        annotate_crossings=True,
    ):
        """
        Plot positive and negative label-ratio curves together.

        Generates three plots:
            1) positive/negative ratio curve
            2) first derivative curve
            3) second derivative curve

        The Gaussian reference curve is shared by positive and negative labels
        under the zero-mean Gaussian assumption.
        """
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        df_plot = self._sweep_df()

        pos_cfg = self.CLASS_CONFIG["positive"]
        neg_cfg = self.CLASS_CONFIG["negative"]

        (
            x,
            pos_emp,
            pos_gau,
            pos_d1_emp,
            pos_d1_gau,
            pos_d2_emp,
            pos_d2_gau,
        ) = self._curve_data(
            df_plot=df_plot,
            empirical_col=pos_cfg["col"],
            gaussian_col=pos_cfg["gaussian_col"],
            smooth_window=smooth_window,
        )

        (
            _,
            neg_emp,
            neg_gau,
            neg_d1_emp,
            neg_d1_gau,
            neg_d2_emp,
            neg_d2_gau,
        ) = self._curve_data(
            df_plot=df_plot,
            empirical_col=neg_cfg["col"],
            gaussian_col=neg_cfg["gaussian_col"],
            smooth_window=smooth_window,
        )

        # Under the zero-mean Gaussian reference:
        # g_positive == g_negative.
        # Use one shared Gaussian curve.
        gaussian = pos_gau
        d1_gaussian = pos_d1_gau
        d2_gaussian = pos_d2_gau

        prefix = f"positive_negative"
        saved = {}
        crossings_summary = []

        ratio_y_lim = self._auto_ylim(
            pos_emp,
            neg_emp,
            gaussian if include_gaussian else None,
            lower=0.0,
            padding_ratio=0.10,
            min_upper=0.02,
            max_upper=1.05,
        )

        saved["ratio"], crossings = self._plot_positive_negative_with_gaussian(
            x=x,
            y_pos=pos_emp,
            y_neg=neg_emp,
            y_gau=gaussian if include_gaussian else None,
            title=f"Positive and Negative Label Ratios vs {self._x_label()}",
            ylabel="Label ratio",
            output_path=os.path.join(output_dir, f"{prefix}_ratio.png"),
            y_lim=ratio_y_lim,
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "ratio", "crossings": crossings})

        saved["first_derivative"], crossings = self._plot_positive_negative_with_gaussian(
            x=x,
            y_pos=pos_d1_emp,
            y_neg=neg_d1_emp,
            y_gau=d1_gaussian if include_gaussian else None,
            title="First Derivative of Positive and Negative Label Ratios",
            ylabel="First derivative",
            output_path=os.path.join(output_dir, f"{prefix}_d1.png"),
            add_zero_line=True,
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "first_derivative", "crossings": crossings})

        saved["second_derivative"], crossings = self._plot_positive_negative_with_gaussian(
            x=x,
            y_pos=pos_d2_emp,
            y_neg=neg_d2_emp,
            y_gau=d2_gaussian if include_gaussian else None,
            title="Second Derivative of Positive and Negative Label Ratios",
            ylabel="Second derivative",
            output_path=os.path.join(output_dir, f"{prefix}_d2.png"),
            add_zero_line=True,
            annotate_crossings=annotate_crossings,
        )
        crossings_summary.append({"curve": "second_derivative", "crossings": crossings})

        self._save_crossings_csv(
            crossings_summary, output_dir, prefix=f"{prefix}_crossings"
        )

        return saved

    def plot_distribution_overview(
        self,
        output_dir=None,
        smooth_window=3,
        include_gaussian=True,
    ):
        """One overview figure with positive / negative / neutral ratios."""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        df_plot = self._sweep_df()
        x = df_plot["x"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(12, 7))

        for class_name in ["positive", "negative", "neutral"]:
            cfg = self.CLASS_CONFIG[class_name]
            y_emp = self._smooth(df_plot[cfg["col"]].to_numpy(dtype=float), smooth_window)
            ax.plot(
                x,
                y_emp,
                marker="o",
                markersize=4,
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
            f"distribution_overview.png",
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
        annotate_crossings=True,
    ):
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(
            x,
            y_emp,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=empirical_label,
        )

        crossings = []
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

            if annotate_crossings:
                crossings = self._find_crossings(x, y_emp, y_gau, max_points=2)
                self._annotate_crossings(ax, crossings, color="tab:blue")

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
        if crossings:
            xs = ", ".join(f"{c['x']:.3f}" for c in crossings)
            # print(f"   ↳ crossing x-values (first {len(crossings)}): {xs}")
        return output_path, crossings

    def _plot_positive_negative_with_gaussian(
        self,
        x,
        y_pos,
        y_neg,
        y_gau,
        title,
        ylabel,
        output_path,
        y_lim=None,
        add_zero_line=False,
        annotate_crossings=True,
    ):
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(
            x,
            y_pos,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="green",
            label="Positive empirical",
        )

        ax.plot(
            x,
            y_neg,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="red",
            label="Negative empirical",
        )

        crossings = {"positive_vs_gaussian": [], "negative_vs_gaussian": []}

        if y_gau is not None:
            ax.plot(
                x,
                y_gau,
                linestyle="--",
                linewidth=2.0,
                color="black",
                alpha=0.8,
                label="Gaussian reference",
            )

            if annotate_crossings:
                pos_crossings = self._find_crossings(x, y_pos, y_gau, max_points=2)
                neg_crossings = self._find_crossings(x, y_neg, y_gau, max_points=2)
                crossings["positive_vs_gaussian"] = pos_crossings
                crossings["negative_vs_gaussian"] = neg_crossings

                self._annotate_crossings(ax, pos_crossings, color="darkgreen", prefix="Pos ", direction="up")
                self._annotate_crossings(ax, neg_crossings, color="darkred", prefix="Neg ", direction="down")

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
        if crossings["positive_vs_gaussian"]:
            xs = ", ".join(f"{c['x']:.3f}" for c in crossings["positive_vs_gaussian"])
            # print(f"   ↳ positive-vs-Gaussian crossing x-values: {xs}")
        if crossings["negative_vs_gaussian"]:
            xs = ", ".join(f"{c['x']:.3f}" for c in crossings["negative_vs_gaussian"])
            # print(f"   ↳ negative-vs-Gaussian crossing x-values: {xs}")
        return output_path, crossings

    @staticmethod
    def _save_crossings_csv(crossings_summary, output_dir, prefix):
        """
        Flatten the crossings collected across a group of plots (ratio/d1/d2,
        possibly split by positive/negative) into one tidy CSV for reporting
        in the dissertation, instead of only relying on the annotated PNGs.
        """
        rows = []
        for entry in crossings_summary:
            curve_name = entry["curve"]
            c = entry["crossings"]

            if isinstance(c, dict):
                # positive/negative split: {"positive_vs_gaussian": [...], "negative_vs_gaussian": [...]}
                for series_name, points in c.items():
                    for order, point in enumerate(points, start=1):
                        rows.append(
                            {
                                "curve": curve_name,
                                "series": series_name,
                                "crossing_order": order,
                                "x": point["x"],
                                "y": point["y"],
                            }
                        )
            else:
                # single empirical-vs-gaussian list
                for order, point in enumerate(c, start=1):
                    rows.append(
                        {
                            "curve": curve_name,
                            "series": "empirical_vs_gaussian",
                            "crossing_order": order,
                            "x": point["x"],
                            "y": point["y"],
                        }
                    )

        if not rows:
            return None

        out_path = os.path.join(output_dir, f"{prefix}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"✅ Crossings summary saved: {out_path}")
        return out_path