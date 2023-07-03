from typing import Callable

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import _box, _violin
from scipy import stats


class ModeNotImplementedError(ValueError):
    pass


def add_significance_indicator(
    fig: go.Figure,
    same_color_only: bool = False,
    xval_pairs: list[tuple] | None = None,
    color_pairs: list[tuple] | None = None,
    stat_func: Callable = stats.ttest_ind,
    p_quantiles: tuple = (0.05, 0.01),
    x_offset_inc: float = 0.13,
) -> go.Figure:
    """
    Add significance indicators between box or violin plots

    Parameters
    ----------
    fig : go.Figure
        the figure to add the indicators to
    same_color_only: bool (True)
        only calculate significance between the same colors (legendgroups)
    xval_pairs: list[tuple] | None (None)
        specify pairs to consider for the significance calculation, of None,
        all combinations will be considered
    color_pairs: list[tuple] | None (None)
        specify colors to consider for the significance calculation, of None,
        all combinations will be considered.
        Only used if same_color_only == False.
    sig_func: Callable (scipy.stats.ttest_ind)
        the significance function to consider
    p_quantiles: tuple[float, float] ((0.05, 0.01))
        the quantiles to be considered for labeling with `*`, `**`, etc.
    x_offset_inc: float (0.05)
        basic offset between the legendgroups as this value cannot be retrieved
        from the traces...

    Returns
    -------
    fig : go.Figure
        the figure with significance indicators added
    """

    # Consider only box and violin plots as distributions
    dists = [
        elm
        for elm in fig.data
        if isinstance(elm, _box.Box) or isinstance(elm, _violin.Violin)
    ]

    # extend to single distributions (one for each x value)
    xmap = get_map_xcat_to_linspace(fig)
    sdists = pd.DataFrame(
        [
            {
                "lgrp": elm["legendgroup"],
                "x": xmap[xval],
                "xlabel": xval,
                "y": elm["y"][elm["x"] == xval],
            }
            for elm in dists
            for xval in np.unique(elm["x"])
        ]
    )
    # Make sure the x axis is reflected as numeric as we cannot draw lines
    # with offsets otherwise
    fig = make_xaxis_numeric(fig)

    if same_color_only:
        color_pairs = [(cp, cp) for cp in sdists.lgrp.unique()]
    elif color_pairs is None:
        # get all pairs
        color_pairs = [
            (cp1, cp2)
            for i, cp1 in enumerate(sdists.lgrp.unique())
            for cp2 in sdists.lgrp.unique()[i:]
        ]

    dstats = compute_stats(sdists, xval_pairs, color_pairs, stat_func)

    # space occupied in min max range by each line
    line_width_frac = 0.05

    ymin = min([e.min() for e in sdists.y])
    ymax = max([e.max() for e in sdists.y])
    dy = ymax - ymin
    # draw the indicator lines
    yline = ymin - dy * line_width_frac
    for rowi, (c1, c2, x1, x2, (stat, pval), n1, n2) in dstats.iterrows():
        x1_offset = get_x_offset(fig, c1, x_offset_inc)
        x2_offset = get_x_offset(fig, c2, x_offset_inc)
        x1p = xmap[x1] + x1_offset
        x2p = xmap[x2] + x2_offset
        xmid = x1p + (x2p - x1p) / 2

        msk = [pval < pq for pq in p_quantiles]
        if not any(msk):
            sig_label = "ns"
        elif all(msk):
            sig_label = "*" * len(msk)
        else:
            # get the first False
            sig_label = "*" * msk.index(False)

        # the line
        fig.add_trace(
            go.Scatter(
                x=[x1p, x2p],
                y=[yline, yline],
                mode="lines+markers",
                marker={"size": 10, "symbol": "line-ns", "line_width": 2},
                line_color="#555555",
                showlegend=False,
                hoverinfo="skip",  # disable hover
            )
        )

        # Invisible marker for hover
        hovertemplate = (
            f"<b>test function</b>: {stat_func.__name__}"
            f"<br><b>N-dist1</b>: {n1}<br><b>N-dist2</b>: {n2}<br>"
            f"<b>statistic</b>: {stat}<br><b>pval</b>: {pval}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=[xmid],
                y=[yline],
                mode="markers",
                showlegend=False,
                marker_color="#ffffff",
                opacity=0,
                hovertemplate=hovertemplate,
            )
        )
        # Annotation to be able to set background color
        fig.add_annotation(
            x=xmid,
            y=yline,
            text=sig_label,
            bgcolor="#ffffff",
            bordercolor="#888888",
            ax=0,
            ay=0,
            opacity=0.9,
        )

        yline -= dy * line_width_frac

    return fig


def get_num_x_pos(fig: go.Figure, xkey: str) -> float:
    """Get the numeric position for an x value on a categorical x axis"""
    xmap = get_map_xcat_to_linspace(fig)
    return xmap[xkey]


def get_x_offset(fig: go.Figure, cg_key: str, x_offset_inc: float) -> float:
    """Compute the x axis offset for a given color group"""

    # via dict to preserver order
    cgrps = list(
        dict.fromkeys(
            [trc.offsetgroup for trc in fig.data if "offsetgroup" in trc]
        )
    )

    if len(cgrps) == 0 or len(cgrps) == 1:
        return 0
    else:
        extend = (len(cgrps) - 1) / 2
        offsets = np.linspace(-extend, extend, len(cgrps)) * (
            x_offset_inc / extend
        )
        offsetmap = {k: v for k, v in zip(cgrps, offsets)}

        return offsetmap[cg_key]


def compute_stats(
    sdists: pd.DataFrame,
    xval_pairs: list[tuple] | None,
    color_pairs: list[tuple],
    stat_func: Callable,
) -> pd.DataFrame:
    """
    Compute a data frame storing the test statistic for
    color1, color2, x1, x2 comparison tuples
    """
    recs = []
    for cp1, cp2 in color_pairs:
        if xval_pairs is not None:
            wxval_pairs = xval_pairs
        else:
            # Build unique pairs accross the color groups
            if cp1 == cp2:
                uxvals = sdists[(sdists.lgrp == cp1)].xlabel.unique()
                wxval_pairs = [
                    (x1, x2)
                    for i, x1 in enumerate(uxvals)
                    for x2 in uxvals[i + 1 :]
                ]
            else:
                # consider all unique
                wxval_pairs = [
                    (x1, x2)
                    for x1 in sdists[(sdists.lgrp == cp1)].xlabel.unique()
                    for x2 in sdists[(sdists.lgrp == cp2)].xlabel.unique()
                ]

        for x1, x2 in wxval_pairs:
            if x1 != x2 or cp1 != cp2:
                dist1 = sdists[(sdists.lgrp == cp1) & (sdists.xlabel == x1)]
                dist2 = sdists[(sdists.lgrp == cp2) & (sdists.xlabel == x2)]

                if True:  # dist1.shape[0] >= 1 and dist2.shape[0] >= 1:
                    # print(f" >> {dist1=}, {dist2=}, {x1=}, {x2=}, {cp1=}, "
                    # f"{cp2=}")
                    recs.append(
                        {
                            "color1": cp1,
                            "color2": cp2,
                            "x1": x1,
                            "x2": x2,
                            "stat": stat_func(
                                dist1.y.iloc[0], dist2.y.iloc[0]
                            ),
                            "n1": len(dist1.y.iloc[0]),
                            "n2": len(dist2.y.iloc[0]),
                        }
                    )

    return pd.DataFrame(recs)


def make_xaxis_numeric(fig: go.Figure) -> go.Figure:
    xmap = get_map_xcat_to_linspace(fig)
    # replace x for each trace
    for trc in fig.data:
        trc.x = np.asarray([xmap[x] for x in trc.x])

    # make axis labels reflect the categories again
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(xmap.values()),
            ticktext=list(xmap.keys()),
        )
    )

    return fig


def get_map_xcat_to_linspace(
    fig: go.Figure, xmin: int = 0, xmax: int = 1
) -> dict:
    xgrps = list(
        dict.fromkeys(
            [u for trc in fig.data for u in np.unique(trc.x) if "x" in trc]
        )
    )
    xpos = np.linspace(0, 1, len(xgrps))
    return {k: v for k, v in zip(xgrps, xpos)}


# Cluster permutation results to a plotly plot
def add_cluster_permut_sig_to_plotly(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    fig: go.Figure,
    xaxes_vals: [None, list, np.ndarray] = None,  # noqa
    row: [None, int] = None,
    col: [None, int] = None,
    pval: float = 0.05,
    nperm: int = 1024,
    mode: str = "p_colorbar",
    showlegend: bool = False,
) -> go.Figure:
    """Add a cluster permutation significance indicator to a plotly figure
    Parameters
    ----------
    curves_a : np.ndarray
        the first set or curves (time on axis=1)
    curves_b : np.ndarray
        the second set of curves set or curves (time on axis=1)
    fig : FigureWidget
        the plotly figure to modify
    xaxes_vals : None, list, np.ndarray
        time values to use for the time dimension
    row : None, int
        the subplot row of the plot to modify
    col : None, int
        the subplot col of the plot to modify
    pval : float
        the pvalue to use as critical value for significance. Note this is used
        for both, finding the significant F-values and for finding significant
        clusters
    mode : str
        how to plot the cluster test statistics, options are:
            'p_bg': a background if p < ptarget
            'spark': sparklines of the rvalue itself
    showlegend : bool
        whether to show the legend

    Returns
    -------
    fig : FigureWidget
        the modified figure
    """
    # --> understand the correct degrees of freedom
    n_conditions = 2
    n_observations = max(curves_a.shape[0], curves_b.shape[0])
    dfn = n_conditions - 1  # degrees of freedom numerator
    dfd = n_observations - n_conditions  # degrees of freedom denominator
    thresh = stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

    print(
        f"Calculating cluster permutation in F-stats with {thresh=} and"
        f" {nperm=}."
    )

    # the last parameter should be relevant for the adjecency -> here time
    fobs, clust, pclust, h0 = mne.stats.permutation_cluster_test(
        [
            curves_a.reshape(*curves_a.shape, 1),
            curves_b.reshape(*curves_b.shape, 1),
        ],
        threshold=thresh,
        n_permutations=nperm,
    )

    time = (
        xaxes_vals if xaxes_vals is not None else np.arange(curves_a.shape[1])
    )

    # dbfig = debug_plot(curves_a, curves_b, fobs, h0, thresh)
    # dbfig.savefig("dbfig_test.png")

    if mode == "spark":
        if row is None or col is None:
            print(
                "Plotting the F-values as spark lines is only possible if "
                f"{row=} and {col=} are defined"
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=fobs[:, 0],
                    name="F-values",
                    mode="lines",
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=[time[0], time[-1]],
                    y=[thresh, thresh],
                    name="F-val thresh",
                    mode="lines",
                    line_color="#338833",
                    line_dash="dash",
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )
    elif mode == "p_colorbar":
        # color the background
        for cl, p in zip(clust, pclust):
            if p < pval:
                x = time[cl[0][:]]
                fig.add_vrect(
                    x0=x[0],
                    x1=x[-1],
                    line_width=1,
                    line_color="#338833",
                    fillcolor="#338833",
                    name="",
                    opacity=0.2,
                    row=row,
                    col=col,
                )

    else:
        raise ModeNotImplementedError(f"Unknown {mode=} for adding signific.")

    return fig


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame(np.random.randn(200, 2), columns=["a", "b"])

    df["label"] = np.random.choice(["aa", "bb", "cc"], 200)
    df["color"] = np.random.choice(["xx", "yy", "zz"], 200)

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(fig, x_offset_inc=0.12)
    fig.show()

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(
        fig,
        x_offset_inc=0.12,
        same_color_only=False,
        color_pairs=[("xx", "xx")],
    )
    fig.show()

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(
        fig,
        x_offset_inc=0.12,
        same_color_only=False,
        color_pairs=[("xx", "zz")],
    )
    fig.show()

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(
        fig, x_offset_inc=0.12, same_color_only=False
    )
    fig.show()

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(
        fig,
        x_offset_inc=0.12,
        same_color_only=False,
        xval_pairs=[("aa", "aa"), ("aa", "cc")],
    )
    fig.show()

    fig = px.box(df, x="label", y="a", color="color")
    fig = add_significance_indicator(
        fig,
        x_offset_inc=0.12,
        same_color_only=False,
        color_pairs=[("xx", "zz"), ("xx", "yy")],
        xval_pairs=[("aa", "aa"), ("aa", "cc")],
    )
    fig.show()
