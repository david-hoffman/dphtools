#!/usr/bin/env python
# -*- coding: utf-8 -*-
# beads.py
"""
Bead specific functons.

Copyright (c) 2021, David Hoffman
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def remove_coord_mean(df, *, coords=["x0", "y0"]):
    """Remove the mean value of the coordinates."""
    df_new = df.copy()
    df_new[coords] = df_new[coords].astype(np.float)
    coord_mean = df_new[coords].mean()
    df_new[coords] -= coord_mean
    return df_new.dropna()


def calc_drift(
    fiducials_df,
    *,
    coords=["x0", "y0"],
    frame_name="slice",
    weighted="amp",
    diagnostics=False,
    frames_index=None
):
    """Calculate image drift from multiple emitters in a FOV.
    
    Given a list of DataFrames with each DF containing the coordinates
    of a single fiducial calculate the mean or weighted mean of the coordinates
    in each frame.
    """
    if len(fiducials_df) == 1:
        # if there is only one fiducial then return that
        logger.debug("Only on fiducial passed to calc_drift")
        toreturn = remove_coord_mean(fiducials_df[0])[coords]
    else:
        mean_removed = [remove_coord_mean(ff) for ff in fiducials_df]
        if diagnostics:
            # debugging diagnostics
            _, axs = plt.subplots(len(coords))
            for ff in mean_removed:
                for coord, ax in zip(coords, axs.ravel()):
                    ff[coord].plot(ax=ax)

        # want to do a weighted average
        # need to reset_index after concatination so that all localzations have unique ID
        # this will make weighting easier down the line.
        df_means = pd.concat(mean_removed).dropna().reset_index()

        # if weighted is something, use that as the weights for the mean
        # if weighted is not a valid column name then it will raise an
        # exception
        if weighted.lower() == "coords":
            # save the labels for weighted coords and weights
            w_coords = []
            weights = []
            # loop through coords generating weights and weighted coords
            for x in coords:
                c = x[0]
                s = "sigma_" + c
                df_means[s + "_inv"] = 1 / df_means[s] ** 2
                weights.append(s + "_inv")
                df_means[x + "_w"] = df_means[x].mul(df_means[s + "_inv"], "index")
                w_coords.append(x + "_w")
            # groupby group_id and sum
            temp_gb = df_means.groupby(frame_name)
            # finish weighted mean
            new_coords = temp_gb[w_coords].sum() / temp_gb[weights].sum().values
            new_coords.columns = [c.replace("_w", "") for c in new_coords.columns]
            # calc new sigma
            # return new data frame
            toreturn = new_coords
        elif weighted:
            # weight the coordinates
            logger.debug("Weighting by {}".format(weighted))
            df_means[coords] = df_means[coords].mul(df_means[weighted], "index")
            # groupby frame
            temp = df_means.groupby(frame_name)
            # calc weighted average
            toreturn = temp[coords].sum().div(temp[weighted].sum(), "index")
        else:
            toreturn = df_means.groupby(frame_name)[coords].mean()
        # remove mean of total drift.
        toreturn = remove_coord_mean(toreturn)
    if diagnostics:
        toreturn.plot(subplots=True)
    if frames_index is None:
        return toreturn
    else:
        assert frames_index.name == frame_name
        return toreturn.reindex(frames_index).interpolate(limit_direction="both")
