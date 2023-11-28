import numpy as np
import polars as pl


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def fill_in_gaps(preds: np.ndarray, n=1) -> np.ndarray:
    shifted_pos_1_preds = shift(preds, 1, fill_value=0)
    shifted_neg_1_preds = shift(preds, -1, fill_value=0)
    shifted_neg_2_preds = shift(preds, -2, fill_value=0)
    # fill in the gaps
    stacked_preds = np.stack([preds, shifted_neg_1_preds, shifted_neg_2_preds])
    stacked_preds = stacked_preds.T
    mask = stacked_preds != 0
    lone_neg = mask[:, 0] & ~mask[:, 1] & mask[:, 2]
    # shift this mask to account for the fact that we shifted the preds
    lone_neg = shift(lone_neg, 1, fill_value=False)
    # find the values around the lone_neg position and get their average
    mean_around_lone_neg = np.stack(
        [shifted_pos_1_preds[lone_neg], shifted_neg_1_preds[lone_neg]]
    ).mean(axis=0)
    preds[lone_neg] = mean_around_lone_neg
    lone_pos = mask[:, 1] & ~mask[:, 0] & ~mask[:, 2]
    lone_pos = shift(lone_pos, 1, fill_value=False)
    preds[lone_pos] = 0
    return preds


def drop_short_events(
    event_df: pl.DataFrame, threshold: int = 1000
) -> pl.DataFrame:
    group = np.arange(len(event_df) // 2)
    group = np.repeat(group, 2)
    event_df = event_df.with_columns(pl.Series(name="group", values=group))
    event_df = (
        event_df.pivot(
            index=["series_id", "group"],
            columns=["event"],
            values=["step"],
        )
        .with_columns((pl.col("wakeup") - pl.col("onset")).alias("duration"))
        .filter(pl.col("duration") > threshold)
    )
    # revert event_df to original format
    event_df = event_df.melt(
        id_vars=["series_id", "group"],
        value_vars=["onset", "wakeup"],
        variable_name="event",
        value_name="step",
    )
    event_df = event_df.drop("group")
    return event_df


def post_process_for_prec(
    preds: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray):
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids: list = list(preds.keys())

    records = []
    for series_id in series_ids:
        this_series_preds = np.array(preds[series_id])
        this_series_preds = fill_in_gaps(this_series_preds, 1)
        this_series_preds_round = np.where(this_series_preds > threshold, 1, 0)
        this_series_preds_diff = np.diff(this_series_preds_round)
        # onset is when diff is 1 (index + 1)
        # wakeup is when diff is -1 (index - 1)
        onset_steps = np.where(this_series_preds_diff == 1)[0] + 1
        wakeup_steps = np.where(this_series_preds_diff == -1)[0]

        for i, event_name in enumerate(["onset", "wakeup"]):
            if event_name == "onset":
                steps = onset_steps
            else:
                steps = wakeup_steps
            scores = this_series_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:
        records.append(
            {
                "series_id": series_ids[0],
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    filtered_sub_df = drop_short_events(sub_df, threshold=1000)
    # drop the event column from sub_df
    sub_df = sub_df.drop("event")
    filtered_sub_df = filtered_sub_df.join(sub_df, on=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(filtered_sub_df)))
    filtered_sub_df = filtered_sub_df.with_columns(row_ids).select(
        ["row_id", "series_id", "step", "event", "score"]
    ).sort(by=["series_id", "step"])
    return filtered_sub_df
