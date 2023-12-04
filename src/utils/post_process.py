import numpy as np
import polars as pl

from src.conf import InferenceConfig


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


def truncate_preds(preds: np.ndarray, series_length: int) -> np.ndarray:
    """truncate preds to series_length

    Args:
        preds (np.ndarray): [description]
        series_length (int): [description]

    Returns:
        np.ndarray: [description]
    """
    if len(preds) > series_length:
        preds = preds[:series_length]
    return preds


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


def merge_short_events(
    event_df: pl.DataFrame,
    series_length: dict[str, int],
    threshold: int = 1000,
) -> pl.DataFrame:
    if "wakeup" not in event_df["event"] or "onset" not in event_df["event"]:
        return event_df
    group = np.arange(len(event_df) // 2)
    group = np.repeat(group, 2)
    if len(event_df) % 2 != 0:
        group = np.append(group, group[-1] + 1)
    group = group.astype(np.int64)
    event_df = event_df.with_columns(pl.Series(name="group", values=group))
    wakeup_event_diff = event_df.filter(
        pl.col("event") == "wakeup"
    ).with_columns(step_diff=(pl.col("step").diff()))
    row_ids = pl.Series(
        name="row_id", values=np.arange(len(wakeup_event_diff))
    )
    wakeup_event_diff = wakeup_event_diff.with_columns(row_ids)
    # if the difference in steps is less than threshold make the next wakeup event the wakeup event for the previous onset event
    new_groups = []
    for i, row in enumerate(wakeup_event_diff.rows(named=True)):
        if i == 0:
            new_groups.append(row["group"])
            continue
        if row["step_diff"] < threshold:
            previous_group = new_groups[-1]
            new_groups.append(previous_group)
        else:
            new_groups.append(row["group"])

    new_groups = pl.Series(name="group", values=new_groups)
    wakeup_event_diff = wakeup_event_diff.with_columns(new_groups)
    real_wakeups = wakeup_event_diff.group_by(
        "group", maintain_order=True
    ).agg(
        pl.col("step").max().alias("step"),
        pl.col("score").last().alias("score"),
        pl.col("series_id").last().alias("series_id"),
    )
    real_wakeups = real_wakeups.with_columns(pl.lit("wakeup").alias("event"))
    real_onsets = event_df.filter(
        (pl.col("event") == "onset")
        & (pl.col("group").is_in(real_wakeups["group"]))
    )
    merged_df = real_onsets.vstack(
        real_wakeups.select(["series_id", "step", "event", "score", "group"])
    )

    merged_df = merged_df.sort(by=["series_id", "step"])
    group = np.arange(len(merged_df) // 2)
    group = np.repeat(group, 2)
    if len(merged_df) % 2 != 0:
        group = np.append(group, group[-1] + 1)
    merged_df = merged_df.with_columns(
        pl.Series(name="group", values=group, dtype=pl.Int64)
    )
    # find wakeups that are very close to the next onset and merge them
    merged_df = merged_df.with_columns(
        pl.col("step").diff().alias("step_diff")
    )
    onset_diffs = merged_df.filter(pl.col("event") == "onset").filter(
        pl.col("step_diff") < threshold
    )
    onset_groups_to_remove = onset_diffs["group"]
    wakeup_groups_to_remove = onset_diffs["group"] - 1

    onset_events = merged_df.filter(
        (pl.col("event") == "onset")
        & ~pl.col("group").is_in(onset_groups_to_remove)
    )
    wakeup_events = merged_df.filter(
        (pl.col("event") == "wakeup")
        & ~pl.col("group").is_in(wakeup_groups_to_remove)
    )
    wakeup_events = wakeup_events.with_columns(
        [pl.col(n).cast(t) for n, t in onset_events.schema.items()]
    )
    merged_df = onset_events.vstack(wakeup_events)

    merged_df = merged_df.select(["series_id", "step", "event", "score"]).sort(
        by=["series_id", "step"]
    )
    # calculate the difference between the current wakeup and the next onset
    row_ids = pl.Series(name="row_id", values=np.arange(len(merged_df)))
    merged_df = merged_df.with_columns(row_ids)

    # convert group to int64
    # merged_df = merged_df.with_columns(pl.col("group").cast(pl.Int32))
    return merged_df


def drop_short_events(event_df: pl.DataFrame, threshold=1000) -> pl.DataFrame:
    if "wakeup" not in event_df["event"] or "onset" not in event_df["event"]:
        return event_df

    group = np.arange(len(event_df) // 2)
    group = np.repeat(group, 2)
    if len(event_df) % 2 != 0:
        group = np.append(group, group[-1] + 1)
    group = group.astype(np.int32)
    event_df = event_df.with_columns(pl.Series(name="group", values=group))
    grouped_df = (
        event_df.group_by("group", maintain_order=True)
        .agg(
            (pl.col("step").last() - pl.col("step").first()).alias(
                "step_diff"
            ),
        )
        .filter(pl.col("step_diff") >= threshold)
    )
    event_df = event_df.filter(pl.col("group").is_in(grouped_df["group"]))
    return event_df


def post_process_for_prec(
    cfg: InferenceConfig,
    preds: dict[str, np.ndarray],
    series_length: dict[str, int],
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
        this_series_preds_round = np.where(
            this_series_preds > cfg.prediction_threshold, 1, 0
        )
        this_series_preds_diff = np.diff(this_series_preds_round)
        # onset is when diff is 1 (index + 1)
        # wakeup is when diff is -1 (index)
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
    # sub_df = sub_df.to_pandas()
    sub_df = sub_df.group_by("series_id", maintain_order=True).apply(
        lambda x: merge_short_events(x, series_length, cfg.duration_threshold)
    )
    sub_df = sub_df.group_by("series_id", maintain_order=True).apply(
        lambda x: drop_short_events(x, cfg.event_threshold)
    )
    sub_df = sub_df.sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = (
        sub_df.with_columns(row_ids)
        .select(["row_id", "series_id", "step", "event", "score"])
        .sort(by=["series_id", "step"])
    )
    return sub_df
