import numpy as np
import polars as pl

from src.conf import InferenceConfig, TrainConfig


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
    cfg: InferenceConfig | TrainConfig,
    preds: dict[str, np.ndarray],
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

        for i, event_name in enumerate(["onset", "wakeup"]):
            steps = np.where(this_series_preds[:, i + 1] > cfg.pp.score_th)[0]
            scores = this_series_preds[steps, i + 1]

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
        lambda x: drop_short_events(x, cfg.pp.distance)
    )
    sub_df = sub_df.sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = (
        sub_df.with_columns(row_ids)
        .select(["row_id", "series_id", "step", "event", "score"])
        .sort(by=["series_id", "step"])
    )
    return sub_df
