import numpy as np
import polars as pl
from scipy.signal import find_peaks


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.01,
    distance: int = 5000,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            # we need to write our own version of this
            steps = find_peaks(
                this_event_preds, height=score_th, distance=distance
            )[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_ids[0],
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(
        ["row_id", "series_id", "step", "event", "score"]
    )
    return sub_df


def post_process_for_prec(
    keys: list[str],
    preds: list[np.ndarray],
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray):
        score_th (float, optional): threshold for score. Defaults to 0.5.

    Returns:
        pl.DataFrame: submission dataframe
    """
    series_ids = keys
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx]
        this_series_preds_round = np.round(this_series_preds)
        this_series_preds_diff = np.diff(this_series_preds_round)
        # onset is when diff is 1 (index + 1)
        # wakeup is when diff is -1 (index + 1)
        onset_steps = np.where(this_series_preds_diff == 1)[0] + 1
        wakeup_stes = np.where(this_series_preds_diff == -1)[0] + 1

        for i, event_name in enumerate(["onset", "wakeup"]):
            if event_name == "onset":
                steps = onset_steps
            else:
                steps = wakeup_stes
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
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(
        ["row_id", "series_id", "step", "event", "score"]
    )
    return sub_df
