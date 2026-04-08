from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

from config import (
    AVERAGE_FACTOR,
    CHANNEL_IDS,
    CHUNK_SECONDS,
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_RATE,
    NOISE_REDUCE_PROP_DECREASE,
    NOISE_REDUCE_STATIONARY,
    OUTPUT_DIR,
    PLOT_DPI,
    PROJECT_ROOT,
    ROUND_DECIMALS,
    SAVE_PLOTS,
    SPECTRAL_GATE_THRESHOLD,
    STEP_SECONDS,
    TEMPLATE_PATH,
    WINDOW_SECONDS,
)
from input_files import WAV_FILES


LOGGER = logging.getLogger("pipeline_minimal")
PROGRESS_UPDATES_PER_FILE = 10


@dataclass
class ChannelState:
    matched_filter_tail: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    averaging_remainder: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    reduced_values: List[float] = field(default_factory=list)
    reduced_sample_indices: List[int] = field(default_factory=list)


def parse_timestamp_from_path(path: Path) -> datetime:
    stem = path.stem
    if len(stem) != 14 or not stem.isdigit():
        raise ValueError(f"Filename must be YYYYMMDDHHMMSS, got: {path.name}")
    return datetime.strptime(stem, "%Y%m%d%H%M%S")


def sorted_wav_files(paths: Sequence[Path]) -> List[Tuple[Path, datetime]]:
    if not paths:
        raise ValueError("WAV_FILES is empty. Edit src/input_files.py and add WAV paths.")

    parsed: List[Tuple[Path, datetime]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input WAV not found: {path}")
        parsed.append((path, parse_timestamp_from_path(path)))
    parsed.sort(key=lambda item: item[1])
    return parsed


def load_template_waveform(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Template Parquet not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Template Parquet is empty: {path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"Template Parquet has no numeric columns: {path}")

    template = df[numeric_cols[0]].to_numpy(dtype=np.float32)
    template = template[np.isfinite(template)]
    if template.size == 0:
        raise ValueError(f"Template waveform has no finite samples: {path}")

    # Normalize for numerical stability.
    peak = float(np.max(np.abs(template)))
    if peak > 0:
        template = template / peak
    return template


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def average_with_remainder(
    current_values: np.ndarray,
    current_start_sample: int,
    state: ChannelState,
    factor: int,
) -> None:
    remainder_len = int(state.averaging_remainder.size)
    if remainder_len > 0:
        joined = np.concatenate([state.averaging_remainder, current_values]).astype(np.float32, copy=False)
        joined_start = current_start_sample - remainder_len
    else:
        joined = current_values
        joined_start = current_start_sample

    complete_blocks = int(joined.size // factor)
    if complete_blocks == 0:
        state.averaging_remainder = joined
        return

    cutoff = complete_blocks * factor
    block_part = joined[:cutoff].reshape(complete_blocks, factor)
    reduced = block_part.mean(axis=1, dtype=np.float64)
    reduced = np.round(reduced, ROUND_DECIMALS)
    state.reduced_values.extend(reduced.tolist())

    sample_indices = joined_start + np.arange(complete_blocks, dtype=np.int64) * factor
    state.reduced_sample_indices.extend(sample_indices.tolist())
    state.averaging_remainder = joined[cutoff:]


def matched_filter_chunk(
    gated_chunk: np.ndarray,
    template_reversed: np.ndarray,
    state: ChannelState,
) -> np.ndarray:
    tail = state.matched_filter_tail
    if tail.size > 0:
        stream = np.concatenate([tail, gated_chunk]).astype(np.float32, copy=False)
    else:
        stream = gated_chunk

    filtered = fftconvolve(stream, template_reversed, mode="valid")
    filtered = filtered.astype(np.float32, copy=False)

    tail_len = max(int(template_reversed.size) - 1, 0)
    if tail_len > 0:
        state.matched_filter_tail = stream[-tail_len:].copy()
    else:
        state.matched_filter_tail = np.array([], dtype=np.float32)
    return filtered


def run_noise_and_gate(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    denoised = nr.reduce_noise(
        y=signal,
        sr=sample_rate,
        stationary=NOISE_REDUCE_STATIONARY,
        prop_decrease=NOISE_REDUCE_PROP_DECREASE,
    )
    gated = nr.reduce_noise(
        y=denoised,
        sr=sample_rate,
        stationary=NOISE_REDUCE_STATIONARY,
        prop_decrease=SPECTRAL_GATE_THRESHOLD,
    )
    return np.asarray(gated, dtype=np.float32)


def apply_sliding_window(
    timestamps: Sequence[datetime],
    values: np.ndarray,
    reduced_rate_hz: float,
) -> pd.DataFrame:
    window_samples = int(WINDOW_SECONDS * reduced_rate_hz)
    step_samples = int(STEP_SECONDS * reduced_rate_hz)
    if values.size < window_samples:
        return pd.DataFrame(columns=["timestamp", "intensity"])

    abs_values = np.abs(values).astype(np.float64, copy=False)
    cumsum = np.concatenate([[0.0], np.cumsum(abs_values, dtype=np.float64)])

    starts = np.arange(0, values.size - window_samples + 1, step_samples, dtype=np.int64)
    intensities = (cumsum[starts + window_samples] - cumsum[starts]) / float(window_samples)
    intensities = np.round(intensities, ROUND_DECIMALS)

    out_ts = [timestamps[int(start)] for start in starts.tolist()]
    return pd.DataFrame({"timestamp": out_ts, "intensity": intensities})


def save_outputs(
    channel_id: int,
    final_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"channel_{channel_id}_sliding_window.csv"
    parquet_path = output_dir / f"channel_{channel_id}_sliding_window.parquet"
    final_df.to_csv(csv_path, index=False)
    final_df.to_parquet(parquet_path, index=False)

    if SAVE_PLOTS and not final_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(final_df["timestamp"], final_df["intensity"], linewidth=1.0)
        ax.set_title(f"Channel {channel_id} Sliding Window Intensity")
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"channel_{channel_id}_sliding_window.png", dpi=PLOT_DPI)
        plt.close(fig)


def main() -> None:
    configure_logging()

    files_with_ts = sorted_wav_files([Path(p) for p in WAV_FILES])
    template = load_template_waveform(TEMPLATE_PATH)
    template_reversed = template[::-1]
    reduced_rate_hz = EXPECTED_SAMPLE_RATE / float(AVERAGE_FACTOR)
    LOGGER.info(
        "Template loaded: %d samples (%.2f sec)",
        template.size,
        template.size / float(EXPECTED_SAMPLE_RATE),
    )

    LOGGER.info("Found %d input files", len(files_with_ts))
    for path, ts in files_with_ts:
        LOGGER.info("Input file: %s | start: %s", path, ts.isoformat())

    states: Dict[int, ChannelState] = {ch: ChannelState() for ch in CHANNEL_IDS}
    first_start = files_with_ts[0][1]
    global_sample_index = 0

    chunk_frames = CHUNK_SECONDS * EXPECTED_SAMPLE_RATE
    if chunk_frames <= 0:
        raise ValueError("CHUNK_SECONDS must be > 0")

    for path, _ in files_with_ts:
        with sf.SoundFile(path, mode="r") as wav:
            if wav.samplerate != EXPECTED_SAMPLE_RATE:
                raise ValueError(
                    f"Invalid sample rate in {path}: {wav.samplerate}, expected {EXPECTED_SAMPLE_RATE}"
                )
            if wav.channels < EXPECTED_CHANNELS:
                raise ValueError(
                    f"Invalid channel count in {path}: {wav.channels}, expected at least {EXPECTED_CHANNELS}"
                )

            total_chunks = max(1, math.ceil(wav.frames / chunk_frames))
            update_every = max(1, total_chunks // PROGRESS_UPDATES_PER_FILE)
            chunk_idx = 0
            file_frames_processed = 0

            LOGGER.info(
                "Processing %s | frames=%d | chunks=%d | channels=%s",
                path.name,
                wav.frames,
                total_chunks,
                CHANNEL_IDS,
            )

            while True:
                block = wav.read(frames=chunk_frames, dtype="float32", always_2d=True)
                if block.size == 0:
                    break

                chunk_idx += 1
                current_start = global_sample_index
                current_len = block.shape[0]
                file_frames_processed += current_len

                for channel_id in CHANNEL_IDS:
                    signal = np.asarray(block[:, channel_id], dtype=np.float32)
                    processed = run_noise_and_gate(signal=signal, sample_rate=EXPECTED_SAMPLE_RATE)
                    matched = matched_filter_chunk(
                        gated_chunk=processed,
                        template_reversed=template_reversed,
                        state=states[channel_id],
                    )
                    average_with_remainder(
                        current_values=matched,
                        current_start_sample=current_start,
                        state=states[channel_id],
                        factor=AVERAGE_FACTOR,
                    )

                global_sample_index += current_len

                if chunk_idx == 1 or chunk_idx % update_every == 0 or chunk_idx == total_chunks:
                    pct = 100.0 * (file_frames_processed / float(wav.frames))
                    LOGGER.info(
                        "File %s progress: chunk %d/%d (%.1f%%)",
                        path.name,
                        chunk_idx,
                        total_chunks,
                        pct,
                    )

            LOGGER.info("Finished %s", path.name)

    LOGGER.info("Converting reduced streams to final sliding-window outputs")
    for channel_id in CHANNEL_IDS:
        channel_state = states[channel_id]

        if not channel_state.reduced_values:
            LOGGER.warning("No reduced values for channel %d. Skipping output.", channel_id)
            continue

        reduced_values = np.asarray(channel_state.reduced_values, dtype=np.float64)
        reduced_timestamps = [
            first_start + timedelta(seconds=sample_idx / EXPECTED_SAMPLE_RATE)
            for sample_idx in channel_state.reduced_sample_indices
        ]
        LOGGER.info(
            "Channel %d: reduced points=%d, building sliding window",
            channel_id,
            reduced_values.size,
        )

        final_df = apply_sliding_window(
            timestamps=reduced_timestamps,
            values=reduced_values,
            reduced_rate_hz=reduced_rate_hz,
        )
        final_df.insert(1, "channel_id", channel_id)
        save_outputs(channel_id=channel_id, final_df=final_df, output_dir=OUTPUT_DIR)
        LOGGER.info("Saved channel %d outputs (%d rows)", channel_id, len(final_df))

    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
