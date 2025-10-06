#!/usr/bin/env python3
"""Hand data recorder
=====================
Records raw landmarks, computed hand ratios, and orientation metadata.

This version intentionally removes any dependency on user noise augmentation.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2

from displayUtils import DisplayUtils
from handRatio import HandRatios
from handLandmarksDetection import SentisHandLandmarkDetector


class HandDataRecorder:
    """Records raw landmarks, hand ratios and orientation."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        """Create an empty recorder."""
        self.records_df: pd.DataFrame = pd.DataFrame()
        self.landmarks_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def add_record(
        self,
        user_name: str,
        image_name: str,
        age: int,
        landmarks: np.ndarray,
    ) -> None:
        """Compute ratios, store raw data, and orientation metadata."""
        # ---- ratios + orientation ------------------------------------ #
        ratio_record = {
            "user": user_name,
            "image": image_name,
            "age": age,
            **HandRatios.all_ratios(landmarks),
        }

        ratio_record["is_augmented"] = False  # original row

        # ---------- append ratio row ---------------------------------- #
        self.records_df = pd.concat(
            [self.records_df, pd.DataFrame([ratio_record])], ignore_index=True
        )

        # ---------- raw landmarks ------------------------------------- #
        rows = [
            {
                "user": user_name,
                "image": image_name,
                "age": age,
                "landmark_index": f"lm{idx + 1}",  # lm1..lm21
                "x": float(x_c),
                "y": float(y_c),
                "z": float(z),
            }
            for idx, (x_c, y_c, z) in enumerate(landmarks)
        ]
        self.landmarks_df = pd.concat(
            [self.landmarks_df, pd.DataFrame(rows)], ignore_index=True
        )

    # ------------------------------------------------------------------ #
    # Reading helpers
    # ------------------------------------------------------------------ #
    def _subset(self, user_name: str, image_name: str) -> pd.DataFrame:
        subset = self.landmarks_df[
            (self.landmarks_df["user"] == user_name)
            & (self.landmarks_df["image"] == image_name)
        ]
        if subset.empty:
            raise KeyError(
                f"No data for user '{user_name}' and image '{image_name}'"
            )
        return subset

    def _resolve_key(self, index: int) -> Tuple[str, str]:
        if not 0 <= index < len(self.records_df):
            raise IndexError(f"No record at index {index}")
        row = self.records_df.iloc[index]
        return str(row["user"]), str(row["image"])

    # ------------------------------------------------------------------ #
    # Public getters
    # ------------------------------------------------------------------ #
    def get_orientation(self, user_or_index: str | int, image_name: str | None = None) -> tuple[str, str]:
        if image_name is None:
            user_name, image_name = self._resolve_key(int(user_or_index))
        else:
            user_name = str(user_or_index)

        row = self.records_df.loc[
            (self.records_df["user"] == user_name)
            & (self.records_df["image"] == image_name)
        ]
        if row.empty:
            raise KeyError(
                f"No ratio record for user '{user_name}' and image '{image_name}'"
            )
        r0 = row.iloc[0]
        return str(r0["handedness"]), str(r0["palm_side"])

    def get_landmarks(self, user_or_index: str | int, image_name: str | None = None) -> np.ndarray:
        if image_name is None:
            user_name, image_name = self._resolve_key(int(user_or_index))
        else:
            user_name = str(user_or_index)

        subset = self._subset(user_name, image_name)
        subset_sorted = subset.sort_values(
            "landmark_index", key=lambda s: s.str.replace("lm", "").astype(int)
        )
        return subset_sorted[["x", "y", "z"]].to_numpy()

    # Note: bounding-box utilities were not part of this stripped version

    # ------------------------------------------------------------------ #
    # DataFrame utilities
    # ------------------------------------------------------------------ #
    def get_ratio_records(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the ratio-level table."""
        return self.records_df.copy() if copy else self.records_df

    def get_landmark_records(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the landmark-level table (21 rows per ratio record)."""
        return self.landmarks_df.copy() if copy else self.landmarks_df

    def get_landmark_records_long(self) -> pd.DataFrame:
        return self.landmarks_df

    def get_landmark_records_wide(self) -> pd.DataFrame:
        if self.landmarks_df.empty:
            return pd.DataFrame()
        df = self.landmarks_df.copy()
        df_wide = df.pivot_table(
            index=["user", "image", "age"],
            columns="landmark_index",
            values=["x", "y", "z"],
        )
        df_wide = df_wide.swaplevel(0, 1, axis=1)
        df_wide = df_wide.reindex(
            sorted(df_wide.columns, key=lambda x: int(x[0].lstrip("lm"))), axis=1
        )
        return df_wide.reset_index()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_to_h5(self, filepath: str) -> None:
        with pd.HDFStore(filepath, "w") as store:
            store.put("ratios", self.get_ratio_records(), format="table")
            store.put("landmarks", self.get_landmark_records_wide(), format="table")
        print(
            f"Saved {len(self.records_df)} ratio records (incl. augmented="
            f"{int(self.records_df.get('is_augmented', pd.Series(dtype=bool)).sum())}) "
            f"and {len(self.get_landmark_records_wide())} wide landmark records to {filepath}"
        )


# ---------------------------------------------------------------------- #
# Example usage for quick manual testing
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    image_path = "./handsPictures/Riccardo/1.jpg"
    user = "Alice"
    age = 28

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load '{image_path}'")

    detector = SentisHandLandmarkDetector()
    lms_norm, lms = detector.detect(img)

    recorder = HandDataRecorder()

    recorder.add_record(user, image_path, age, lms)

    print("Ratio records ->", len(recorder.get_ratio_records()))
    recorder.save_to_h5("hand_test_data.h5")

    DisplayUtils.display_with_coords(
        img,
        lms_norm,
        handedness=recorder.get_orientation(user, image_path)[0],
        palm_side=recorder.get_orientation(user, image_path)[1],
    )
