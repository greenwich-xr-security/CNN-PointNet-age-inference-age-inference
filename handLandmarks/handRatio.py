import cv2
import numpy as np

from handLandmarksDetection import SentisHandLandmarkDetector

class HandRatios:
    """Pose-invariant hand-metric utilities for 21-point MediaPipe-style landmarks.

    The class exposes helper methods that operate purely on bone-segment lengths,
    so derived measurements are robust to finger spread, flexion, and camera
    viewpoint.

    **Features**
    ---------------------------------
    * ``finger_lengths`` – pose-invariant anatomical length for **all five fingers**.
    * ``finger_length_ratios`` – every pair-wise finger-length ratio, generalising
      the classic *2D:4D* (index:ring) metric to *any* finger pair.
    * ``pairwise_length_ratios`` – ratios of **every bone segment** against every
      other (20 × 20).
    * ``all_ratios`` – bundled anatomical ratios from across methods.
    """

    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    FINGER_MAP = {
        "thumb":  [ 0, 1, 2, 3, 4],
        "index":  [ 0, 5, 6, 7, 8],
        "middle": [ 0, 9, 10, 11, 12],
        "ring":   [ 0, 13, 14, 15, 16],
        "pinky":  [ 0, 17, 18, 19, 20],
    }

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @classmethod
    def bone_lengths(cls, landmarks):
        return [cls._dist(landmarks[i], landmarks[j]) for i, j in cls.CONNECTIONS]

    @classmethod
    def total_finger_length(cls, landmarks, finger):
        joints = cls.FINGER_MAP[finger]
        return sum(cls._dist(landmarks[a], landmarks[b]) for a, b in zip(joints[:-1], joints[1:]))

    @classmethod
    def finger_lengths(cls, landmarks):
        return {f: cls.total_finger_length(landmarks, f) for f in cls.FINGER_MAP}

    @classmethod
    def finger_length_ratios(cls, landmarks, return_matrix=False):
        order = list(cls.FINGER_MAP.keys())
        lengths = [cls.total_finger_length(landmarks, f) for f in order]

        if return_matrix:
            mat = np.empty((5, 5), dtype=float)
            for i in range(5):
                for j in range(5):
                    mat[i, j] = lengths[i] / lengths[j] if lengths[j] else np.nan
            return mat

        return {
            (order[i], order[j]): lengths[i] / lengths[j] if lengths[j] else np.nan
            for i in range(5)
            for j in range(i + 1, 5)
        }

    @classmethod
    def pairwise_length_ratios(cls, landmarks, return_matrix=False):
        lengths = cls.bone_lengths(landmarks)
        n = len(lengths)

        if return_matrix:
            mat = np.empty((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    mat[i, j] = lengths[i] / lengths[j] if lengths[j] else np.nan
            return mat

        return {
            (i, j): lengths[i] / lengths[j] if lengths[j] else np.nan
            for i in range(n)
            for j in range(i + 1, n)
        }

    @staticmethod
    def all_ratios(landmarks):
        ratios = {}

        # Finger-to-finger length ratios
        named_fingers = list(HandRatios.FINGER_MAP.keys())
        finger_lens = {f: HandRatios.total_finger_length(landmarks, f) for f in named_fingers}
        for i in range(len(named_fingers)):
            for j in range(i + 1, len(named_fingers)):
                f1, f2 = named_fingers[i], named_fingers[j]
                key = f"{f1}_{f2}_ratio"
                ratios[key] = finger_lens[f1] / finger_lens[f2] if finger_lens[f2] else np.nan

        # Bone segment ratios
        seg_ratios = HandRatios.pairwise_length_ratios(landmarks)
        for (i, j), val in seg_ratios.items():
            ratios[f"L{i}_L{j}_ratio"] = val

        return ratios


if __name__ == "__main__":

    # Example usage
    image_path = "./handsPictures/Riccardo/1.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load '{image_path}'.")

    # detect
    detector = SentisHandLandmarkDetector()
    landmarks, box = detector.detect(img)
    
    hm = HandRatios

    print("\nAll bundled ratios:")
    ratios = hm.all_ratios(landmarks)
    for k, v in list(ratios.items()):  # Print just first 10 for brevity
        print(f"  {k}: {v:.2f}")
    print(f"{len(ratios)} total ratios")
