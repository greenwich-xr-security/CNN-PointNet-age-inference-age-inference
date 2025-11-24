import numpy as np

class HandMetrics:
    """
    Static class for computing scale-invariant ratios from MediaPipe hand landmarks.

    Landmarks are expected as a numpy array of shape (21, 3), where each row is (x, y, z)
    in normalized coordinates (crop-relative or image-normalized).
    """

    @staticmethod
    def _dist(p, q):
        """Euclidean distance between two 3D points p and q."""
        return np.linalg.norm(p - q)

    @staticmethod
    def proximal_to_intermediate_ratio(landmarks, digit=2):
        """
        Ratio of the intermediate phalanx to the proximal phalanx for a given finger.
        digit: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
        Uses landmarks:
          Proximal: MCP -> PIP
          Intermediate: PIP -> DIP
        """
        # MediaPipe indices: base of thumb is 1, index MCP is 5, etc.
        start = {1: (1, 2), 2: (5, 6), 3: (9, 10), 4: (13, 14), 5: (17, 18)}[digit]
        mid  = {1: (2, 3), 2: (6, 7), 3: (10, 11), 4: (14, 15), 5: (18, 19)}[digit]
        L_prox = HandMetrics._dist(landmarks[start[0]], landmarks[start[1]])
        L_int  = HandMetrics._dist(landmarks[mid[0]],   landmarks[mid[1]])
        return L_int / L_prox

    @staticmethod
    def intermediate_to_distal_ratio(landmarks, digit=2):
        """
        Ratio of the distal phalanx to the intermediate phalanx for a given finger.
        digit: same indexing as proximal_to_intermediate_ratio
        Uses landmarks: intermediate: PIP -> DIP, distal: DIP -> TIP
        """
        mid  = {1: (2, 3), 2: (6, 7), 3: (10, 11), 4: (14, 15), 5: (18, 19)}[digit]
        end  = {1: (3, 4), 2: (7, 8), 3: (11, 12), 4: (15, 16), 5: (19, 20)}[digit]
        L_int = HandMetrics._dist(landmarks[mid[0]], landmarks[mid[1]])
        L_dist= HandMetrics._dist(landmarks[end[0]], landmarks[end[1]])
        return L_dist / L_int

    @staticmethod
    def two_four_digit_ratio(landmarks):
        """
        2D:4D ratio - overall length index finger (TIP2-to-wrist) divided by ring finger (TIP4-to-wrist).
        Often used in biological studies.
        """
        wrist = landmarks[0]
        tip2  = landmarks[8]
        tip4  = landmarks[16]
        return HandMetrics._dist(tip2, wrist) / HandMetrics._dist(tip4, wrist)

    @staticmethod
    def thumb_pinky_ratio(landmarks):
        """
        Ratio of overall thumb length (TIP1-to-wrist) to pinky length (TIP5-to-wrist).
        """
        wrist = landmarks[0]
        tip1  = landmarks[4]
        tip5  = landmarks[20]
        return HandMetrics._dist(tip1, wrist) / HandMetrics._dist(tip5, wrist)

    @staticmethod
    def middle_to_palm_ratio(landmarks):
        """
        Ratio of middle finger length (TIP3-to-wrist) to palm length (Wrist-to-middle MCP).
        Useful in ergonomics to relate finger length to palm size.
        """
        wrist = landmarks[0]
        tip3  = landmarks[12]
        mcp3  = landmarks[9]
        return HandMetrics._dist(tip3, wrist) / HandMetrics._dist(mcp3, wrist)

    @staticmethod
    def palm_width_to_length(landmarks):
        """
        Ratio of palm width (index MCP-to-pinky MCP) to palm length (wrist-to-middle MCP).
        Core anthropometric measure.
        """
        wrist = landmarks[0]
        mcp2  = landmarks[5]
        mcp5  = landmarks[17]
        mcp3  = landmarks[9]
        width = HandMetrics._dist(mcp2, mcp5)
        length= HandMetrics._dist(wrist, mcp3)
        return width / length

    @staticmethod
    def fingertip_spread(landmarks):
        """
        Max fingertip span: distance between the two farthest fingertip points, normalized by palm width.
        Useful for measuring hand opening.
        """
        tips = [landmarks[i] for i in (4,8,12,16,20)]
        # compute all pairwise distances
        dists = [HandMetrics._dist(a, b) for i,a in enumerate(tips) for b in tips[i+1:]]
        max_span = max(dists)
        # normalize by palm width
        palm_width = HandMetrics.palm_width_to_length(landmarks) * HandMetrics._dist(landmarks[0], landmarks[9])
        return max_span / palm_width

    @staticmethod
    def index_middle_spread_ratio(landmarks):
        """
        Ratio of index-to-middle fingertip distance over middle-to-ring distance.
        Captures relative finger spread.
        """
        tip2 = landmarks[8]
        tip3 = landmarks[12]
        tip4 = landmarks[16]
        d1 = HandMetrics._dist(tip2, tip3)
        d2 = HandMetrics._dist(tip3, tip4)
        return d1 / d2

    @staticmethod
    def pip_joint_angle_ratio(landmarks):
        """
        Ratio of PIP joint flexion angles between index and middle fingers.
        Angle(MCP-PIP-DIP) index / Angle(MCP-PIP-DIP) middle.
        Angles reveal relative joint mobility.
        """
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
            return np.arccos(np.clip(cosang, -1.0, 1.0))

        # index: MCP(5)-PIP(6)-DIP(7); middle: MCP(9)-PIP(10)-DIP(11)
        ang_idx = angle(landmarks[5], landmarks[6], landmarks[7])
        ang_mid = angle(landmarks[9], landmarks[10], landmarks[11])
        return ang_idx / ang_mid

    @staticmethod
    def all_ratios(landmarks):
        """
        Compute all available ratios and return as a dict.
        """
        return {
            'prox_int_index': HandMetrics.proximal_to_intermediate_ratio(landmarks, digit=2),
            'int_dist_index': HandMetrics.intermediate_to_distal_ratio(landmarks, digit=2),
            '2D4D': HandMetrics.two_four_digit_ratio(landmarks),
            'thumb_pinky': HandMetrics.thumb_pinky_ratio(landmarks),
            'middle_palm': HandMetrics.middle_to_palm_ratio(landmarks),
            'palm_w_l': HandMetrics.palm_width_to_length(landmarks),
            #'max_spread': HandMetrics.fingertip_spread(landmarks),
            #'idx_mid_spread': HandMetrics.index_middle_spread_ratio(landmarks),
            #'pip_angle': HandMetrics.pip_joint_angle_ratio(landmarks)
        }
