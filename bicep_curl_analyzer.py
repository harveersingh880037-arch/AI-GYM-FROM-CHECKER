from angle_calculator import calculate_angle, smooth_value
from pose_detector import PoseDetector

THRESHOLDS = {
    "Beginner": {"up_angle": 55, "down_angle": 150, "drift_px": 95, "shoulder_angle_max": 42},
    "Advanced": {"up_angle": 40, "down_angle": 160, "drift_px": 55, "shoulder_angle_max": 30},
}


class CurlAnalyzer:
    """
    BICEP CURL ke liye 2 mukhya cheezein check karta hai (real-time):
      1) Full range of motion -> neeche poora seedha, upar poora curl
      2) Elbow stability      -> upper arm/elbow body se chipka rehna chahiye
                                  (swinging/cheating detect karta hai)

    Rep sirf tab count hoti hai jab DOWN -> UP -> DOWN poora cycle complete ho.
    """

    def __init__(self, mode="Advanced"):
        self.counter = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "down"
        self.feedback = "Camera ke saamne khade ho jaayein"
        self.rep_flags = set()
        self._elbow_hist = []
        self._shoulder_hist = []
        self._start_elbow_x = None
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode if mode in THRESHOLDS else "Advanced"
        self.thresholds = THRESHOLDS[self.mode]

    def reset(self):
        self.__init__(mode=self.mode)

    @property
    def form_score(self):
        if self.counter == 0:
            return 100
        return round((self.good_reps / self.counter) * 100)

    def _empty_state(self, feedback):
        return {
            "feedback": feedback,
            "reps": self.counter,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "form_score": self.form_score,
            "angle": None,
            "joint_point": None,
            "last_rep_event": None,
        }

    def analyze(self, landmarks, frame_shape, mp_pose):
        get = PoseDetector.get_landmark_xy
        L = mp_pose.PoseLandmark
        t = self.thresholds

        shoulder, sh_v = get(landmarks, L.LEFT_SHOULDER.value, frame_shape)
        elbow, el_v = get(landmarks, L.LEFT_ELBOW.value, frame_shape)
        wrist, wr_v = get(landmarks, L.LEFT_WRIST.value, frame_shape)
        hip, hip_v = get(landmarks, L.LEFT_HIP.value, frame_shape)

        if min(sh_v, el_v, wr_v, hip_v) < 0.5:
            return self._empty_state("Poora upper body (side view) camera me laayein")

        raw_elbow_angle = calculate_angle(shoulder, elbow, wrist)
        raw_shoulder_angle = calculate_angle(hip, shoulder, elbow)
        elbow_angle = smooth_value(self._elbow_hist, raw_elbow_angle)
        shoulder_angle = smooth_value(self._shoulder_hist, raw_shoulder_angle)

        if self._start_elbow_x is None:
            self._start_elbow_x = elbow[0]

        elbow_drift = abs(elbow[0] - self._start_elbow_x)
        last_rep_event = None

        if elbow_angle > t["down_angle"]:
            current_pos = "down"
        elif elbow_angle < t["up_angle"]:
            current_pos = "up"
        else:
            current_pos = "mid"

        if current_pos == "up":
            if elbow_drift > t["drift_px"] or shoulder_angle > t["shoulder_angle_max"]:
                self.feedback = "GALTI: Elbow ko body ke paas sthir rakho, swing mat karo!"
                self.rep_flags.add("swing")
            else:
                self.feedback = "Achha curl! Ab dheere se neeche lao"
            self.stage = "up"

        elif current_pos == "mid":
            self.feedback = "Curl karte raho..." if self.stage == "down" else "Neeche la rahe ho, control me rakho"

        elif current_pos == "down":
            if self.stage == "up":
                self.counter += 1
                if self.rep_flags:
                    self.bad_reps += 1
                    self.feedback = f"Rep {self.counter} poori - elbow stable rakho"
                    last_rep_event = "bad"
                else:
                    self.good_reps += 1
                    self.feedback = f"Rep {self.counter} - Perfect curl!"
                    last_rep_event = "good"
                self.rep_flags = set()
                self._start_elbow_x = elbow[0]
            elif self.counter == 0:
                self.feedback = "Baju seedhi rakho, ab curl shuru karo"
            self.stage = "down"

        return {
            "feedback": self.feedback,
            "reps": self.counter,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "form_score": self.form_score,
            "angle": elbow_angle,
            "joint_point": elbow,
            "last_rep_event": last_rep_event,
        }
