from angle_calculator import calculate_angle, smooth_value
from pose_detector import PoseDetector

# Beginner mode thresholds thode loose hai (galti pakadne me lenient),
# Advanced mode thresholds strict hai (professional form expect karta hai).
THRESHOLDS = {
    "Beginner": {"depth_angle": 118, "up_angle": 150, "back_angle_min": 30, "toe_margin": 48},
    "Advanced": {"depth_angle": 100, "up_angle": 160, "back_angle_min": 48, "toe_margin": 20},
}


class SquatAnalyzer:
    """
    SQUAT ke liye 3 cheezein check karta hai (real-time):
      1) Depth         -> knee angle threshold se neeche jaana chahiye (standing->squat->standing)
      2) Back posture  -> torso zyada aage nahi jhukna chahiye
      3) Knee-over-toe -> ghutna toe line se zyada aage nahi jaana chahiye

    Rep sirf tab count hoti hai jab UP -> DOWN -> UP poora cycle complete ho.
    analyze() ek dict return karta hai jisme feedback, reps, form_score,
    angle (visualization ke liye) aur last_rep_event (sound alert trigger
    karne ke liye) sab kuch hota hai.
    """

    def __init__(self, mode="Advanced"):
        self.counter = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "up"
        self.feedback = "Camera ke saamne khade ho jaayein"
        self.rep_flags = set()
        self._knee_hist = []
        self._back_hist = []
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

        hip, hip_v = get(landmarks, L.LEFT_HIP.value, frame_shape)
        knee, knee_v = get(landmarks, L.LEFT_KNEE.value, frame_shape)
        ankle, ankle_v = get(landmarks, L.LEFT_ANKLE.value, frame_shape)
        shoulder, sh_v = get(landmarks, L.LEFT_SHOULDER.value, frame_shape)
        foot_index, foot_v = get(landmarks, L.LEFT_FOOT_INDEX.value, frame_shape)

        if min(hip_v, knee_v, ankle_v, sh_v, foot_v) < 0.5:
            return self._empty_state("Poora sharir (side view) camera me laayein")

        raw_knee_angle = calculate_angle(hip, knee, ankle)
        raw_back_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = smooth_value(self._knee_hist, raw_knee_angle)
        back_angle = smooth_value(self._back_hist, raw_back_angle)

        # Knee-over-toe check - facing direction-aware (left ya right, dono taraf kaam karta hai)
        foot_dir = foot_index[0] - ankle[0]
        facing_sign = 1 if foot_dir >= 0 else -1
        knee_forward = (knee[0] - ankle[0]) * facing_sign
        toe_forward = abs(foot_dir)
        knee_over_toe = knee_forward > (toe_forward + t["toe_margin"])

        last_rep_event = None

        if knee_angle > t["up_angle"]:
            current_pos = "up"
        elif knee_angle < t["depth_angle"]:
            current_pos = "down"
        else:
            current_pos = "mid"

        if current_pos == "down":
            if back_angle < t["back_angle_min"]:
                self.feedback = "GALTI: Peeth aage jhuk rahi hai - chest upar rakho!"
                self.rep_flags.add("back")
            elif knee_over_toe:
                self.feedback = "GALTI: Ghutna toes se aage ja raha hai!"
                self.rep_flags.add("knee")
            else:
                self.feedback = "Sahi depth! Ab dheere se upar aao"
            self.stage = "down"

        elif current_pos == "mid" and self.stage == "down":
            self.feedback = "Upar aa rahe ho, control me rakho"

        elif current_pos == "up":
            if self.stage == "down":
                self.counter += 1
                if self.rep_flags:
                    self.bad_reps += 1
                    self.feedback = f"Rep {self.counter} poori - lekin form thik karo"
                    last_rep_event = "bad"
                else:
                    self.good_reps += 1
                    self.feedback = f"Rep {self.counter} - Perfect squat!"
                    last_rep_event = "good"
                self.rep_flags = set()
            elif self.counter == 0:
                self.feedback = "Squat shuru karne ke liye neeche baitho"
            self.stage = "up"

        return {
            "feedback": self.feedback,
            "reps": self.counter,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "form_score": self.form_score,
            "angle": knee_angle,
            "joint_point": knee,
            "last_rep_event": last_rep_event,
        }
