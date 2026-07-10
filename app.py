import os
import tempfile
import time

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from bicep_curl_analyzer import CurlAnalyzer
from frame_processor import process_frame
from pose_detector import PoseDetector
from sound_utils import get_audio_html
from squat_analyzer import SquatAnalyzer
from workout_summary import build_summary_text, estimate_calories
from yolo_detector import YOLOPersonDetector

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Gym Form Checker", page_icon="🏋️", layout="wide")

mp_pose = mp.solutions.pose
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

EXERCISE_INFO = {
    "Squat": {
        "steps": [
            "Pair kholo sidhe khade ho jao or pair chodai  me rakho, toes thodi bahar taraf.",
            "Chest upar rakho, peeth seedhi, core tight karo.",
            "Hips ko piche aur neeche le jao jaise kisi kursi par baith rahe ho.",
            "Ghutne toes ki seedh me rakho - andar mudne na do, aage bhi na nikalne do.",
            "Jab tak jaanghein (thighs) floor ke parallel ho jaayein, tab tak neeche jao.",
            "Ediyon (heels) se push karke wapas seedhe khade ho jao.",
        ],
        "camera_tip": "Camera ko apni SIDE (profile) se rakho, poora sharir (sar se paon tak) frame me aana chahiye.",
        "checks": ["Squat ki depth (knee angle)", "Peeth ka jhukav (back angle)", "Knee-over-toe (ghutna toe line se aage)"],
    },
    "Bicep Curl": {
        "steps": [
            "Seedhe khade ho jao, dumbbell/weight haath me pakdo (ya bina weight ke bhi practice kar sakte ho).",
            "Upper arm (kandhe se kohni tak) body se chipka ke rakho.",
            "Sirf kohni (elbow) mod kar weight ko upar lao, kandha mat hilao.",
            "Upar poora curl karo (biceps fully contract), phir dheere se neeche lao.",
            "Neeche baju poori seedhi (full extension) hone tak lao.",
            "Poora movement slow aur control me karo - jhatka/swing na maro.",
        ],
        "camera_tip": "Camera ko apni SIDE se rakho taaki kandha, kohni aur kalai (wrist) teeno saaf dikhein.",
        "checks": ["Full range of motion (upar/neeche poora)", "Elbow stability (swing/cheating detect)"],
    },
}


# ----------------------------------------------------------------------
# CACHED / PERSISTED HEAVY RESOURCES
# ----------------------------------------------------------------------
@st.cache_resource
def get_yolo_detector():
    return YOLOPersonDetector()


def get_upload_pose_detector():
    if "upload_pose_detector" not in st.session_state:
        st.session_state.upload_pose_detector = PoseDetector()
    return st.session_state.upload_pose_detector


# ----------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ----------------------------------------------------------------------
_defaults = {
    "mode": "Advanced",
    "use_yolo": False,
    "sound_enabled": True,
    "weight_kg": 70,
    "target_reps": 10,
    "input_source": "Webcam (Live)",
    "exercise": "Squat",
    "session_start_time": None,
    "last_seen_good": 0,
    "last_seen_bad": 0,
    "video_cap": None,
    "video_playing": False,
    "video_temp_path": None,
    "uploaded_file_id": None,
    "last_video_state": None,
    "upload_squat_analyzer": None,
    "upload_curl_analyzer": None,
    "_reset_webcam_flag": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.upload_squat_analyzer is None:
    st.session_state.upload_squat_analyzer = SquatAnalyzer(mode=st.session_state.mode)
if st.session_state.upload_curl_analyzer is None:
    st.session_state.upload_curl_analyzer = CurlAnalyzer(mode=st.session_state.mode)
if st.session_state.session_start_time is None:
    st.session_state.session_start_time = time.time()


# ----------------------------------------------------------------------
# WEBCAM VIDEO PROCESSOR
# ----------------------------------------------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector()
        self.squat = SquatAnalyzer(mode="Advanced")
        self.curl = CurlAnalyzer(mode="Advanced")
        self.exercise = "Squat"
        self.use_yolo = False
        self.yolo_detector = None
        self.last_state = {
            "reps": 0, "good_reps": 0, "bad_reps": 0,
            "form_score": 100, "feedback": "Camera shuru ho rahi hai...",
        }

    def set_config(self, exercise, mode, use_yolo, yolo_detector):
        self.exercise = exercise
        self.use_yolo = use_yolo
        self.yolo_detector = yolo_detector
        self.squat.set_mode(mode)
        self.curl.set_mode(mode)

    def reset(self):
        self.squat.reset()
        self.curl.reset()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror view - natural feel
        annotated, state = process_frame(
            img, self.exercise, self.detector, self.squat, self.curl,
            mp_pose, yolo_detector=self.yolo_detector, use_yolo=self.use_yolo,
        )
        self.last_state = state
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

st.session_state.input_source = st.sidebar.radio(
    "Input Source", ["Webcam (Live)", "Upload Video"],
    index=["Webcam (Live)", "Upload Video"].index(st.session_state.input_source),
)

exercise = st.sidebar.selectbox(
    "Exercise", list(EXERCISE_INFO.keys()),
    index=list(EXERCISE_INFO.keys()).index(st.session_state.exercise),
)
st.session_state.exercise = exercise

mode = st.sidebar.radio(
    "Difficulty Mode", ["Beginner", "Advanced"],
    index=["Beginner", "Advanced"].index(st.session_state.mode),
    help="Beginner: thresholds thode loose. Advanced: strict/professional form.",
)
if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.upload_squat_analyzer.set_mode(mode)
    st.session_state.upload_curl_analyzer.set_mode(mode)

yolo_detector = get_yolo_detector()
yolo_help = (
    "Multi-person scene me background ke logo ko ignore karta hai."
    if yolo_detector.available
    else "Enable karne ke liye: pip install ultralytics"
)
st.session_state.use_yolo = st.sidebar.checkbox(
    "Use YOLOv8 person filter (optional)",
    value=st.session_state.use_yolo,
    disabled=not yolo_detector.available,
    help=yolo_help,
)
if not yolo_detector.available:
    st.sidebar.caption("⚠️ YOLOv8 installed nahi hai - app full-frame mode me chal raha hai.")

st.session_state.sound_enabled = st.sidebar.checkbox("🔊 Sound alerts", value=st.session_state.sound_enabled)
st.session_state.weight_kg = st.sidebar.number_input(
    "Aapka weight (kg) - calorie estimate ke liye", min_value=30, max_value=200,
    value=st.session_state.weight_kg,
)
st.session_state.target_reps = st.sidebar.number_input(
    "Target reps (goal)", min_value=1, max_value=200, value=st.session_state.target_reps,
)

st.sidebar.divider()
if st.sidebar.button("🔄 New Session (reset counters)"):
    st.session_state.upload_squat_analyzer.reset()
    st.session_state.upload_curl_analyzer.reset()
    st.session_state.session_start_time = time.time()
    st.session_state.last_seen_good = 0
    st.session_state.last_seen_bad = 0
    st.session_state.video_playing = False
    if st.session_state.video_cap is not None:
        st.session_state.video_cap.release()
    st.session_state.video_cap = None
    st.session_state.video_temp_path = None
    st.session_state.uploaded_file_id = None
    st.session_state.last_video_state = None
    st.session_state._reset_webcam_flag = True
    st.rerun()


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🏋️ AI Gym Form Checker")
st.caption("Squat aur Bicep Curl ke liye real-time AI form correction, rep counting, aur workout summary - webcam ya video upload dono se.")

with st.expander("🌟 Is Project Ki Main Features Hain", expanded=False):
    st.markdown(
        "- **Real-time pose detection**\n"
        "- **Automatic repetition counting**\n"
        "- **Incorrect posture detection**\n"
        "- **Instant feedback**\n"
        "- **Easy-to-use interface**\n"
        "- **Low-cost solution using only a webcam**"
    )

col_video, col_info = st.columns([2.2, 1])
current_state = None

with col_video:
    if st.session_state.input_source == "Webcam (Live)":
        ctx = webrtc_streamer(
            key="gym-form-checker",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.set_config(
                st.session_state.exercise, st.session_state.mode,
                st.session_state.use_yolo, yolo_detector,
            )
            if st.session_state._reset_webcam_flag:
                ctx.video_processor.reset()
                st.session_state._reset_webcam_flag = False
            current_state = ctx.video_processor.last_state

        if ctx.state.playing:
            st_autorefresh(interval=700, key="webcam_refresh")

    else:  # ---------------- Upload Video mode ----------------
        uploaded = st.file_uploader("Video upload karo", type=["mp4", "mov", "avi", "mkv"])

        if uploaded is not None:
            new_id = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.uploaded_file_id != new_id:
                if st.session_state.video_cap is not None:
                    st.session_state.video_cap.release()
                st.session_state.video_cap = None
                st.session_state.video_playing = False
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
                tfile.write(uploaded.read())
                tfile.close()
                st.session_state.video_temp_path = tfile.name
                st.session_state.uploaded_file_id = new_id
                st.session_state.upload_squat_analyzer.reset()
                st.session_state.upload_curl_analyzer.reset()
                st.session_state.last_video_state = None

        c1, c2, c3 = st.columns(3)
        start_clicked = c1.button("▶ Start", disabled=st.session_state.video_temp_path is None)
        pause_clicked = c2.button("⏸ Pause")
        stop_clicked = c3.button("⏹ Stop & Reset")

        if start_clicked and st.session_state.video_temp_path:
            if st.session_state.video_cap is None:
                st.session_state.video_cap = cv2.VideoCapture(st.session_state.video_temp_path)
                st.session_state.session_start_time = time.time()
            st.session_state.video_playing = True

        if pause_clicked:
            st.session_state.video_playing = False

        if stop_clicked:
            st.session_state.video_playing = False
            if st.session_state.video_cap is not None:
                st.session_state.video_cap.release()
            st.session_state.video_cap = None
            st.session_state.video_temp_path = None
            st.session_state.uploaded_file_id = None
            st.session_state.upload_squat_analyzer.reset()
            st.session_state.upload_curl_analyzer.reset()
            st.session_state.last_video_state = None
            st.rerun()

        video_placeholder = st.empty()

        if st.session_state.video_playing and st.session_state.video_cap is not None:
            ret, frame = st.session_state.video_cap.read()
            if not ret:
                st.session_state.video_playing = False
                st.warning("Video khatam ho gaya. 'Stop & Reset' dabaakar naya video try karo.")
            else:
                pose_detector_upload = get_upload_pose_detector()
                annotated, state = process_frame(
                    frame, st.session_state.exercise, pose_detector_upload,
                    st.session_state.upload_squat_analyzer, st.session_state.upload_curl_analyzer,
                    mp_pose, yolo_detector=yolo_detector, use_yolo=st.session_state.use_yolo,
                )
                st.session_state.last_video_state = state
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

                fps = st.session_state.video_cap.get(cv2.CAP_PROP_FPS) or 25
                interval_ms = max(30, int(1000 / fps))
                st_autorefresh(interval=interval_ms, key="video_playback_refresh")
        elif st.session_state.last_video_state is not None:
            st.caption("⏸ Paused - Start dabaakar wapas chalu karo.")

        current_state = st.session_state.last_video_state

    st.info(f"📷 {EXERCISE_INFO[st.session_state.exercise]['camera_tip']}")

    # ------------------------- DASHBOARD -------------------------
    st.subheader("📊 Live Dashboard")

    if current_state:
        reps = current_state.get("reps", 0)
        good = current_state.get("good_reps", 0)
        bad = current_state.get("bad_reps", 0)
        form_score = current_state.get("form_score", 100)
        feedback = current_state.get("feedback", "")
    else:
        reps, good, bad, form_score = 0, 0, 0, 100
        feedback = "Session shuru karne ke liye camera/video Start karo"

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Reps", reps)
    m2.metric("Good / Bad", f"{good} / {bad}")
    m3.metric("Form Score", f"{form_score}%")

    goal = max(st.session_state.target_reps, 1)
    st.progress(min(reps / goal, 1.0), text=f"{reps}/{goal} reps (goal)")

    if "GALTI" in feedback:
        st.error(feedback)
    elif "detect nahi" in feedback or "laayein" in feedback:
        st.warning(feedback)
    else:
        st.success(feedback)

    elapsed = time.time() - st.session_state.session_start_time
    calories = estimate_calories(st.session_state.exercise, elapsed, st.session_state.weight_kg)
    st.caption(f"⏱️ Session time: {int(elapsed // 60)}m {int(elapsed % 60)}s   |   🔥 Est. Calories: {calories} kcal")

    if st.session_state.sound_enabled:
        if good > st.session_state.last_seen_good:
            st.markdown(get_audio_html("good"), unsafe_allow_html=True)
        elif bad > st.session_state.last_seen_bad:
            st.markdown(get_audio_html("bad"), unsafe_allow_html=True)
    st.session_state.last_seen_good = good
    st.session_state.last_seen_bad = bad

    st.divider()
    summary_text = build_summary_text(
        st.session_state.exercise, st.session_state.mode, reps, good, bad,
        form_score, elapsed, calories,
    )
    st.download_button(
        "📥 Download Workout Summary", data=summary_text,
        file_name="workout_summary.txt", mime="text/plain",
    )

with col_info:
    st.subheader(f"📋 {exercise} - Sahi Tarika")
    for i, step in enumerate(EXERCISE_INFO[exercise]["steps"], 1):
        st.markdown(f"**{i}.** {step}")

    st.subheader("✅ Ye app kya check karta hai")
    for c in EXERCISE_INFO[exercise]["checks"]:
        st.markdown(f"- {c}")

    st.divider()
    st.caption("Note: Pehli baar camera permission allow karni padegi. Achi lighting aur poora sharir frame me rakhna accuracy ke liye zaroori hai.")
