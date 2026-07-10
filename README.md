# 🏋️ AI Gym Form Checker

Ek **production-ready** real-time AI fitness trainer jo **sirf 2 exercises** par focused hai taaki accuracy sabse zyada high rahe:

1. **Squat**
2. **Bicep Curl**

App camera (live webcam) ya uploaded video se aapki body ke landmarks detect karta hai (MediaPipe Pose), joint angles nikal kar batata hai ki exercise **sahi ho rahi hai ya nahi**, reps count karta hai, aur turant (real-time) galti bata deta hai - angle number ke saath frame par visualize bhi karta hai.

---

## 🌟 Is Project Ki Main Features Hain:

* **Real-time pose detection**
* **Automatic repetition counting**
* **Incorrect posture detection**
* **Instant feedback**
* **Easy-to-use interface**
* **Low-cost solution using only a webcam**

---

## ✨ Detailed Features

| Feature | Detail |
|---|---|
| 🎥 **Webcam (Live)** | Real-time browser webcam feed pe form check |
| 📁 **Upload Video** | Pehle se recorded video upload karke analyze karo |
| 🔁 **Rep Counter** | Accurate state-machine based rep counting (Up→Down→Up / Down→Up→Down) |
| ✅ **Real-time Form Feedback** | Har frame par turant "GALTI" ya "Sahi" feedback |
| 📐 **Angle Visualization** | Joint (knee/elbow) ke paas exact angle number screen par dikhta hai |
| 🎯 **Form Score (0-100%)** | Good reps vs bad reps ke hisaab se overall score |
| 🔥 **Calorie Estimate** | MET-formula se session duration + weight ke hisaab se calories |
| 📊 **Progress Bar** | Target reps goal ke against live progress |
| 🔊 **Sound Alerts** | Sahi rep par high-beep, galat rep par low-buzzer |
| 🧑‍🎓 **Beginner / Advanced Mode** | Thresholds automatically loose (Beginner) ya strict (Advanced) ho jaate hai |
| 🧍 **YOLOv8 Person Filter (optional)** | Multi-person scene me background logo ko ignore karta hai |
| 📥 **Workout Summary Download** | Session ke end me `.txt` summary download karo |

---

## 🧠 Kaise kaam karta hai (Tech Stack)

| Cheez | Use |
|---|---|
| **MediaPipe Pose** | Body ke 33 landmarks (joints) real-time detect karta hai |
| **OpenCV** | Video frame process aur draw karta hai (skeleton, angle, overlay) |
| **NumPy** | Joint angles calculate karne ke liye math |
| **Streamlit** | Web app UI (dashboard, sidebar, controls) |
| **streamlit-webrtc** | Live camera feed ko browser se seedha Python tak laata hai |
| **streamlit-autorefresh** | Dashboard ko smoothly live update karta hai (bina buttons block kiye) |
| **YOLOv8 (optional, ultralytics)** | Person detect karke frame ko crop karta hai - background noise filter |

Har exercise ke liye ek alag **Analyzer** class hai jo sirf angles ke through decide karti hai ki form sahi hai ya galat - isliye speed aur accuracy dono high rehte hai.

### 🏋️ Squat me kya check hota hai
- **Depth**: Knee angle threshold se neeche jaana chahiye (poori squat)
- **Back posture**: Peeth zyada aage nahi jhukni chahiye (shoulder-hip-knee angle)
- **Knee-over-toe**: Ghutna toes ki line se aage nahi jaana chahiye

### 💪 Bicep Curl me kya check hota hai
- **Full range of motion**: Neeche baju poori seedhi, upar poora curl
- **Elbow stability**: Kohni body se chipki rehni chahiye (swing/cheating detect)

### 🎚️ Beginner vs Advanced Mode
- **Beginner**: Thresholds thode loose - naye users ke liye encouraging
- **Advanced**: Thresholds strict - professional/gym-level form expect karta hai

---

## 📂 Project Structure

Sab files ek hi folder me hai:

```
gym_form_checker/
├── app.py                     # Main Streamlit app (UI, webcam, video upload, dashboard)
├── pose_detector.py           # MediaPipe Pose wrapper
├── angle_calculator.py        # Joint angle math + smoothing
├── squat_analyzer.py          # Squat rep counter + form checker
├── bicep_curl_analyzer.py     # Curl rep counter + form checker
├── yolo_detector.py           # Optional YOLOv8 person filter
├── sound_utils.py             # Beep/alert sound generator
├── workout_summary.py         # Calorie estimate + summary text generator
├── frame_processor.py         # Shared pipeline (webcam + video dono use karte hai)
├── requirements.txt           # Core dependencies
├── requirements-yolo.txt      # Optional YOLO dependency
└── README.md
```

---

## 🛠️ Installation (Setup Steps)

1. **Python 3.9 - 3.12** install hona chahiye.

2. Project folder me terminal khol kar virtual environment banao:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows par: venv\Scripts\activate
   ```

3. Core dependencies install karo:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional)** Agar YOLOv8 person-filter feature bhi chahiye:
   ```bash
   pip install -r requirements-yolo.txt
   ```
   Iske bina bhi app poori tarah kaam karega - YOLO checkbox automatically disabled ho jayega.

5. App run karo:
   ```bash
   streamlit run app.py
   ```

6. Browser me app khulega (usually `http://localhost:8501`).

---

## 📖 Use Karne Ka Tarika (Step by Step)

1. **Sidebar** se settings choose karo:
   - **Input Source**: Webcam (Live) ya Upload Video
   - **Exercise**: Squat ya Bicep Curl
   - **Difficulty Mode**: Beginner ya Advanced
   - **YOLOv8 filter**: agar installed hai to on kar sakte ho (multi-person background filter)
   - **Sound alerts**: on/off
   - **Weight (kg)**: calorie estimate ke liye
   - **Target reps**: aapka goal (progress bar ke liye)

2. **Webcam mode**: "START" button dabao, camera permission allow karo.
   **Upload Video mode**: video file upload karo, phir "▶ Start" dabao.

3. **"Sahi Tarika"** panel (right side) me diye steps ek baar padh lo - exercise karne se pehle.

4. Camera ke saamne **side (profile) angle** se khade ho jao taaki poora sharir frame me dikhe (camera tip bhi screen par dikhta hai).

5. Exercise shuru karo - Live Dashboard turant dikhayega:
   - **Total Reps**, **Good/Bad reps**, **Form Score %**
   - **Progress bar** (target ke against)
   - **Feedback message** (green = sahi, red = "GALTI: ...")
   - **Session time + estimated calories**
   - Angle number seedha video frame par joint ke paas

6. Jab exercise khatam ho jaaye, **"📥 Download Workout Summary"** dabaakar apna session summary save kar lo.

7. Naya session shuru karne ke liye sidebar me **"🔄 New Session"** button dabao (counters reset ho jaayenge).

---

## ✅ Best Results Ke Liye Tips

- Achi roshni (lighting) me practice karo - andhera hone par landmark detection kamzor ho jata hai.
- Camera se poori body distance par khade ho jao (2-3 meter door, side view).
- Fitted/plain kapde pehno - bahut loose kapde landmarks ko galat detect kara sakte hai.
- Agar bahut saare log frame me hai, YOLOv8 filter on karo taaki app sirf exercise karne wale insaan par focus kare.

---

## ⚠️ Troubleshooting

- **Camera nahi khul raha**: Browser ki site settings me camera permission manually allow karo.
- **"mediapipe" install error**: `requirements.txt` me pinned version (`0.10.13`) hi use karo - naye versions me legacy Pose API missing ho sakti hai.
- **YOLO checkbox disabled hai**: `pip install -r requirements-yolo.txt` chalao (optional feature, torch install karta hai isliye size bada hai).
- **Video upload slow/laggy lag raha hai**: Chhoti duration ka video try karo ya resolution kam karo.
- **Landmarks flicker ho rahe / accuracy kam lag rahi**: Lighting improve karo aur camera se thoda door khade ho jao taaki poora body frame me aaye.

---

## 🧪 Testing

Ye project banate waqt actual test kiya gaya hai:
- Sabhi modules syntax + import verified
- Squat aur Curl analyzers ko simulated joint-angle data se test kiya gaya (depth check, back-lean check, knee-over-toe check, elbow-drift check - sab verified)
- Poora Streamlit app `streamlit.testing.v1.AppTest` se run karke verify kiya gaya (sidebar controls, exercise/mode switching, video upload + playback loop, Start/Pause/Stop/New Session buttons) - zero errors
- Real Streamlit server start karke HTTP response bhi verify kiya gaya
#   A I - G Y M - F R O M - C H E C K E R  
 #   A I - G Y M - F R O M - C H E C K E R  
 #   A I - G Y M - F R O M - C H E C K E R  
 