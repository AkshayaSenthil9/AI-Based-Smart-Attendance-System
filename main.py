
import cv2
import os
import json
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(APP_DIR, "dataset")
TRAINER_DIR = os.path.join(APP_DIR, "trainer")
STUDENTS_FILE = os.path.join(APP_DIR, "students.json")
ATTENDANCE_FILE = os.path.join(APP_DIR, "attendance.csv")

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_students():
    if not os.path.exists(STUDENTS_FILE):
        return {}
    with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_students(data):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Register Number", "Name", "Date", "Time", "Status"])

def register_student():
    reg = simpledialog.askstring("Student Registration", "Enter Register Number:")
    if not reg:
        return
    name = simpledialog.askstring("Student Registration", "Enter Student Name:")
    if not name:
        return

    students = load_students()
    students[str(reg)] = name
    save_students(students)

    person_dir = os.path.join(DATASET_DIR, str(reg))
    os.makedirs(person_dir, exist_ok=True)

    face_detector = cv2.CascadeClassifier(CASCADE)
    cam = cv2.VideoCapture(0)
    count = 0

    messagebox.showinfo(
        "Camera",
        "Camera will open.\nLook at the camera and slowly move your face.\nPress Q to stop early."
    )

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Samples: {count}/20", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Register Student - Press Q to stop", frame)
        k = cv2.waitKey(100) & 0xff
        if k == ord("q") or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Registration", f"{name} registered successfully with {count} face samples.")

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CASCADE)

    face_samples = []
    ids = []

    for reg in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, reg)
        if not os.path.isdir(person_dir):
            continue
        try:
            student_id = int(reg)
        except ValueError:
            continue

        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = detector.detectMultiScale(img)
            if len(faces) == 0:
                face_samples.append(img)
                ids.append(student_id)
            else:
                for (x, y, w, h) in faces:
                    face_samples.append(img[y:y+h, x:x+w])
                    ids.append(student_id)

    if not face_samples:
        messagebox.showwarning("Training", "No face samples found. Register at least one student first.")
        return

    recognizer.train(face_samples, __import__("numpy").array(ids))
    model_path = os.path.join(TRAINER_DIR, "trainer.yml")
    recognizer.write(model_path)
    messagebox.showinfo("Training", f"Model trained successfully.\nSamples: {len(face_samples)}")

def mark_attendance(reg, name):
    ensure_attendance_file()
    today = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["Register Number"] == str(reg) and row["Date"] == today:
            return False

    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            str(reg), name, today, now.strftime("%H:%M:%S"), "Present"
        ])
    return True

def start_attendance():
    model_path = os.path.join(TRAINER_DIR, "trainer.yml")
    if not os.path.exists(model_path):
        messagebox.showwarning("Attendance", "Train the model first.")
        return

    students = load_students()
    if not students:
        messagebox.showwarning("Attendance", "Register students first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    face_detector = cv2.CascadeClassifier(CASCADE)

    cam = cv2.VideoCapture(0)
    last_name = "Unknown"

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            student_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            reg = str(student_id)

            if confidence < 70 and reg in students:
                name = students[reg]
                last_name = f"{name} ({reg})"
                new_record = mark_attendance(reg, name)
                status = "PRESENT" if new_record else "ALREADY MARKED"
                color = (0, 255, 0)
            else:
                name = "Unknown"
                status = "NOT REGISTERED"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, status, (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(frame, "Press Q to exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("AI Smart Attendance", frame)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

def view_report():
    ensure_attendance_file()
    win = tk.Toplevel(root)
    win.title("Attendance Report")
    win.geometry("760x420")

    text = tk.Text(win, font=("Consolas", 11))
    text.pack(fill="both", expand=True, padx=10, pady=10)

    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        text.insert("1.0", f.read())

root = tk.Tk()
root.title("AI-Based Smart Attendance System")
root.geometry("520x430")
root.resizable(False, False)

tk.Label(root, text="AI-BASED SMART ATTENDANCE SYSTEM",
         font=("Arial", 18, "bold")).pack(pady=(30, 8))
tk.Label(root, text="Using Face Recognition",
         font=("Arial", 13)).pack(pady=(0, 25))

buttons = [
    ("1. Register Student", register_student),
    ("2. Train Face Recognition Model", train_model),
    ("3. Start Attendance", start_attendance),
    ("4. View Attendance Report", view_report),
]
for label, command in buttons:
    tk.Button(root, text=label, command=command,
              width=34, height=2, font=("Arial", 11)).pack(pady=8)

tk.Label(root, text="Demo project • Python + OpenCV + SQLite/CSV",
         font=("Arial", 9)).pack(side="bottom", pady=15)

ensure_attendance_file()
root.mainloop()
