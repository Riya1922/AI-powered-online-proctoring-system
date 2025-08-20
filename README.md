# AI-powered-online-proctoring-system
The AI Online Proctoring System is a web-based platform that ensures fair online examinations using Artificial Intelligence.
It detects and records suspicious activities such as:

👀 Gaze detection (looking away from screen)

🚶 Unauthorized movement/violence detection

❌ Multiple face detection / impersonation

📷 Recording violations with evidence

This system uses YOLOv5/YOLOv8 models for detection and integrates with a web-based exam platform.

⚡ Features

Secure student login & authentication

Real-time monitoring with webcam feed

AI-powered detection of violations

Automatic alerts for suspicious behavior

Teacher dashboard to review violations

Exam creation & management

🛠️ Tech Stack

Backend: Python (Flask/Django – whichever you used)

Frontend: HTML, CSS, JavaScript (inside templates/ & static/)

AI Models: YOLOv5, YOLOv8

Database: (SQLite / MySQL – whichever you used)

📂 Project Structure
├── exam/             # Exam logic
├── models/           # Model-related code
├── notebook/         # Jupyter notebooks (experiments)
├── onlineexam/       # Core app
├── static/           # CSS, JS, Images
├── student/          # Student module
├── teacher/          # Teacher module
├── templates/        # HTML Templates
├── violations/       # Stored violation logs
├── violence/         # Violence detection logic
├── add_questions.py  # Add exam questions
├── gaze_detector.py  # Gaze detection script
├── main.py           # Entry point
├── requirements.txt  # Dependencies
└── questions.json    # Sample questions

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/Riya1922/AI-ppowered-online-proctoring-system.git


Create virtual environment & install dependencies:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
Download pre-trained YOLO models

YOLOv5 Models

YOLOv8 Models

Place them inside the models/ folder.

Run the project:

python main.py
