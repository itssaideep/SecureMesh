# AI Powered IoT Honeypot

An advanced, multi-faceted honeypot system designed to attract, log, and analyze malicious attacks on IoT devices and SSH/Telnet services. The system features a realistic ESP8266-based hardware honeypot, a Cowrie-based software honeypot, a FastAPI backend for data aggregation, and a React frontend for visualization and AI-driven analysis.

## 🏗️ Architecture & Project Structure

The project is divided into four main components:

- **`backend/` (FastAPI)**: The core API that receives logs from the honeypots, stores them in MongoDB, and provides endpoints for the frontend to fetch data and trigger AI analysis.
- **`frontend/` (React + Vite)**: A modern dashboard built with React, TailwindCSS, and Chart.js to visualize attack metrics, display logs in real-time, and view AI-powered threat insights.
- **`esp8266/` (C++ / Arduino)**: Code for an ESP8266 microcontroller that acts as a physical honeypot. It hosts a fake "Smart Camera Administration" login portal and logs all credential guessing attempts to the backend.
- **`docker-compose.yml` (Cowrie & MongoDB)**: Containerized deployment for Cowrie (an interactive SSH and Telnet honeypot) and a MongoDB instance used by the backend.

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose
- [Node.js](https://nodejs.org/) (for the frontend)
- [Python 3.8+](https://www.python.org/) (for the backend)
- Arduino IDE (if deploying the ESP8266 honeypot)

### 1. Start the Containerized Services (MongoDB & Cowrie)

From the root of the project, spin up the Cowrie honeypot and MongoDB database:

```bash
docker-compose up -d
```

*Note: Cowrie will listen on ports `2222` (SSH) and `2223` (Telnet).*

### 2. Run the FastAPI Backend

Navigate to the `backend` directory, install dependencies, and start the server:

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

*The backend API will be available at `http://localhost:8000`.*

### 3. Run the React Frontend

Navigate to the `frontend` directory, install dependencies, and start the development server:

```bash
cd frontend
npm install
npm run dev
```

*The dashboard will be accessible at `http://localhost:5173`.*

### 4. Deploy the ESP8266 Hardware Honeypot (Optional)

1. Open `esp8266/esp8266.ino` in the Arduino IDE.
2. Update the `ssid` and `password` variables with your local WiFi credentials.
3. Update `backend_url` to point to the IP address of your machine running the FastAPI backend.
4. Flash the code to your ESP8266 board.
5. The ESP8266 will host a fake login page on port 80 and log all access attempts to the backend.

## 📊 Features

- **Multi-Vector Threat Detection**: Captures web-based credential stuffing (ESP8266) and interactive terminal sessions (Cowrie).
- **Real-Time Visualization**: Dashboard graphs showing attack frequency, top targeted credentials, and geographic locations.
- **AI Analysis**: Backend routes prepared for integration with AI models to automatically analyze attack patterns, commands executed in the Cowrie shell, and behavioral anomalies.

## 📜 License

This project is licensed under the MIT License.
