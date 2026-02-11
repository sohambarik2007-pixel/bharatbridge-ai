# BharatBridge AI  
## Bridging the Indian Sign Language (ISL) Communication Gap in Indian Campuses

BharatBridge AI is an AI-powered real-time communication assistant designed to bridge the Indian Sign Language (ISL) communication gap in educational campuses. The system enables seamless interaction between deaf/hard-of-hearing students and hearing faculty or peers using computer vision and speech technologies.

---

## 🎯 Problem Statement

Indian campuses lack real-time ISL interpretation systems. Deaf and hard-of-hearing students often struggle to participate fully in lectures, discussions, and administrative interactions due to the absence of accessible communication tools.

The limited availability of professional interpreters creates a communication barrier that impacts academic inclusion and equal opportunity.

---

## 💡 Solution Overview

BharatBridge AI provides a real-time bidirectional communication system:

- 🤟 **Sign-to-Text Conversion**
- 🎤 **Speech-to-Text (Live Captions)**
- 🔊 **Text-to-Speech Output**
- 🌍 Optional regional language support
- ⚡ Low-latency classroom-ready performance

This ensures inclusive and accessible communication within campus environments.

---

## 🔄 System Workflow

### 1️⃣ Sign → Text → Speech
- Camera captures ISL gestures
- Hand landmarks detected using MediaPipe
- CNN-LSTM model recognizes gesture sequences
- Recognized signs converted to text
- Text optionally converted to speech output

### 2️⃣ Speech → Text
- Microphone captures instructor speech
- Speech-to-text model (e.g., Whisper) transcribes audio
- Real-time captions displayed for students

---

## 🧠 AI Modules

### • Sign Language Recognition
- MediaPipe Hands for landmark detection
- CNN for spatial feature extraction
- LSTM for temporal sequence modeling
- Confidence scoring and gesture validation

### • Speech-to-Text
- Whisper / Google Speech API
- Real-time caption generation

### • Text-to-Speech
- Neural TTS engine for accessibility
- Adjustable speed and voice options

---

## 🏗 Backend Architecture

- FastAPI-based backend
- WebSocket for real-time streaming
- REST APIs for processing modules
- Session-based communication handling

---

## ☁ Cloud Infrastructure (AWS)

- EC2 (GPU) for Sign Recognition Model
- EC2 (CPU) for Speech Processing
- S3 for temporary storage
- API Gateway for routing
- Basic horizontal scaling support

---

## 🔒 Security & Privacy

- Encrypted video/audio streaming (HTTPS/WSS)
- No permanent storage of classroom video
- JWT-based authentication
- Temporary file lifecycle management

---

## 🚀 Prototype Scope

This project is designed as a hackathon-ready MVP focused on:

- Real-time gesture recognition
- Live speech captioning
- Low-latency classroom interaction
- Scalable campus deployment

---

## 🔮 Future Enhancements

- Expand ISL vocabulary (100 → 500+ signs)
- Text-to-Sign animated avatar
- Offline campus deployment
- Integration with LMS platforms
- Multi-classroom scaling

---

## 📌 Tech Stack

- Python (FastAPI)
- MediaPipe
- PyTorch / TensorFlow
- Whisper / Google Speech API
- AWS (EC2, S3, API Gateway)

---

## 🏁 Status

Hackathon Submission – AI for Bharat  
Version 2.0  
Focused on inclusive campus communication.

---

## 👥 Team

BharatBridge AI Team  
AI for Bharat Hackathon Submission