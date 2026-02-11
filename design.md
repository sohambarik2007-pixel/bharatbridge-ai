# Bridging the Indian Sign Language (ISL) Communication Gap in Indian Campuses
## System Design Document

## 1. System Overview

### The Communication Gap Problem

Indian educational campuses face a significant accessibility challenge: deaf and hard-of-hearing students struggle to participate in real-time classroom discussions, group projects, and casual conversations with hearing peers and faculty. While sign language interpreters exist, they are:

- Expensive and not available for every class
- Limited in number, creating scheduling conflicts
- Not available for informal student inter3actions
- Unable to scale across multiple simultaneous sessions

This creates an isolation barrier that affects academic performance, social integration, and overall campus experience for deaf/hard-of-hearing students.

### Solution Vision

An AI-powered real-time communication assistant that enables bidirectional communication between deaf/hard-of-hearing students and the hearing campus community. The system acts as a digital interpreter, providing:

**For Deaf/Hard-of-Hearing Students**:
- Sign language recognition converting ISL gestures to text/speech
- Real-time display of spoken conversations as text
- Ability to respond using sign language

**For Hearing Faculty/Students**:
- Speech-to-text conversion of their spoken words
- Text-to-speech output of sign language responses
- No need to learn sign language for basic communication

**Key Design Principles**:
- Real-time processing with <2 second latency
- Accessible on common devices (laptops, tablets, phones)
- Works in typical classroom environments
- Privacy-focused with no permanent video storage
- Scalable to multiple concurrent sessions


## 2. End-to-End Workflow

### 2.1 Sign-to-Text-to-Speech Pipeline

**Use Case**: Deaf student communicates with hearing teacher/peers

**Step 1: Video Capture**
- Student's device camera captures sign language gestures
- Frontend samples frames at 10 FPS (reduced from 30 FPS for bandwidth)
- Frames sent to backend via WebSocket connection
- Real-time streaming with minimal buffering

**Step 2: Hand Detection**
- MediaPipe Hands model detects hand landmarks (21 keypoints per hand)
- Processes both hands simultaneously for two-handed signs
- Filters out low-confidence frames (threshold: 0.7)
- Normalizes coordinates relative to frame center

**Step 3: Gesture Recognition**
- Sliding window collects 30 frames (3 seconds of gesture)
- LSTM model processes temporal sequence
- Outputs recognized ISL sign with confidence score
- Handles continuous signing with overlapping windows

**Step 4: Text Formation**
- Recognized signs converted to English/Hindi text
- Basic grammar correction applied
- Sentence formation from sign sequence
- Text displayed on screen in real-time

**Step 5: Speech Synthesis (Optional)**
- Text-to-speech converts text to audio
- Plays through speaker for hearing participants
- Adjustable voice speed and volume
- Can be muted for text-only mode

**Latency Target**: <2 seconds from gesture completion to text display

### 2.2 Speech-to-Text Pipeline

**Use Case**: Hearing teacher/peer communicates with deaf student

**Step 1: Audio Capture**
- Microphone captures spoken audio
- Frontend performs noise cancellation
- Audio chunks sent to backend (1-second intervals)
- Continuous streaming for real-time transcription

**Step 2: Speech Recognition**
- Whisper model or Google STT processes audio
- Supports English and Hindi (primary campus languages)
- Handles Indian accents and classroom acoustics
- Outputs transcribed text with timestamps

**Step 3: Text Display**
- Transcribed text displayed on deaf student's screen
- Large, readable font with high contrast
- Auto-scrolling with conversation history
- Speaker identification (if multiple speakers)

**Step 4: Optional Translation**
- Basic translation to regional languages if needed
- Primarily for non-English speaking faculty
- Uses simple translation API (Google Translate)
- Not the primary focus for MVP

**Latency Target**: <1 second from speech to text display

### 2.3 Text-Based Communication Flow

**Use Case**: Asynchronous or supplementary text communication

**Deaf Student Input**:
- Can type text directly if preferred
- Useful for complex concepts or names
- Faster than signing for some contexts
- Text can be converted to speech for hearing participants

**Hearing Participant Input**:
- Can type instead of speaking in quiet environments
- Useful for sharing links, references, or complex terms
- Text displayed directly to deaf student
- No processing latency

**Shared Features**:
- Conversation history saved for session duration
- Copy/paste functionality
- Screenshot capability for notes
- Session export as text file


## 3. AI Module Architecture

### 3.1 Sign Language Recognition Module

**Model Architecture**: CNN-LSTM Hybrid

**Components**:

1. **Hand Detection**: MediaPipe Hands
   - Pre-trained Google model for hand landmark detection
   - Detects 21 3D landmarks per hand
   - Runs on-device or server-side depending on deployment
   - Lightweight and fast (30+ FPS capable)

2. **Feature Extraction**: Convolutional Neural Network
   - Input: Hand landmark coordinates (21 × 3 × 2 hands = 126 features)
   - 3 convolutional layers with batch normalization
   - Extracts spatial features (hand shape, finger positions)
   - Output: 256-dimensional feature vector per frame

3. **Sequence Modeling**: Bidirectional LSTM
   - Input: Sequence of 30 feature vectors (3 seconds)
   - 2 LSTM layers with 128 hidden units each
   - Captures temporal dynamics (movement patterns)
   - Dropout (0.3) for regularization

4. **Classification**: Fully Connected Layers
   - 2 dense layers (256 → 128 → num_classes)
   - Softmax activation for probability distribution
   - Output: Predicted ISL sign + confidence score

**Training Data**:
- INCLUDE dataset (Indian Sign Language)
- Custom collected data: 150 common campus-related signs
- Signs include: greetings, questions, academic terms, common phrases
- 30-50 samples per sign from 8-10 different signers
- Data augmentation: rotation, scaling, speed variation

**Model Size**: ~50MB (deployable on edge devices)
**Inference Time**: ~100ms per gesture prediction
**Accuracy Target**: 85%+ on test set

### 3.2 Speech-to-Text Module

**Model Choice**: OpenAI Whisper (Small or Base model)

**Rationale**:
- Excellent performance on Indian accents
- Supports English and Hindi
- Open-source and free to use
- Can run on CPU or GPU

**Alternative**: Google Cloud Speech-to-Text
- Better accuracy but requires API costs
- Use for production if budget allows
- Fallback to Whisper for cost-sensitive deployment

**Configuration**:
- Model: Whisper-small (244M parameters) or Whisper-base (74M)
- Language: English (primary), Hindi (secondary)
- Audio format: 16kHz, mono, 16-bit PCM
- Chunk size: 1-second audio segments for real-time processing

**Performance**:
- Word Error Rate (WER): 10-15% on Indian English
- Latency: 500ms - 1s per audio chunk
- Handles classroom noise reasonably well

### 3.3 Text-to-Speech Module

**Model Choice**: gTTS (Google Text-to-Speech) or Piper TTS

**Primary Option**: gTTS
- Free API with reasonable limits
- Natural-sounding voices
- Supports English and Hindi
- Simple integration

**Offline Alternative**: Piper TTS
- Open-source, runs locally
- Smaller model size (~10-20MB per language)
- Slightly lower quality but acceptable
- No API costs or internet dependency

**Configuration**:
- Voice: Female (default, research shows better clarity)
- Speed: Adjustable (0.8x - 1.2x)
- Language: Match input text language
- Output format: MP3 or WAV

**Performance**:
- Latency: 500ms - 1s for typical sentence
- Quality: Natural prosody, clear pronunciation

### 3.4 Optional Translation Module

**Model Choice**: Google Translate API (Basic Tier)

**Scope** (MVP):
- English ↔ Hindi translation only
- Used sparingly to minimize API costs
- Primarily for faculty who prefer Hindi
- Not core to the ISL communication flow

**Future Enhancement**:
- Add more regional languages (Tamil, Telugu, Bengali)
- Use open-source models (IndicTrans2) to reduce costs
- Context-aware translation for academic terms

**Performance**:
- Latency: 200-500ms per translation
- Accuracy: 80-85% for simple sentences


## 4. Gesture Recognition Technical Details

### 4.1 Hand Landmark Detection

**MediaPipe Hands Pipeline**:

**Input**: RGB video frame (640×480 or 1280×720)

**Processing Steps**:
1. Palm detection: Locates hand regions in frame
2. Hand landmark model: Predicts 21 3D landmarks
3. Tracking: Maintains hand identity across frames

**Landmark Points** (21 per hand):
- Wrist (1 point)
- Thumb (4 points: CMC, MCP, IP, TIP)
- Index finger (4 points: MCP, PIP, DIP, TIP)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

**Output Format**:
```python
{
  "left_hand": {
    "landmarks": [[x1, y1, z1], [x2, y2, z2], ...],  # 21 points
    "confidence": 0.95
  },
  "right_hand": {
    "landmarks": [[x1, y1, z1], [x2, y2, z2], ...],
    "confidence": 0.92
  }
}
```

**Normalization**:
- Coordinates normalized to [0, 1] range
- Origin shifted to wrist position
- Scale normalized by hand size
- Rotation invariance through alignment

### 4.2 Frame Sampling Strategy

**Challenge**: Balance between accuracy and latency

**Approach**:
- Camera captures at 30 FPS
- Sample every 3rd frame → 10 FPS for processing
- Reduces bandwidth by 67%
- Still captures smooth gesture motion

**Buffering**:
- Maintain sliding window of 30 frames (3 seconds)
- 50% overlap between windows (15 frames)
- Allows continuous gesture recognition
- Handles variable-speed signing

**Quality Filtering**:
- Reject frames with hand confidence <0.7
- Interpolate missing frames (up to 3 consecutive)
- Discard sequences with >30% low-quality frames
- Request user to re-sign if quality insufficient

### 4.3 Feature Engineering

**Spatial Features** (per frame):
- Finger angles: 5 angles per hand (10 total)
- Palm orientation: 3D rotation vector
- Hand shape descriptor: Distance matrix between landmarks
- Hand position: Relative to frame center
- Feature vector size: 126 raw + 50 engineered = 176 features

**Temporal Features** (across frames):
- Movement velocity: Frame-to-frame displacement
- Acceleration: Second derivative of position
- Trajectory: Path traced by key landmarks (wrist, fingertips)
- Movement smoothness: Jerk (third derivative)

**Normalization**:
- Z-score normalization per feature
- Removes signer-specific variations (hand size, speed)
- Improves model generalization

### 4.4 Sequence Modeling with LSTM

**Architecture**:
```
Input: (batch_size, 30 frames, 176 features)
↓
Bidirectional LSTM Layer 1: 128 units
↓
Dropout: 0.3
↓
Bidirectional LSTM Layer 2: 128 units
↓
Dropout: 0.3
↓
Dense Layer: 256 units, ReLU
↓
Dense Layer: 128 units, ReLU
↓
Output Layer: num_classes, Softmax
```

**Why LSTM?**:
- Captures long-term dependencies in gesture sequences
- Handles variable-length signs naturally
- Bidirectional processing improves accuracy
- Proven effective for sign language recognition

**Training Details**:
- Loss: Categorical cross-entropy
- Optimizer: Adam (learning rate: 0.001)
- Batch size: 32
- Epochs: 50 with early stopping
- Validation split: 20%

### 4.5 Confidence Scoring

**Prediction Confidence**:
- Softmax probability of predicted class
- Threshold: 0.6 for acceptance
- Below threshold: Display "unclear, please repeat"

**Sequence Quality Score**:
- Average hand detection confidence across frames
- Percentage of valid frames in sequence
- Movement smoothness metric
- Combined score: weighted average

**User Feedback**:
- High confidence (>0.8): Green indicator
- Medium confidence (0.6-0.8): Yellow indicator
- Low confidence (<0.6): Red indicator, request re-sign
- Helps users improve signing clarity

### 4.6 Handling Edge Cases

**Partial Occlusion**:
- MediaPipe handles minor occlusions
- If one hand occluded, use single-hand signs only
- Confidence drops trigger quality warning

**Background Clutter**:
- MediaPipe's palm detection is robust
- Encourage plain background for best results
- Lighting: Recommend well-lit environment

**Signer Variation**:
- Model trained on diverse signers
- Handles different hand sizes, speeds, styles
- Continuous learning from user corrections

**Ambiguous Signs**:
- Some ISL signs are visually similar
- Use context from previous signs
- Allow manual correction by user
- Log ambiguous cases for model improvement


## 5. Backend Architecture

### 5.1 Framework: FastAPI

**Why FastAPI?**:
- Native async support for WebSocket connections
- Fast performance (comparable to Node.js)
- Automatic API documentation (Swagger UI)
- Type hints and validation with Pydantic
- Easy integration with ML models (Python ecosystem)

**Project Structure**:
```
backend/
├── app/
│   ├── main.py                 # FastAPI app initialization
│   ├── config.py               # Configuration settings
│   ├── models/
│   │   ├── sign_recognition.py # ISL model wrapper
│   │   ├── speech_to_text.py   # Whisper model wrapper
│   │   └── text_to_speech.py   # TTS wrapper
│   ├── routes/
│   │   ├── websocket.py        # WebSocket endpoints
│   │   ├── speech.py           # Speech-to-text API
│   │   └── tts.py              # Text-to-speech API
│   ├── services/
│   │   ├── gesture_processor.py
│   │   ├── audio_processor.py
│   │   └── session_manager.py
│   └── utils/
│       ├── mediapipe_handler.py
│       └── preprocessing.py
├── models/                     # Trained model files
├── requirements.txt
└── Dockerfile
```

### 5.2 WebSocket for Real-time Streaming

**Sign Language Recognition WebSocket**:

**Endpoint**: `ws://domain/ws/sign-language/{session_id}`

**Client → Server** (Video frames):
```json
{
  "type": "frame",
  "data": "base64_encoded_image",
  "timestamp": 1234567890,
  "frame_number": 42
}
```

**Server → Client** (Recognition results):
```json
{
  "type": "recognition",
  "sign": "hello",
  "text": "Hello",
  "confidence": 0.92,
  "timestamp": 1234567890
}
```

**Connection Management**:
- Heartbeat 

**Connection Management**:
- Heartbeat ping every 30 seconds
- Auto-reconnect on connection drop
- Session timeout: 2 hours of inactivity
- Graceful disconnection handling

**Speech-to-Text WebSocket**:

**Endpoint**: `ws://domain/ws/speech-to-text/{session_id}`

**Client → Server** (Audio chunks):
```json
{
  "type": "audio",
  "data": "base64_encoded_audio",
  "format": "wav",
  "sample_rate": 16000
}
```

**Server → Client** (Transcription):
```json
{
  "type": "transcription",
  "text": "What is the assignment deadline?",
  "language": "en",
  "confidence": 0.88
}
```

### 5.3 REST APIs

**Text-to-Speech API**:
```
POST /api/tts/generate
Request:
{
  "text": "Hello, how are you?",
  "language": "en",
  "speed": 1.0
}

Response:
{
  "audio_url": "https://s3.../audio_123.mp3",
  "duration_seconds": 2.5
}
```

**Translation API** (Optional):
```
POST /api/translate
Request:
{
  "text": "Hello",
  "source_lang": "en",
  "target_lang": "hi"
}

Response:
{
  "translated_text": "नमस्ते",
  "confidence": 0.95
}
```

**Session Management**:
```
POST /api/session/create
Response:
{
  "session_id": "abc123",
  "expires_at": "2026-02-11T18:00:00Z"
}

GET /api/session/{session_id}/history
Response:
{
  "messages": [
    {"type": "sign", "text": "Hello", "timestamp": "..."},
    {"type": "speech", "text": "Hi there", "timestamp": "..."}
  ]
}
```

### 5.4 Service Components

**Gesture Processor Service**:
- Receives video frames from WebSocket
- Runs MediaPipe hand detection
- Buffers frames in sliding window
- Feeds sequences to LSTM model
- Returns recognized signs

**Audio Processor Service**:
- Receives audio chunks from WebSocket
- Preprocesses audio (noise reduction, normalization)
- Runs Whisper model for transcription
- Returns transcribed text

**Session Manager Service**:
- Creates and manages communication sessions
- Stores conversation history (in-memory for MVP)
- Handles user authentication
- Cleans up expired sessions

**Model Manager**:
- Loads ML models at startup
- Manages model versions
- Handles model inference requests
- Implements request queuing for GPU

### 5.5 Deployment Configuration

**Docker Container**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/
COPY models/ /models/

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Requirements.txt** (Key dependencies):
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6
mediapipe==0.10.8
opencv-python==4.8.1
numpy==1.24.3
torch==2.1.0
openai-whisper==20231117
gTTS==2.4.0
pydantic==2.5.0
python-jose[cryptography]==3.3.0
```


## 6. Cloud Infrastructure (AWS-based, Hackathon Realistic)

### 6.1 Compute: Amazon EC2

**Instance Configuration**:

**GPU Instance** (for Sign Language Recognition):
- Type: g4dn.xlarge (1 NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM)
- Quantity: 1 instance (can scale to 2-3 for multiple classrooms)
- Purpose: Run LSTM model inference, MediaPipe processing
- Cost: ~$0.526/hour = ~$380/month (if running 24/7)

**CPU Instance** (for Speech-to-Text):
- Type: t3.medium (2 vCPUs, 4 GB RAM)
- Quantity: 1 instance
- Purpose: Run Whisper model (small version works on CPU)
- Cost: ~$0.0416/hour = ~$30/month

**Cost Optimization for Hackathon**:
- Use Spot Instances (70% discount, risk of interruption acceptable for demo)
- Run instances only during demo/testing hours
- Estimated hackathon cost: $50-100 for 1 week of testing

**AMI Setup**:
- Ubuntu 22.04 LTS
- Pre-installed: Docker, NVIDIA drivers, CUDA toolkit
- Custom AMI with models pre-loaded for fast startup

### 6.2 Storage: Amazon S3

**Buckets**:

1. **Model Storage Bucket**:
   - Stores trained ML models (LSTM, Whisper)
   - Size: ~500 MB
   - Access: Private, EC2 instances only
   - Cost: Negligible (~$0.01/month)

2. **Audio Files Bucket**:
   - Temporary storage for generated TTS audio
   - Lifecycle policy: Delete after 1 hour
   - Size: ~100 MB at any time
   - Cost: Negligible

3. **Session Data Bucket** (Optional):
   - Store conversation history for analytics
   - Encrypted at rest
   - Cost: ~$0.50/month for 10 GB

**Total S3 Cost**: <$1/month

### 6.3 API Gateway

**Purpose**:
- Single entry point for all API requests
- WebSocket API support for real-time communication
- Request throttling and rate limiting
- HTTPS termination

**Configuration**:
- REST API for TTS, translation endpoints
- WebSocket API for sign language and speech-to-text
- Custom domain: isl-bridge.yourdomain.com
- Rate limit: 100 requests/minute per user

**Cost**: ~$3.50 for 1M requests (hackathon: <$1)

### 6.4 Load Balancing (Optional for MVP)

**Application Load Balancer**:
- Distributes traffic across EC2 instances
- Health checks every 30 seconds
- Sticky sessions for WebSocket connections
- Only needed if scaling beyond 1 instance

**Cost**: ~$16/month + $0.008/LCU-hour
**Hackathon Decision**: Skip for MVP, add if scaling needed

### 6.5 Database (Minimal for MVP)

**Option 1: In-Memory (Redis on EC2)**:
- Store active sessions and conversation history
- Fast access, no persistence needed
- Run on same EC2 instance as backend
- Cost: $0 (included in EC2)

**Option 2: DynamoDB** (if persistence needed):
- Store user profiles, session history
- On-demand billing
- Cost: ~$1-2/month for hackathon usage

**Hackathon Choice**: In-memory storage, no separate database

### 6.6 Networking

**VPC Configuration**:
- Single VPC with public subnet
- Security group: Allow ports 80, 443, 8000 (API), 22 (SSH)
- Elastic IP for stable endpoint
- Cost: $0 (free tier)

**Data Transfer**:
- Inbound: Free
- Outbound: First 100 GB free, then $0.09/GB
- Estimated hackathon usage: <10 GB
- Cost: $0

### 6.7 Deployment Strategy

**Simple Deployment** (Hackathon):
1. Launch EC2 instance with custom AMI
2. Pull Docker container from Docker Hub or ECR
3. Run container with environment variables
4. Configure security groups and Elastic IP
5. Point domain to Elastic IP

**CI/CD** (Optional):
- GitHub Actions for automated deployment
- Push to main branch → Build Docker image → Deploy to EC2
- Overkill for hackathon, but good for demo

**Monitoring**:
- CloudWatch for basic metrics (CPU, memory, network)
- Application logs to CloudWatch Logs
- Set up alarms for high CPU or errors
- Cost: Free tier covers hackathon usage

### 6.8 Total Infrastructure Cost

**Hackathon (1 week of testing)**:
- EC2 GPU (Spot): ~$15
- EC2 CPU: ~$5
- S3: <$1
- API Gateway: <$1
- Data Transfer: $0
- **Total: ~$25-30**

**Monthly (if running 24/7)**:
- EC2 GPU: ~$380
- EC2 CPU: ~$30
- S3: ~$1
- API Gateway: ~$5
- **Total: ~$420/month**

**Cost Optimization**:
- Use Spot Instances: Save 70% → ~$130/month
- Run only during campus hours (8 AM - 6 PM): Save 60% → ~$170/month
- Combined: ~$50-80/month for production


## 7. Scalability Plan

### 7.1 Handling Multiple Classrooms

**Current Capacity** (1 GPU instance):
- Sign language recognition: ~10 FPS processing
- Concurrent users: 5-8 simultaneous sessions
- Bottleneck: GPU inference time

**Scaling Strategy**:

**Horizontal Scaling**:
- Add more EC2 GPU instances as needed
- Load balancer distributes sessions across instances
- Each instance handles 5-8 concurrent sessions
- Linear scaling: 2 instances = 10-16 sessions

**Session Affinity**:
- Once a session starts, it stays on the same instance
- Prevents model loading overhead
- Implemented via sticky sessions in load balancer

**Auto Scaling Policy**:
- Metric: Average GPU utilization
- Scale up: When GPU >75% for 2 minutes
- Scale down: When GPU <30% for 10 minutes
- Min instances: 1, Max instances: 5 (covers 25-40 classrooms)

**Cost Consideration**:
- Start with 1 instance for pilot
- Scale based on actual usage patterns
- Most campuses won't need >2 instances initially

### 7.2 Low-Latency Communication

**Target Latency Breakdown**:
- Video frame transmission: 100-200ms
- Hand detection (MediaPipe): 30-50ms
- LSTM inference: 50-100ms
- Text transmission: 50-100ms
- **Total: 230-450ms** (well under 2-second target)

**Optimization Techniques**:

**Frame Compression**:
- Compress frames before transmission (JPEG quality: 80)
- Reduces bandwidth by 60-70%
- Minimal impact on hand detection accuracy

**Model Optimization**:
- Use TorchScript for faster inference
- Batch processing when possible (batch size: 4-8)
- GPU memory management to avoid swapping

**Network Optimization**:
- WebSocket for persistent connections (no HTTP overhead)
- Binary data transfer instead of JSON for frames
- CDN for static assets (frontend)

**Caching**:
- Cache frequently recognized signs
- Reduce redundant processing
- In-memory cache on GPU instance

### 7.3 Horizontal Scaling of Sign Recognition Service

**Microservice Architecture**:

**Service Separation**:
1. **Sign Recognition Service**: GPU instances
2. **Speech-to-Text Service**: CPU instances
3. **TTS Service**: CPU instances or API calls
4. **API Gateway**: Lightweight routing

**Benefits**:
- Scale each service independently
- Sign recognition needs GPU, others don't
- Cost-efficient resource allocation

**Load Balancing Strategy**:
- Round-robin for new sessions
- Least-connections for optimal distribution
- Health checks to avoid routing to overloaded instances

**State Management**:
- Sessions are stateful (video stream continuity)
- Use sticky sessions to maintain connection
- Session data stored in Redis (shared across instances)

**Failover Handling**:
- If instance fails, session reconnects to new instance
- User sees brief interruption (<5 seconds)
- Conversation history preserved in Redis

### 7.4 Database Scaling (If Needed)

**Current Approach**: In-memory (Redis)
- Fast, simple, sufficient for MVP
- No persistence needed for real-time communication

**If Persistence Required**:
- DynamoDB with on-demand billing
- Auto-scales with traffic
- No manual capacity planning

**Data Model**:
- Sessions: Partition key = session_id
- User profiles: Partition key = user_id
- Conversation history: Partition key = session_id, Sort key = timestamp

### 7.5 Bandwidth Considerations

**Per Session Bandwidth**:
- Video upload: 10 FPS × 50 KB/frame = 500 KB/s = 4 Mbps
- Audio upload: 16 kHz × 2 bytes = 32 KB/s = 256 Kbps
- Text/results download: <10 KB/s = 80 Kbps
- **Total per session: ~4.5 Mbps**

**Campus Network**:
- Typical campus WiFi: 50-100 Mbps per access point
- Can support 10-20 concurrent sessions per AP
- Recommend dedicated network for accessibility services

**Optimization**:
- Adaptive frame rate based on network conditions
- Reduce resolution if bandwidth limited (480p sufficient)
- Graceful degradation: Lower FPS if network congested

### 7.6 Monitoring and Alerting

**Key Metrics**:
- Active sessions count
- Average latency per pipeline stage
- GPU utilization
- Error rate (failed recognitions)
- User satisfaction (feedback scores)

**Alerts**:
- High latency (>2 seconds): Investigate bottleneck
- GPU >90%: Scale up
- Error rate >10%: Check model or data quality
- Instance health check failure: Auto-restart

**Dashboard**:
- Real-time session count
- Latency graphs
- Recognition accuracy trends
- Cost tracking


## 8. Security Considerations

### 8.1 Secure Video Streaming

**WebSocket Security**:
- Use WSS (WebSocket Secure) protocol, not WS
- TLS 1.3 encryption for all video/audio streams
- Prevents man-in-the-middle attacks
- Certificate from Let's Encrypt (free)

**Video Privacy**:
- No permanent storage of video frames
- Frames processed in-memory and discarded immediately
- Only text transcriptions stored (if user opts in)
- Clear privacy policy displayed to users

**Session Isolation**:
- Each session has unique ID (UUID)
- Sessions cannot access each other's data
- In-memory data cleared on session end
- No cross-session data leakage

### 8.2 HTTPS for All APIs

**SSL/TLS Configuration**:
- HTTPS enforced for all REST API endpoints
- HTTP requests automatically redirected to HTTPS
- HSTS header to prevent downgrade attacks
- Certificate auto-renewal with Certbot

**API Gateway**:
- Terminates SSL at gateway level
- Backend communication within VPC (secure by default)
- Custom domain with valid SSL certificate

### 8.3 User Authentication

**JWT-based Authentication**:

**Registration/Login**:
```
POST /api/auth/register
{
  "email": "student@university.edu",
  "password": "hashed_password",
  "role": "student"  // or "faculty"
}

Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "expires_in": 3600
}
```

**Token Structure**:
- Access token: 1-hour expiration
- Refresh token: 7-day expiration
- Stored in HttpOnly cookies (not localStorage)
- Includes user_id, role, session_id

**Protected Endpoints**:
- All WebSocket connections require valid JWT
- Token validated on connection establishment
- Expired tokens rejected with 401 Unauthorized

**Password Security**:
- Bcrypt hashing with salt (cost factor: 12)
- Minimum password length: 8 characters
- No plain-text password storage

### 8.4 Rate Limiting

**API Rate Limits**:
- Per user: 100 requests/minute
- Per IP: 200 requests/minute
- WebSocket connections: 5 concurrent per user
- Prevents abuse and DoS attacks

**Implementation**:
- Token bucket algorithm
- Redis for distributed rate limiting
- 429 Too Many Requests response when exceeded

### 8.5 Input Validation

**Video Frame Validation**:
- Maximum frame size: 2 MB
- Allowed formats: JPEG, PNG only
- Validate base64 encoding
- Reject malformed data

**Audio Validation**:
- Maximum chunk size: 1 MB
- Allowed formats: WAV, MP3
- Sample rate: 16 kHz or 44.1 kHz
- Reject suspicious audio files

**Text Input Sanitization**:
- Escape HTML/JavaScript in user text
- Maximum text length: 1000 characters
- Prevent SQL injection (use parameterized queries)
- XSS protection

### 8.6 Access Control

**Role-based Access**:
- Student role: Can create sessions, use all features
- Faculty role: Same as student + view analytics
- Admin role: Manage users, view all sessions

**Session Access**:
- Only session participants can access session data
- Session creator can invite others via share link
- Time-limited share links (expire in 24 hours)

### 8.7 Data Privacy Compliance

**GDPR/Data Protection**:
- User consent for data collection
- Right to access personal data
- Right to delete account and data
- Data retention policy: 90 days for conversation history

**Campus Compliance**:
- Comply with university data policies
- Student data handled per FERPA guidelines (if US)
- No sharing of data with third parties
- Transparent privacy policy

### 8.8 Infrastructure Security

**EC2 Security**:
- Security groups: Whitelist only required ports
- SSH access: Key-based authentication only
- Disable password authentication
- Regular security updates (automated with unattended-upgrades)

**IAM Roles**:
- EC2 instance role for S3 access (no hardcoded credentials)
- Principle of least privilege
- Separate roles for different services
- Regular audit of permissions

**Secrets Management**:
- AWS Secrets Manager for API keys
- Environment variables for non-sensitive config
- No secrets in code or Docker images
- Rotate secrets every 90 days

**Logging and Monitoring**:
- CloudWatch Logs for all application logs
- Log failed authentication attempts
- Alert on suspicious activity
- Retain logs for 30 days


## 9. Estimated Prototype Cost (Hackathon Level Only)

### 9.1 Development Phase (2-3 months)

**Team Composition** (Student Team):
- 2 Full-stack Developers (Frontend + Backend)
- 1 ML Engineer (Model training and optimization)
- 1 UI/UX Designer (Accessibility-focused design)

**Cost**: $0 (student project, no salaries)

**Development Tools**:
- GitHub: Free for students
- VS Code: Free
- Postman: Free tier
- Figma: Free for students
- **Total**: $0

### 9.2 Cloud Infrastructure (Hackathon/Demo Phase)

**AWS Costs for 1 Week of Testing**:

**Compute**:
- EC2 g4dn.xlarge (GPU): $0.526/hr × 40 hours = $21
  - Run only during testing/demo (not 24/7)
- EC2 t3.medium (CPU): $0.0416/hr × 40 hours = $1.66
- **Subtotal**: $22.66

**Storage**:
- S3: 1 GB storage + 1000 requests = $0.05
- **Subtotal**: $0.05

**Networking**:
- API Gateway: 10,000 requests = $0.035
- Data Transfer: 5 GB = $0 (within free tier)
- **Subtotal**: $0.04

**Third-party APIs**:
- Google TTS: 10,000 characters = $0.16
- Google Translate (optional): 100,000 characters = $2
- **Subtotal**: $2.16

**Total Hackathon Cost**: ~$25

### 9.3 Model Training Costs

**GPU for Training**:
- Option 1: Use university GPU lab (Free)
- Option 2: Google Colab Pro ($10/month)
- Option 3: AWS EC2 g4dn.xlarge for 20 hours = $10.52

**Dataset Collection**:
- Use existing INCLUDE dataset (Free)
- Collect custom campus signs with volunteers (Free)

**Total Training Cost**: $0-10

### 9.4 Domain and SSL

**Domain Name**: $10-15/year (e.g., isl-bridge.tech)
**SSL Certificate**: Free (Let's Encrypt)

**Total**: $10-15 (one-time for hackathon)

### 9.5 Total Hackathon Budget

**Minimum Budget** (Using free resources):
- AWS: $25
- Domain: $10
- **Total: $35**

**Comfortable Budget** (With some paid tools):
- AWS: $50 (extra testing time)
- Model training: $10
- Domain: $15
- Buffer: $25
- **Total: $100**

### 9.6 Post-Hackathon Pilot (1 Month)

**If Deploying for Campus Pilot**:

**AWS (Running 12 hours/day, 5 days/week)**:
- EC2 GPU: $0.526/hr × 240 hrs = $126
- EC2 CPU: $0.0416/hr × 240 hrs = $10
- S3: $1
- API Gateway: $2
- Third-party APIs: $20
- **Total: ~$160/month**

**Cost Optimization**:
- Use Spot Instances: Save 70% → $50/month for GPU
- Run only during class hours: Already factored in
- Cache TTS results: Reduce API costs by 50%
- **Optimized: ~$80-100/month**

### 9.7 Funding Options

**For Students**:
- University innovation grants
- Hackathon prize money
- AWS Educate credits ($100-200 free credits)
- GitHub Student Developer Pack (includes cloud credits)
- Accessibility-focused grants and competitions

**Sustainability**:
- Partner with university disability services
- Seek CSR funding from tech companies
- Government accessibility grants
- Crowdfunding for social impact


## 10. Future Enhancements

### 10.1 Larger ISL Vocabulary

**Current Limitation**: 150 common campus signs

**Expansion Plan**:
- Phase 1: 500 signs (cover 90% of campus conversations)
- Phase 2: 1000+ signs (comprehensive vocabulary)
- Phase 3: Fingerspelling recognition (for names, technical terms)

**Data Collection Strategy**:
- Collaborate with deaf student associations
- Partner with ISL training institutes
- Crowdsource sign videos from community
- Continuous learning from user corrections

**Technical Approach**:
- Incremental model retraining
- Transfer learning from existing model
- Active learning: Prioritize signs users request most
- Model versioning for backward compatibility

**Timeline**: 6-12 months for 500 signs

### 10.2 Two-way Translation (Text-to-Sign Avatar)

**Vision**: Animated avatar that performs ISL signs

**Use Case**:
- Hearing person types text → Avatar signs it
- Enables asynchronous communication
- Useful for pre-recorded lectures, announcements

**Technical Approach**:
- 3D avatar with rigged hand and body model
- Motion capture data from ISL signers
- Text → Sign sequence mapping
- Smooth animation interpolation

**Challenges**:
- High-quality motion capture data needed
- Complex animation pipeline
- Facial expressions and body language important
- Computationally intensive rendering

**Alternative**: Video-based approach
- Pre-record videos of common signs
- Stitch videos together for sentences
- Lower quality but faster to implement

**Timeline**: 12-18 months (complex project)

### 10.3 Offline Campus Deployment

**Motivation**: Reduce latency and cloud costs

**Architecture**:
- Edge server on campus network
- All processing happens locally
- No internet dependency
- Lower latency (<500ms total)

**Hardware Requirements**:
- Server with NVIDIA GPU (e.g., RTX 3090)
- 32 GB RAM
- 1 TB SSD storage
- Cost: ~$3000-5000 one-time

**Benefits**:
- No monthly cloud costs
- Better privacy (data never leaves campus)
- Faster response times
- Works during internet outages

**Deployment Model**:
- Docker containers on campus server
- Accessible via campus WiFi
- IT department manages infrastructure
- Automatic updates via CI/CD

**Timeline**: 3-6 months after successful pilot

### 10.4 Mobile App

**Current**: Web-based (works on mobile browsers)

**Native App Benefits**:
- Better camera access and performance
- Offline capability with on-device models
- Push notifications for session invites
- Background audio processing

**Platforms**:
- iOS (Swift/SwiftUI)
- Android (Kotlin/Jetpack Compose)
- React Native for cross-platform (faster development)

**Features**:
- Optimized video streaming
- On-device hand detection (MediaPipe mobile)
- Lightweight LSTM model for offline recognition
- Sync with cloud for full vocabulary

**Timeline**: 4-6 months

### 10.5 Multi-language Support

**Current**: English and Hindi

**Expansion**:
- Regional languages: Tamil, Telugu, Bengali, Marathi
- Useful for regional campuses
- Translation between ISL and regional languages

**Technical Approach**:
- Integrate IndicTrans2 for better regional language support
- Language-specific TTS voices
- UI localization

**Timeline**: 2-3 months per language

### 10.6 Classroom Integration Features

**Lecture Recording with Captions**:
- Record lectures with automatic ISL/speech transcription
- Generate searchable transcripts
- Useful for all students, not just deaf students

**Presentation Mode**:
- Display captions on projector screen
- Large font for visibility from back of class
- Speaker identification for multi-speaker discussions

**Group Discussion Support**:
- Multi-user sessions (3-5 participants)
- Speaker diarization (who said what)
- Conversation threading

**Assignment Submission**:
- Students can submit video assignments in ISL
- Automatic transcription for faculty review
- Reduces barrier for deaf students

**Timeline**: 6-9 months

### 10.7 Analytics and Insights

**For Students**:
- Track communication patterns
- Identify frequently used signs
- Personalized vocabulary recommendations

**For Faculty**:
- Understand deaf student participation
- Identify communication barriers
- Improve teaching strategies

**For Administrators**:
- Measure accessibility program impact
- Usage statistics across campus
- ROI analysis for funding justification

**Privacy-preserving Analytics**:
- Aggregate data only, no individual tracking
- Opt-in for detailed analytics
- Anonymized data for research

**Timeline**: 3-4 months

### 10.8 Integration with Learning Management Systems

**LMS Integration** (Moodle, Canvas, Blackboard):
- Embed ISL assistant in LMS
- Accessible from course pages
- Integrate with assignment submissions
- Single sign-on with university credentials

**Benefits**:
- Seamless user experience
- No separate login required
- Centralized accessibility tools

**Timeline**: 2-3 months per LMS

### 10.9 Research and Continuous Improvement

**Model Improvement**:
- Collect user feedback on recognition accuracy
- Retrain models with corrected data
- A/B test new model versions
- Publish research papers on ISL recognition

**User Studies**:
- Conduct usability studies with deaf students
- Measure impact on academic performance
- Gather qualitative feedback
- Iterate on UX design

**Open Source Contribution**:
- Release ISL dataset for research community
- Open-source core recognition models
- Collaborate with other universities
- Build ecosystem of ISL tools

**Timeline**: Ongoing

---

**Document Version**: 1.0  
**Last Updated**: February 11, 2026  
**Project**: ISL Communication Bridge for Indian Campuses  
**Status**: Hackathon Design Document  
**Target**: Realistic prototype with clear path to production
