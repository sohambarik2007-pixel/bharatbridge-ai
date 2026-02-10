# BharatBridge AI - AI System Architecture

## 1. System Overview

BharatBridge AI employs a multi-stage AI pipeline that transforms visual document data into accessible, multilingual audio content. The system orchestrates five core AI components in sequence:

1. **OCR Engine**: Extracts text from document images with support for multiple Indian scripts
2. **Language Detector**: Identifies the source language from extracted text
3. **Neural Translation Model**: Converts text between Indian languages while preserving context
4. **Content Simplifier**: Reduces complexity of bureaucratic and technical language using LLM reasoning
5. **Text-to-Speech Synthesizer**: Generates natural-sounding audio in regional languages

The architecture is designed for modularity, allowing independent optimization of each component while maintaining seamless data flow between stages.

## 2. End-to-End Data Flow

- **Output**: Preprocessed image tensor ready for OCR

### Stage 2: OCR Processing
- **Image Enhancement**: Adaptive thresholding, noise reduction, deskewing
- **Script Detection**: Identify Devanagari, Tamil, Bengali, Telugu, and other Indic scripts
- **Text Extraction**: Character-level recognition with bounding box coordinates
- **Post-OCR Correction**: Dictionary-based spell checking for common OCR errors
- **Output**: Raw extracted text with confidence scores per word


### Stage 3: Language Detection
- **Input**: Extracted text string (minimum 10 characters for reliable detection)
- **Feature Extraction**: Character n-grams, script patterns, Unicode range analysis
- **Classification**: Multi-class prediction across 20+ Indian languages
- **Confidence Thresholding**: Flag ambiguous cases for user confirmation
- **Output**: ISO 639 language code with confidence score (e.g., "hi" for Hindi, 0.98 confidence)

### Stage 4: Translation Pipeline
- **Tokenization**: Language-specific tokenizers for source text
- **Contextual Encoding**: Transformer-based encoder captures semantic meaning
- **Cross-lingual Mapping**: Attention mechanism aligns source and target language representations
- **Decoding**: Auto-regressive generation of target language text
- **Post-processing**: Grammar correction, punctuation normalization
- **Output**: Translated text in target regional language

### Stage 5: Text Simplification Module
- **Complexity Analysis**: Identify technical terms, long sentences, passive voice
- **LLM Prompting**: Few-shot prompts with examples of simplified government/medical text
- **Terminology Extraction**: Build glossary of domain-specific terms with simple definitions
- **Sentence Restructuring**: Break compound sentences, convert passive to active voice
- **Readability Scoring**: Ensure output meets 6th-grade reading level
- **Output**: Simplified text + glossary of technical terms

### Stage 6: Text-to-Speech Generation
- **Text Normalization**: Expand abbreviations, handle numbers and dates
- **Phoneme Conversion**: Grapheme-to-phoneme mapping for target language
- **Prosody Modeling**: Determine pitch, duration, and stress patterns
- **Waveform Synthesis**: Neural vocoder generates audio waveform
- **Audio Post-processing**: Normalize volume, apply compression
- **Output**: MP3/WAV audio file with natural-sounding speech

### Stage 7: Final Output to User
- **Display**: Show original text, translated text, and simplified version side-by-side
- **Audio Player**: Embedded player with playback controls
- **Confidence Indicators**: Visual cues for low-confidence translations
- **Feedback Loop**: User corrections fed back to improve models
- **Export Options**: Save as PDF, share audio file, copy text


## 3. Model Selection & Justification

### OCR Model: EasyOCR + Tesseract Hybrid
**Primary Choice**: EasyOCR
- **Rationale**: Native support for 15+ Indian languages including Hindi, Tamil, Telugu, Kannada, Bengali
- **Advantages**: Pre-trained on diverse fonts, handles low-quality images, open-source
- **Performance**: 92-95% accuracy on printed Indic text
- **Deployment**: Lightweight models (50-100MB per language) suitable for mobile

**Fallback**: Tesseract 5.0 with Indic language packs
- **Use Case**: When EasyOCR confidence < 0.7
- **Advantages**: Mature OCR engine, extensive language support, configurable
- **Integration**: Ensemble approach - combine outputs when both models disagree

**Cloud Alternative**: AWS Textract (for high-value documents)
- **Use Case**: Legal documents, certificates requiring highest accuracy
- **Advantages**: 98%+ accuracy, table extraction, form field detection
- **Trade-off**: Cost per API call, requires internet connectivity

### Language Detection Model: FastText + LangDetect
**Primary Choice**: FastText Language Identification
- **Rationale**: Trained on 176 languages including all major Indian languages
- **Advantages**: Fast inference (<10ms), works on short text (10+ characters), 99.1% accuracy
- **Model Size**: 126MB compressed, suitable for on-device deployment
- **Fallback**: LangDetect library for ambiguous cases

**Script-based Pre-filtering**:
- Unicode range analysis to narrow down language candidates
- Devanagari script → Hindi, Marathi, Sanskrit, Nepali
- Tamil script → Tamil only
- Reduces false positives and speeds up detection

### Translation Model: IndicTrans2 + mBART50
**Primary Choice**: IndicTrans2 (AI4Bharat)
- **Rationale**: Specifically designed for Indian languages, state-of-the-art performance
- **Coverage**: 22 scheduled Indian languages + English
- **Architecture**: Transformer-based (600M parameters), distilled version available (200M)
- **Performance**: BLEU scores 30-45 across language pairs, outperforms Google Translate for Indic languages
- **Training Data**: 230M sentence pairs from government documents, news, Wikipedia

**Fallback**: mBART50 (Meta AI)
- **Use Case**: When IndicTrans2 doesn't support language pair
- **Advantages**: Multilingual pre-training, 50 languages including 10 Indian languages
- **Performance**: BLEU scores 25-35 for Indic languages

**Domain Adaptation Strategy**:
- Fine-tune on domain-specific corpora (government forms, medical documents)
- Build custom dictionaries for technical terms (medical, legal, financial)
- Implement terminology consistency across document


### Text Simplification: LLM-based with Rule-based Fallback
**Primary Approach**: GPT-4 / Claude API with Custom Prompts
- **Rationale**: Superior understanding of context, handles domain-specific simplification
- **Prompt Engineering**: Few-shot examples of government/medical text simplification
- **Advantages**: Maintains meaning while reducing complexity, explains jargon
- **Limitations**: API costs, latency (2-4 seconds), requires internet

**Fallback**: Rule-based Simplification Engine
- **Components**:
  - Sentence splitter: Break sentences >25 words
  - Passive-to-active converter: Regex-based transformation
  - Jargon detector: Dictionary of 5000+ complex terms with simple alternatives
  - Readability scorer: Flesch-Kincaid adapted for Indian languages
- **Advantages**: Fast (<100ms), works offline, predictable
- **Limitations**: Less contextual, may miss nuanced simplifications

**Hybrid Strategy**:
- Use LLM for first-time document types, cache simplification patterns
- Apply rule-based for common document types (Aadhaar, PAN, prescriptions)
- User feedback loop to improve rule database

### Text-to-Speech Engine: gTTS + Coqui TTS
**Primary Choice**: Google Text-to-Speech (gTTS)
- **Rationale**: Supports 15+ Indian languages with natural-sounding voices
- **Advantages**: High-quality neural voices, free tier available, simple API
- **Performance**: 1-2 seconds latency for 100-word text
- **Limitations**: Requires internet, limited voice customization

**On-device Alternative**: Coqui TTS (Open-source)
- **Use Case**: Offline mode, privacy-sensitive documents
- **Advantages**: Fully local, customizable voices, no API costs
- **Model Size**: 50-100MB per language
- **Trade-off**: Slightly lower voice quality than cloud TTS

**Premium Option**: Amazon Polly
- **Use Case**: Enterprise deployment, high-volume usage
- **Advantages**: Neural voices, SSML support for prosody control, 10+ Indian languages
- **Features**: Breathing sounds, whispering, emphasis control
- **Cost**: Pay-per-character pricing

**Voice Selection Strategy**:
- Default to female voices (research shows higher trust for informational content)
- Allow user preference for voice gender and speed
- Cache audio for frequently accessed documents


## 4. Data Processing Strategy

### Handling Noisy Images
**Challenge**: Real-world documents often have poor lighting, shadows, wrinkles, or low resolution

**Preprocessing Pipeline**:
1. **Contrast Enhancement**: Adaptive histogram equalization (CLAHE) to improve text visibility
2. **Noise Reduction**: Bilateral filtering to remove noise while preserving edges
3. **Binarization**: Otsu's thresholding for converting to black-and-white
4. **Deskewing**: Hough transform to detect and correct rotation (±15 degrees)
5. **Border Removal**: Crop unnecessary margins to focus on content area

**Quality Assessment**:
- Calculate image sharpness score using Laplacian variance
- If score < threshold, prompt user to retake photo with guidance
- Provide real-time feedback during camera capture (lighting, focus, angle)

**Augmentation for Robustness**:
- Train OCR models on augmented data (blur, noise, rotation, shadows)
- Use ensemble of models trained on different augmentation strategies
- Implement confidence-based retry mechanism

### Preprocessing Steps
**Image Normalization**:
- Resize to standard dimensions (1024x1024 or maintain aspect ratio)
- Convert color space (RGB → Grayscale for OCR)
- Normalize pixel values (0-255 → 0-1 range)

**Text Region Detection**:
- Use EAST (Efficient and Accurate Scene Text) detector to locate text regions
- Filter out non-text areas (logos, images, decorative elements)
- Prioritize regions based on size and position (headers, body text)

**Layout Analysis**:
- Detect document structure (headers, paragraphs, tables, lists)
- Preserve reading order for multi-column layouts
- Handle mixed-language documents (English headers, regional language body)

### Handling Multiple Indian Scripts
**Script Identification**:
- Unicode range detection:
  - Devanagari: U+0900 to U+097F
  - Bengali: U+0980 to U+09FF
  - Tamil: U+0B80 to U+0BFF
  - Telugu: U+0C00 to U+0C7F
  - Kannada: U+0C80 to U+0CFF
  - Malayalam: U+0D00 to U+0D7F
  - Gujarati: U+0A80 to U+0AFF
  - Punjabi (Gurmukhi): U+0A00 to U+0A7F

**Script-specific OCR Models**:
- Load appropriate language model based on detected script
- Use script-specific preprocessing (e.g., Tamil requires different binarization)
- Handle conjunct characters and ligatures common in Indic scripts

**Mixed-script Documents**:
- Segment document into script-specific regions
- Process each region with appropriate OCR model
- Merge results while preserving spatial layout


### Error Handling
**OCR Failures**:
- **Low Confidence Detection**: Flag words with confidence < 0.6, highlight in UI
- **Fallback Strategy**: Switch to alternative OCR engine (Tesseract if EasyOCR fails)
- **User Correction**: Allow manual text editing with inline correction interface
- **Retry Mechanism**: Suggest image retake with specific guidance (better lighting, closer distance)

**Language Detection Errors**:
- **Ambiguity Handling**: If top 2 languages have confidence difference < 0.1, show both options to user
- **Manual Override**: Always allow user to manually select source language
- **Context Clues**: Use document type hints (Aadhaar → Hindi/English, Ration Card → Regional language)

**Translation Failures**:
- **Unsupported Language Pairs**: Pivot through English (Source → English → Target)
- **API Timeouts**: Implement exponential backoff, cache partial results
- **Fallback Translation**: Use simpler word-by-word translation if neural model fails
- **Quality Checks**: Detect untranslated text (same as source), flag for review

**TTS Failures**:
- **Network Issues**: Queue audio generation, process when connectivity restored
- **Unsupported Characters**: Replace special characters with phonetic equivalents
- **Fallback**: Provide text-only output if TTS unavailable
- **Caching**: Store generated audio to avoid regeneration

**Graceful Degradation**:
- Partial results better than complete failure
- Show progress indicators for each pipeline stage
- Allow users to proceed with imperfect results
- Log errors for continuous improvement


## 5. Architecture Flow Diagram Explanation

### Component Layout (Top to Bottom)
The architecture follows a linear pipeline with feedback loops:

**Layer 1: Input Layer**
- Three input sources: Camera Module, Gallery Upload, PDF Import
- All converge into Image Preprocessing Unit
- Preprocessing includes: Quality Check → Enhancement → Normalization → Format Conversion

**Layer 2: OCR Layer**
- Image Preprocessing feeds into Script Detector
- Script Detector routes to appropriate OCR Engine (EasyOCR or Tesseract)
- OCR Engine outputs: Extracted Text + Confidence Scores + Bounding Boxes
- Low confidence triggers User Correction Interface (feedback loop to Layer 1)

**Layer 3: Language Processing Layer**
- Extracted Text flows into Language Detection Module
- FastText classifier identifies source language
- Parallel path: Script Analysis provides additional confidence
- Output: Language Code + Confidence Score
- Ambiguous results trigger User Language Selection (feedback loop)

**Layer 4: Translation Layer**
- Source Text + Language Code enter Translation Router
- Router selects model: IndicTrans2 (primary) or mBART50 (fallback)
- Translation Model processes with attention mechanism
- Post-processor applies grammar correction and formatting
- Output: Translated Text in Target Language

**Layer 5: Simplification Layer**
- Translated Text enters Complexity Analyzer
- Analyzer identifies: Technical Terms, Long Sentences, Passive Voice
- Routing decision: LLM API (online) or Rule Engine (offline)
- LLM Path: GPT-4 with few-shot prompts → Simplified Text + Explanations
- Rule Path: Sentence Splitter → Jargon Replacer → Readability Scorer
- Output: Simplified Text + Glossary

**Layer 6: Audio Generation Layer**
- Simplified Text enters Text Normalizer (expand abbreviations, handle numbers)
- Phoneme Converter maps text to pronunciation
- TTS Engine (gTTS or Coqui) generates audio waveform
- Audio Post-processor normalizes and compresses
- Output: Audio File (MP3/WAV)

**Layer 7: Output Layer**
- Presentation Module displays three panels:
  - Original Text (with OCR confidence highlights)
  - Translated & Simplified Text (with glossary tooltips)
  - Audio Player (with speed controls)
- Export Module enables: Save PDF, Share Audio, Copy Text
- Feedback Module collects user corrections → feeds back to Model Training Pipeline

**Cross-cutting Components**:
- **Cache Layer**: Sits between all layers, stores intermediate results
- **Error Handler**: Monitors all components, triggers fallbacks
- **Analytics Module**: Logs performance metrics, user interactions
- **Model Registry**: Manages model versions, enables A/B testing

**Data Flow Arrows**:
- Solid arrows: Primary data flow
- Dashed arrows: Fallback/alternative paths
- Dotted arrows: Feedback loops for user corrections
- Bidirectional arrows: Cache read/write operations


## 6. Future AI Enhancements

### Offline Model Optimization
**Challenge**: Current pipeline relies on cloud APIs (translation, TTS, simplification)

**Optimization Strategies**:
1. **Model Quantization**: Convert FP32 models to INT8, reducing size by 75% with <2% accuracy loss
2. **Knowledge Distillation**: Train smaller student models (50M params) from larger teachers (600M params)
3. **Pruning**: Remove redundant neurons, achieve 40-60% sparsity without performance degradation
4. **Mobile-optimized Architectures**: Deploy TensorFlow Lite or ONNX Runtime for on-device inference

**Target Specifications**:
- OCR: 30MB per language, <500ms inference on mid-range phones
- Translation: 100MB distilled IndicTrans2, <1s for 100-word text
- TTS: 50MB Coqui model, <2s audio generation
- Total app size: <500MB with 5 language packs

**Hybrid Approach**:
- Offline models for common use cases (90% of documents)
- Cloud fallback for complex documents or rare language pairs
- Progressive model download based on user's language preferences

### Handwritten Text Recognition
**Current Limitation**: OCR optimized for printed text, fails on handwritten documents

**Proposed Solution**:
1. **HTR Model Integration**: Deploy IAM Handwriting Database-trained models
2. **Indic Script HTR**: Fine-tune on IIIT-HWS dataset (Hindi, Telugu, Tamil handwriting)
3. **Writer-independent Recognition**: Handle diverse handwriting styles
4. **Confidence-based Routing**: Detect handwritten vs printed, route to appropriate model

**Technical Approach**:
- CNN-RNN-CTC architecture for sequence recognition
- Attention mechanism for handling cursive writing
- Data augmentation: Synthetic handwriting generation
- User feedback loop: Collect corrections to improve model

**Expected Performance**:
- Character accuracy: 85-90% for clear handwriting
- Word accuracy: 75-80% with language model correction
- Processing time: 2-3x slower than printed text OCR


### Dialect Adaptation
**Challenge**: Standard language models don't capture regional dialects and colloquialisms

**Dialect Recognition**:
- Extend language detection to identify dialects (e.g., Awadhi vs Standard Hindi, Hyderabadi vs Telugu)
- Use phonetic features and vocabulary markers for classification
- Build dialect taxonomy for major Indian languages

**Dialect-aware Translation**:
- Fine-tune translation models on dialect-specific corpora
- Implement dialect normalization layer (dialect → standard → target language)
- Preserve cultural context and idiomatic expressions

**Data Collection Strategy**:
- Crowdsource dialect samples from regional communities
- Partner with local NGOs for authentic dialect data
- Use speech-to-text to capture spoken dialects

**TTS Dialect Support**:
- Train voice models on dialect-specific speech data
- Capture regional pronunciation patterns and intonation
- Allow users to select dialect preference for audio output

### Personalization
**User Profile Learning**:
- Track frequently translated document types (medical, banking, government)
- Learn user's vocabulary level and preferred simplification style
- Remember language pair preferences and TTS settings

**Adaptive Simplification**:
- Adjust complexity based on user's comprehension feedback
- Build personal glossary of terms user has learned
- Reduce explanations for familiar concepts over time

**Context-aware Processing**:
- Use document history to improve translation consistency
- Maintain terminology database across user's documents
- Suggest related documents or information based on content

**Smart Recommendations**:
- Predict document type from image preview
- Pre-select likely source/target languages
- Suggest relevant government schemes or actions based on document content

**Privacy-preserving Personalization**:
- On-device learning without sending personal data to cloud
- Federated learning to improve models while preserving privacy
- User control over data retention and model personalization

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026  
**Project**: BharatBridge AI - AI System Architecture  
**Focus**: AI Pipeline Design and Model Selection
# BharatBridge AI - Cloud Infrastructure and Technology Stack

## 1. Backend Architecture

### Framework Selection: FastAPI
- **Rationale**: Async support for handling multiple OCR/translation requests, automatic API documentation, built-in validation
- **Alternative**: Flask (simpler but synchronous)

### API Design
```
POST /api/v1/extract-text        # OCR from image
POST /api/v1/detect-language     # Language detection
POST /api/v1/translate           # Translation service
POST /api/v1/simplify            # Content simplification
POST /api/v1/text-to-speech      # TTS generation
POST /api/v1/process-complete    # End-to-end pipeline
```

### Architecture Approach: Monolithic with Modular Design
- **For Hackathon**: Monolithic (faster development, easier deployment)
- **Production Path**: Microservices (independent scaling of OCR, translation, TTS services)

## 2. Cloud Infrastructure (AWS)

### Storage
- **Amazon S3**
  - Bucket 1: `bharatbridge-uploads` (user images, TTL: 24 hours)
  - Bucket 2: `bharatbridge-audio` (generated TTS files, TTL: 1 hour)
  - Lifecycle policies for automatic cleanup

### Compute
- **AWS Lambda** (Recommended for hackathon)
  - Serverless, pay-per-use
  - 15-minute timeout sufficient for processing
  - Memory: 2GB-3GB for OCR operations
- **Alternative: EC2 t3.medium** (if Lambda limits are restrictive)

### API Management
- **AWS API Gateway**
  - REST API endpoints
  - Request throttling (1000 req/sec)
  - CORS configuration
  - API key management

### Database
- **Amazon DynamoDB**
  - Store request metadata, usage analytics
  - Schema: `{user_id, request_id, timestamp, source_lang, target_lang, status}`
  - Optional for MVP

### Hosting Strategy
- **Frontend**: AWS Amplify or S3 + CloudFront
- **Backend**: Lambda + API Gateway or EC2 with Application Load Balancer

### CDN
- **Amazon CloudFront**
  - Cache static assets
  - Reduce latency for global users
  - Edge locations across India

## 3. Scalability Plan

### Auto-Scaling
- **Lambda**: Automatic (up to 1000 concurrent executions)
- **EC2**: Auto Scaling Group (min: 1, max: 5 instances)
  - Scale-up trigger: CPU > 70%
  - Scale-down trigger: CPU < 30%

### Load Balancing
- **Application Load Balancer** (if using EC2)
  - Health checks every 30 seconds
  - Distribute traffic across availability zones

### Concurrent Users
- **Target**: 100-500 concurrent users for hackathon demo
- **Lambda concurrency**: Reserved 100 concurrent executions
- **API Gateway**: 10,000 requests per second burst capacity

## 4. Security Considerations

### HTTPS
- SSL/TLS certificates via AWS Certificate Manager (free)
- Enforce HTTPS-only on API Gateway and CloudFront

### IAM Roles
- Lambda execution role with minimal permissions:
  - S3: PutObject, GetObject (specific buckets only)
  - DynamoDB: PutItem, GetItem
  - CloudWatch: Logs write access
- No hardcoded credentials

### Data Encryption
- **At Rest**: S3 server-side encryption (SSE-S3)
- **In Transit**: TLS 1.2+
- **Sensitive Data**: Encrypt user metadata in DynamoDB

### Secure File Handling
- File size limits: 10MB max
- Allowed formats: JPG, PNG, PDF only
- Virus scanning (optional): AWS Lambda with ClamAV
- Signed URLs for S3 uploads (expiry: 5 minutes)

## 5. Deployment Strategy

### CI/CD Pipeline
```
GitHub → GitHub Actions → AWS Lambda/EC2
```

**Workflow**:
1. Push to `main` branch
2. Run tests (pytest)
3. Build Docker image (optional)
4. Deploy to AWS using AWS CLI/CDK
5. Run smoke tests

### Containerization
- **Docker** (optional for hackathon, recommended for production)
  - Base image: `python:3.11-slim`
  - Multi-stage builds for smaller images
  - ECR for container registry

### Version Control
- **GitHub Repository Structure**:
```
/backend          # FastAPI application
/frontend         # React/Next.js
/infrastructure   # AWS CDK/Terraform scripts
/tests            # Unit and integration tests
/.github/workflows # CI/CD pipelines
```

## 6. Estimated Implementation Cost

### AWS Free Tier (First 12 months)
- Lambda: 1M requests/month free
- S3: 5GB storage free
- API Gateway: 1M requests/month free
- DynamoDB: 25GB storage free

### Hackathon Prototype (Beyond Free Tier)
| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Lambda | 10,000 requests | $0.20 |
| S3 | 10GB storage + requests | $0.50 |
| API Gateway | 10,000 requests | $0.04 |
| CloudFront | 10GB data transfer | $0.85 |
| EC2 (if used) | t3.medium (100 hours) | $4.00 |
| **Total** | | **~$5-6/month** |

### Production Scale (1000 daily users)
- Estimated: $50-100/month
- Includes: Lambda scaling, increased S3 storage, CloudFront bandwidth

### Cost Optimization Tips
- Use Lambda for variable workloads
- Enable S3 lifecycle policies
- Implement request caching
- Monitor with AWS Cost Explorer

---

## Technology Stack Summary

**Backend**: FastAPI + Python 3.11  
**OCR**: Tesseract / AWS Textract  
**Translation**: Google Translate API / AWS Translate  
**TTS**: gTTS / AWS Polly  
**Cloud**: AWS (Lambda, S3, API Gateway, CloudFront)  
**Database**: DynamoDB (optional)  
**CI/CD**: GitHub Actions  
**Monitoring**: AWS CloudWatch  

---

*This infrastructure is designed for rapid hackathon deployment with a clear path to production scaling.*
