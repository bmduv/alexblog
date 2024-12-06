---
title: System Design Template
draft: false
tags:
---
# Overview:
1. Clarifying requirements
2. Framing the problem as an ML task
3. Data preparation
4. Model development
5. Evaluation
6. Deployment and serving
7. Monitoring and infrastructure
![[public/posts/imgs/ch1-02-5AHN2LDX.webp]]
## Clarifying Requirements
- Business Objective
- Features the system needs to support
- Data: what are the data sources? How large is the datatset? Is the data labeled?
- Constraints: How much computing power available? Is is cloud-based or on-device?
- Scale of the system: How many users do we have?
- Performance: How fast must prediction be? Does accuracy have more priority or latency?
## Framing the problem as an ML task
- Define the ML objective
   - I.e. Application: Ad Click Prediction System + Business Objective: Increase User Clicks + ML Objective: Maximize Click-through Rate
- Specify system's input and output
- Choose right ML Category
   - Supervised Learning
   - Unsupervised Learning
   - Reinforcement Learning
   - Classification Model
## Data preparation
- Data Sources: Who collected it? How clean is data?
- Data Storage
![[public/posts/imgs/ch1-07-IWLEAUDN.webp]]
- Extract, Transform, and Load (ETL):
   - Extract: Extracts data from different sources
   - Transform: Data is cleansed, mapped, and transformed into specific format
   - Load: Transformed data is loaded into the target destination (file, database, data warehouse)
![[public/posts/imgs/ch1-08-YK5FEBTW.webp]]
- Data Types:
   - Unstructured Data: No schema + difficult to search
      - Resides in NoSQL databases / Data lakes
      - i.e. Text files, Audio files, Images, Video
- Feature Engineering: Deletion, imputation (filling in missing values with mean, median...)
   - Feature scaling: normalization, standardization, log scaling (can make data distribution less skewed to enable faster convergence)
- Discretization (bucketing): Convert continuous feature into categorical feature
- Encoding Categorical Features: Integer encoding, one-hot encoding, embedding learning
## Model Development
- Model Selection: 
   - Establish a simple baseline
   - Experiment with simple models: After having baseline, explore models that are quick to train
   - Switch to to more complex models: If simple models cannot perform well, consider more complex models
- Model Training:
   - Construct the dataset:
      - Collect the raw data
      - Identify features and labels
      - Select sampling strategy: convenience sampling, snowball sampling, stratified sampling, reservoir sampling, importance sampling
      - Split the data: training, evaluation (validation), and test
      - Address any class imbalances: resampling, altering loss function (more weight to data points from minority class)
   - Distributed training:
      - Data Parallelism
      - Model Parallelism
## Evaluation
- Offline Evaluation;
   - Classification: Precision, recall, F1 Score, accuracy, ROC-AUC, confusion matrix
- Online Evaluation:
   - i.e. for Ad Click Prediction: click-through rate, revenue lift
## Deployment and Serving
- Cloud vs. On-device Deployment
- Model Compressions: 
   - Knowledge distillation: train small model (student) to mimic larger model (teacher)
   - Pruning: finding least useful parameters and setting them to zero (sparser model for efficient storage)
   - Quantization: use fewer bits (i.e. 32 bits to 16 bits) to represent parameters, reducing model size.
## Test in Production
- Shadow deployment: deploy new model in parallel with existing mode (incoming request directed to both models but only existing model's prediction is served)
- A/B testing: deploy new model in parallel with existing model -> portion of traffic is routed to new model (traffic routed has to be random)
## Prediction Pipeline
- Batch prediction
- Online prediction
## Monitoring
- Data distribution constantly changes
- Train on large datasets
- Regularly retrain model

--------------------------------------------------------------------
### 1. **Problem Definition:**
- **Objective:** Start by defining the problem you're solving. For instance, "Develop a real-time gesture recognition system using sEMG data for a wearable device."
- **Requirements:** List functional and non-functional requirements. For example:
    - High accuracy in gesture recognition.
    - Low latency for real-time performance.
    - Robust across different users and sessions.
    - Efficient enough to run on a wearable device.
### 2. **Data Preparation:**
- **Data Sources:** Show how data is collected. Use arrows to indicate data flow from sensors to storage. (IMPORTANT: PRIVACY such as de-identification (hashing) during collection)
- **Preprocessing:** Highlight preprocessing steps such as filtering (e.g., high-pass filter at 20 Hz), normalization, and feature extraction (e.g., covariance matrix, MUAP detection).
   - Time-domain Features:
	- Root Mean Square: computationally efficient and quick
	- Variance
	- Mean Absolute Value: (method of detecting and gauging muscle contraction levels)
	- Simple Square Integral:  expresses the energy of the EMG signal as a useable feature
	- Slope Sign Change
	- Zero Crossing
	- Waveform Length: cumulative length of the waveform over the segment
   - Frequency Features:
      - Frequency Median
      - FFT
  - Time-domain:
     - Spectrogram
 - **Segmentation and Labeling**:
    - **Start and End Points**: Precisely label the start and end points of each gesture within the continuous data. This is crucial for accurate training.
    - **Sliding Window Labeling**: Assign a label to each window of EMG data based on the gesture occurring within that window. If a window contains a transition between gestures, it might be labeled as the previous gesture, the next gesture, or discarded, depending on the application and the model’s capability to handle transitions.
    - **Transition Windows**: Consider adding a transition class to handle windows where no clear gesture is being made, or where the signal is transitioning from one gesture to another.
- **Class Imbalance**: In continuous data, some gestures may be performed more frequently than others. Ensure that the dataset is balanced or apply techniques like oversampling or data augmentation to address class imbalance.
- **Continuous/Rest Labels**: Include a "rest" or "no gesture" class to capture periods where no specific gesture is performed. This helps the model distinguish between intentional gestures and the absence of gestures.
-  **Data Pipeline:** Indicate steps like data ingestion, real-time processing, and storage (e.g., using a message queue for streaming data).
#### Multivariate Power Frequency Feature:
- 40 Hz High-Pass Filter: 
   - The first step involves filtering the raw sEMG data with a 40 Hz high-pass filter -> removes low-frequency components below 40 Hz that could include noise/artifacts
- Cross Spectral Density Calculation:
   - Rolling Window: CSD calculated over rolling window of 160 sEMG samples, with window moving forward by 40 samples (stride of 40) -> ensures analysis captures temporal dynamics of signal
      - Discrete Fourier Transform with 64 Samples and Stride of 10: converts time-domain sEMG signals to frequency domain.
      - Binning into Frequency Bins: 6 frequency bins
      - Output: 6 symmetric and positive definite 16x16 matrices (one for each frequency bin) updated every 40 samples
   - Riemannian Tangent Space Mapping: each 16x16 matrices projected into respective RTS
   - Feature Vector Construction: Off-diagonals -> first three off-diagonals preserved  (variance/relationship between different channels). Half-vectorization -> flattened into vector and only taking half of the symmetric matrix (other half is redundant)
   - Final Result: 384-dimensional vector (6 frequency bins * 3 off-diagonals * 16 elements per diagonal) for each 80 ms window of sEMG data
#### Discrete Gesture:
- **Discrete Gesture Time Alignment:** Initially, approximated timing of gesture execution, worked pretty well when inter-gesture interval was greater than uncertainty of timing, BUT, rapid sequences makes it difficult for timing of individual gestures
   - Infer timing of all gestures in a sequence -> use generative model of MPF features to search for sequence of gesture timings (waveforms) that best explain observed data
   - i.e. for a sequence of three gestures (e.g., fist, open hand, thumbs-up), the model might start with an initial guess of when each gesture occurred based on the prompts. The templates for these gestures are then shifted in time to find the best match with the observed data.
#### Handwriting:
- Data Augmentation: SpecAugment(Time warping, frequency masking, time masking) + rotational augmentation (randomly rotating all channels by either -1, 0, +1)
### 3. **Model Development:**
#### Wrist Pose:
- Input: MPF 
- Architecture: CNN (dropout) + LayerNorm + FullyConnectedLayer_Relu + CNN (droput) + LayerNorm + FullyConnectedLayer_Relu
- Optimization: AdamW optimizer + cosine annealing learning rate scheduler + early stopping
#### Discrete Gestures:
- Input: High-pass filtered (40 Hz) sEMG signal
- Model Architecture: CNN (stride 10) + LayerNorm + 3 LSTM + LayerNorm + LinearReadoutLayer (for nine classes)
- Optimization: AdamW optimizer + gradient clipping (mitigate divergence)
   - Binary Cross-Entropy Loss
#### Handwriting:
- Input: MPF along with data augmentation methods (SpecAugment + rotational augmentation)
- Model Architecture: FullyConnectedLayer_LeakyRelu (rotational-invariance) + conformer_15Layers (4 attention heads + time-convolutional kernel size 8 stride 1 with stride 2 at layers 5 + 10)  + channel_average_pooling + LinearLayer + Softmax
   - Modified conformer to ensure causality: self-attention applied to fixed local window directly before current time step (16 window size for initial 10 conformer layers and 8 for last 5).
- Optimization: AdamW optimzier + cosine annealing learning rate schedule + gradient clipping
   - CTC Loss
   - Character Error Rate Sum(Levenshtein_distance_i) / Sum(Prompt_length_i)

- **Feature Extraction Layer:** Include components like convolutional layers to process raw sEMG signals and extract relevant features.
- **Model Layers:** Show the different layers of the model:
    - **CNN for Initial Feature Extraction:** 1D Convolutional layers to process sEMG signals.
    - **LSTM Layers:** For capturing temporal dependencies in the sequence of muscle activations.
    - **Fully Connected Layers:** To classify gestures based on extracted features.
- **Output Layer:** Softmax layer for gesture classification.

 **Compression Techniques:**
    - **Pruning:** Show a step where unnecessary parameters are removed.
    - **Quantization:** Illustrate the process of reducing the precision of weights and activations (e.g., 32-bit to 8-bit).
    - **Distillation:** If applicable, indicate how a smaller student model learns from a larger teacher model.
### 4. **Evaluation:**
- **Performance Metrics:** Include metrics like accuracy, latency, energy efficiency, and robustness.
- **Monitoring System:** Display how the system monitors performance in real-time and logs any errors or anomalies.
- **Closed-Loop System:**
   - **Feedback Mechanism:** Indicate how the system can adapt its parameters in real-time based on feedback (e.g., adjusting model weights on-device).
   - **User Interaction:** If relevant, show how user interactions are fed back into the system for continuous learning or adaptation.
#### Wrist Pose:
- Acquisition Time
- Dial-in Time: time from first target to last target entry
#### Discrete Gestures:
- Completion Rate: time required to complete trial divided by minimum number of discrete gestures required to complete trial
- Confusion Matrix: number of times each gesture detected when given gesture was expected divided by total number of gestures executed when that given gesture was expected
- First Hit Probability: proportion of prompted gestures in which first executed gesture was expected
#### Handwriting:
- Character Error Rate
- Adjusted Word per Minute
### 6. **Inference Pipeline:**
- **Real-Time Inference:** Show how data from the wearable device is fed into the deployed model.
- **Feedback Loop:** If applicable, indicate a feedback loop where the system adapts based on real-time performance metrics.
### 9. **Scalability Considerations:**
- **Scaling Data Collection:** Illustrate how the system can scale to collect data from multiple users.
- **Distributed Training:** If applicable, show how model training can be distributed across multiple GPUs or cloud instances.
### 10. **Future Enhancements:**
- **Personalization:** Indicate plans for model personalization based on user-specific data.
   - FEDERATE LEARNING
   - Self-Supervised Learning (Masking signal and predicting masked signal)
- **Integration:** Show potential integration with other systems (e.g., mobile apps, cloud services).

EMG time series (high throughput) -> generalize

- Look at componets of system to put -> focus on how you build the infrastructure -> how to setup study to collect data using custom devices -> ensure studies generalize to users -> how to get quality data for werable emg device -> build end-to-end system for a decoder that can detect 2 actions -> deadline is imminnet and we need to act fast on data collection, prototyping, and evaluation -> best way to collect data