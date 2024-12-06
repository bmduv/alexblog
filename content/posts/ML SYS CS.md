---
title: Untitled
draft: true
tags:
---

# Data Preparation:
- User: {ID, Username, Age, Gender, City, Country, Language}
- Video: {ID, OwnerID, Manual Tags}
- User-Interaction: {UserID, QueryedID, DisplayedID, Interaction Type, Time}
# Feature Engineering:
- Feature Selection: Correlation-based feature selection, Recursive feature elimination
- Transformation: Standardization, Min-max scaling, Log transformation, Binning
- Feature Extraction: Dimensionality reduction (PCA, t-SNE), Feature aggregation
- Feature Encoding: Convert categorical values to numerical
- Feature Importance: Tree-based models (Random forest, XGBoost)
# Data Preprocessing:
- Filling in missing data, Remove outliers
- Partition into Train/Valid/Test Datasets
   - Random sampling, Stratified sampling, K-fold Cross
# Model Selection, Training, Eval:
- ANN: Collaborative Filtering
- Recommendation: 
   - Two-tower Model: User/Post Encoder towers with simliartiy of output encoding representing similarity
   - Loss: Cross-entropy
   - **Evaluation**: AUC, Precision, Recall, F1 Score
   - Retrieval and ranking metrics: recision@k, Recall@k (do not consider ranking quality), mAP, MRR, nDCG
   -  Regression metrics: MSE, MAE,
   - Hyperparameter Tuning: Grid search, Cross-validation
- Image: 
   - ResNet, ViT - Use Constrastive Learning
   - 
# A/B Testing:
- Clear goal: i.e. improve click-through rates/user engagement
- Divide and the target audience/dataset into random/non-overlapping groups
# Serving:

# Monitoring:
- Train Model on new dataset
- Monitor evalution metrics of model