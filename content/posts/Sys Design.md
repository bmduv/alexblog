---
title: Untitled
draft: true
tags:
---
 Content-based  Filtering: Uses video features to recommend new videos similar to those users found relevant in the past
 - Pros: Ability to recomment new videos, Abilty to capture unique interest of users
 - Cons: Difficult to capture user's new interests, Requires domain knowledge (in order to decide on relevancy of videos)

Collaborative Filtering: Uses user-user similarities or video-video similarities to recommend new videos using intuition that similar users are intereseted in similar videos
- Pros: No domain knowledge needed (don't need video features), Easy to discover user's new interests, Efficient
- Cons: Cold-start problem, Cannot handle niche interests (not a lot of users in niche areas)

Hybrid Filtering: Combination of both content-based and collaborative filtering either using in parallel or sequentially

Data:
- Example Video Features: Video ID, Length, Manual Tags, Manual Title, Likes, Views, Language
- Example User Features: ID, Username, Name, Age, Gender, City, Country, Language, Time Zone
- Example User-Video Interaction Features: User ID, Video ID, Interaction type, Interaction value, Location, Timestamp

Feature Engineering:
- Video Features: Video ID -> Embedded, Duration, Language -> Embedded, Title -> Map to Feature Vector (BERT), Tags -> Map to Feature Vector (CBOW) ---> Concatenate
![[posts/imgs/ch6-07-1-JV5OED2M.webp]]
- User Features: Categorize user features into following buckets: 1. User demographics 2. Contextual information 3. User historical interactions
![[public/posts/imgs/h6-08-1-PCWNBYAP.webp]]
- Contextual Information: Time of Day, Device, Day of week
- User Historical Interactions: Search history -> Map each search query into embedding vector and average all query embeddings, Liked videos -> Map into embedding vectors and average all embeddings, Watch videos and impressions -> Map into embedding vector and average
![[posts/imgs/ch6-10-1-JEQFGCK2 1.webp]]

Model Development:
- Feedback Matrix: Represents users' opinions about videos with rows being user and columns being videos
   - Explicit Feedback: Accurately reflects users' opinions about videos (captures explicit actions on video such as likes or sharing), BUT it is sparse since only handful of users interact explicitly
   - Implicit Feedback: Implicit actions indicating users' opinions about videos such as clicks and watch time (may be noisy and not as accurate)
   - Matrix Factorization Model: Decomposes user-video feedback matrix into two lower-dimensional matrices with one representing user embeddings and the other representing video embeddings. Learns to map each user into an embedding vector and each video into an embedding vector such that their distances represent relevance
- Two-tower Neural Network: Two encoder towers (user encoder, video encoder), the distance between the two embeddings represent relevance
   - Label positive pairs <user, video> based on user feedback (liked, watched at least half). Label negative data points by choosing random nonrelevant videos to user or those explicitly disliked
Evaluation:
- Offlince Metrics: 
   - Precision@k: Measures proportion of relevant videos among top k recommended videos
   - mAP: Measures ranking quality of recommended videos
   - Diversity: Measures how dissimilar recommended videos are to each other (calculate average pairwise similarity)
- Online Metrics:
  - CTR: Ratio between clicked videos and total number of recommended videos
  - Number of completed videos
  - Total watch time
  - Explicit user Feedback
Serving:
- Candidate Generation: Narrow down large amount of videos to smaller number of candidates (billions to thousands). Obtains user embedding from user encoder, and retrieves most similar videos from ANN service
- Scoring/ranking: Take user and candidate as input and output ranked list. 
- Re-ranking: Re-rank videos by adding additional criteria/constraints (Region-restricted, video freshness, video spreading misinformation...)