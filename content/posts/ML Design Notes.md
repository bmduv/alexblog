---
title: ML Design Notes
draft: true
tags:
---
Product Objectives and Requirements:
- First thing to do is to clarify business/product priorities for the system. 
   - i.e. if goal is to increase user engagement -> start discussion breaking that down: new user retention, increasing session length, number of sessions...
- Next, discuss product requirements:
   - Is this feed being calculated in real time or can you use batch?
   - Does this apply to all user segments? How many users are there?
   - How quickly do items get stale?
- Also bring up technical scaling requirements:
   - How many items are you dealing with?
   - How many users are there?
   - What are the peak number of requests per second?
   - Hammer out timing SLAs (e.g. we'll incorporate user actions into recommendations within X seconds/minutes/hours)
- Call out any assumptions you are making (but no need to explicitly ask) -> you might call out that it seems important to be able to show recent posts from friends, but you could get clarity on what 'recent' means

High Level Design
- Best if you can create a list of high level solutions and call out pros and cons (but, iterating through multiple approaches doesn't lend itself to every problem...sometimes there's one well known high level approach)
- Look for buy in from interviewer -> incorporate feedback on areas to focus on and see if they bring up red flags or criteria you've missed
- Important design decisions is whether the system is real time, pre calculated batch, or some hybrid. Real time systems limit complexity of the methods available while batch calculations have issues dealing with staleness and new users.

Data Brainstorming and Feature Engineering
- Split up into couple layers of abstraction:
   - Data Sources: high level sources of signals
   - High Level Features: iterate on the types of features available within each data source
   - Feature Representation: pick a subset that provides coverage of a wide range of feature engineering techniques (numeric values (demeaning, scaling, removing outliers), embedding)

Feature Selection:
- Discuss some techniques for feature importance ranking and selection

Offline Training and Evaluation:
- Be aware of data imbalances!
- Discuss how you'd organize the data for evaluation (e.g. k-fold...) 
- Discuss metrics you'd use to compare models

Modeling Techniques:
- Discuss actual implementation -> brining up each model and its pros and cons (i.e. LR is fast to train and compact, but only finds linear relationships)

Online Evaluation:
- Make sure to bring up how you would launch the system and actually evaluate whether its achieving its business objectives. -> almost always via A/B testing -> talk about which metrics you'd measure and statistical test you'd perform for an A/B test