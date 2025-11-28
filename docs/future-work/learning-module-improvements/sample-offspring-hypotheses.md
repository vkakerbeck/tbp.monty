---
title: Sample Offspring Hypotheses
description: Sample new hypotheses that are similar to the existing top hypotheses.
rfc: optional
estimated-scope: medium
improved-metric: speed, accuracy, pose
output-type: prototype, monty-feature, PR
skills: python, research
contributor: 
status: open
---

Similar to some SLAM approaches, sample new hypotheses based on the current top contenders. At the moment, new hypotheses are sampled based on the current observed features and points stored in the model. They do not take into account the current top hypotheses. Offspring hypotheses would be slight variations on the current top hypotheses, to refine the hypothesized location and orientation of the object.