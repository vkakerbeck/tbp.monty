---
title: Less Dependency on First Observation
description: Currently we only initialize hypotheses on the first step of an episode which make us dependent on the first observation having low noise .
rfc: 
estimated-scope: large
improved-metric: noise, multi-object
output-type: prototype, monty-feature, PR, analysis
skills: python, research, monty
contributor: ramyamounir
status: in-progress
---

The first step in an experiment should not be treated in any special way. We can sample new hypotheses at any point and don't sample any differently on the first step.