---
title: Improve Bounded Evidence Performance
description: If we don't allow evidence to grow infinitely, memory fades. Develop mechanisms to deal with this.
rfc: optional
estimated-scope: medium
improved-metric: multi-object
output-type: prototype, experiments, analysis
skills: python, research, monty
contributor: ramyamounir, scottcanoe
status: paused
---

In the current model, evidence grows infinitely as Monty compares each hypothesis’s predictions with the observed features at every matching step.
Hypotheses accumulate evidence proportional to their similarity to the observed features, which produces unbounded scores.
This behavior is undesirable both for computational reasons and for biological plausibility, real neurons appear to operate within finite ranges.

Bounding evidence through methods such as [exponential moving averages](https://youtu.be/A1cOwvZpgjU?si=1oyk6-BTij6TaRdG&t=1668), weighted averages, or normalization by hypothesis age can keep scores within a bounded range.
However, these techniques compress historical information and cause memory to fade over time.
Monty loses the influence of important past features, such as a mug handle, as a result, recognition can become less accurate in long sequences.

A promising direction is to pair bounded evidence with saliency driven saccades or other attention mechanisms.
If the system consistently revisits the key discriminative features that are important for recognition, then those features remain present in recent history.

This is still an open problem. Ideas around this issue are welcome.

