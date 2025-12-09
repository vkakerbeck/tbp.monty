---
title: Dynamic Adjustment of Hypothesis Space Size
description: Dynamically delete and resample hypotheses throughout the course of an episode.
rfc: https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0013_resampling_dynamic_adjustment.md
estimated-scope: large
improved-metric: speed, pose, accuracy, noise, multi-object
output-type: prototype, PR, monty-feature, publication
skills: python, research, monty
contributor: ramyamounir
status: in-progress
---

> [!NOTE]
> See the [feat.dynamic_resizing](https://github.com/thousandbrainsproject/feat.dynamic_resizing) feature branch to see the current prototype. This prototype validated that dynamically deleting and sampling hypotheses based on their evidence slope helps with accuracy (both object and pose detection) which speeding up inference as most steps have a very small hypothesis space. This feature is planned to be integrated into Monty. However, further ideas for heuristics to delete or resample new hypotheses are appreciated.