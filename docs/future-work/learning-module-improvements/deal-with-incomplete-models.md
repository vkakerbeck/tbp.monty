---
title: Deal with Incomplete Models
description: Monty has no concept of whether an object doesn't exist at a location or it's model is just incomplete.
rfc: required
estimated-scope: unknown
improved-metric: learning
output-type: prototype, monty-feature, PR
skills: python, research, monty
contributor: 
status: open
---

As described in [this document](https://www.overleaf.com/read/qxchttxzpfnd#ea9111) (section 3.3.2), Monty's object models have no concept of whether the absence of a point in the model means that the object doesn't exist there or the sensor hasn't explored this area yet.

![Problem with the current algorithm and incomplete models. Assume we are trying to infer a cube.
There is no mechanism to distinguish between the scenario on the left (learned a partial model of the cube and sampling a new point on it) and the one on the right (sampling a point that is not on the cube). (The one in the middle relates to [surface representations to allow for sparser models](better-model-surface-representation.md).) In the case on the left, we would want to keep the cube hypothesis and extend the model for the cube with new points. In the case on the right, we would want to say that we are not on a cube anymore. Currently, this is what happens in both cases.](../../figures/future-work/incomplete_models.png)

It's unclear how this would best be solved. A possibility would be to use heuristics of how an object's surface might continue based on the already stored surface models. One could also store points in the model that represent the end of an object (although this seems to already be encoded in the existing surface normals). 