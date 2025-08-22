---
title: Reference Frame Transforms
---
In a sensorimotor learning setup one naturally encounters several different reference frames in which information can be represented. Additionally, Monty has it's internal reference frames in which is learns it's object models. Those can get confusing to wrap your head around and keep track of so here is a brief overview of all the reference frames involved in a typical Monty setup.

- [dark green] The object in the world that has an orientation in the world (unknown to LM)
- [darkest green] All the features on the object in the world which have an orientation relative to the object (also unknown to the LM)
- [yellow] The sensor’s orientation in the world (known through proprioception or motor efference copies)
- [dark yellow] The feature orientation relative to the sensor (surface normal and curvature direction extracted from the camera image)
- [bright green] The estimated orientation of the sensed feature in the world (sensor_rel_world * feature_rel_sensor, currently happens to the depth values in DepthTo3DLocations while the surface normal and curvature extraction then happens in the SM but we are planning to pull the transform into the SM)
- [bluementa] Hypothesized orientation of the learned model relative to the object (in the world). This orientation needs to be inferred by the LM based on its sensory inputs. There are usually multiple hypotheses.
- [dark blue] Rotation of the currently sensed feature relative to the object model in the LM (feature_rel_world * object_rel_model).

![](../../figures/how-monty-works/reference_frames_overview.png)

The transform in the sensor module combines the sensor pose in the world with the sensed pose of the features relative to the sensor. This way, if the sensor moves while fixating on a point on the object, that feature pose will not change (see animation below). 
This is one of the key definitions of the CMP: The pose sent out from all SMs is in a common reference frame. In Monty, we use the DepthTo3DLocations transform for this calculation and report locations and orientations in an (arbitrary) world reference frame.

![](../../figures/how-monty-works/sensor_moves.gif)

If the object changes its pose in the world, the pose hypothesis in the LM comes into play. The LM has a hypothesis on how the object is rotated in the world relative to its model of the object. This is essentially the rotation that needs to be applied to rotate the incoming features and movements into the model’s reference frame.
The plot below only shows one pose hypothesis, but in practice, Monty has many of them, and it needs to infer this pose (dark green orientations are unknown).
If the object is in a different rotation than how it was learned, all the features on the object will be sensed in different orientations. This needs to be compensated by the light blue projection of the pose hypothesis.

![](../../figures/how-monty-works/object_moves.gif)

# Reference Frame Transforms for Voting
Voting in Monty happens in object space. We directly translate between the object’s RF of the sending LM to the object RF of the receiving LM. This relies on the assumption that both LMs learned the object at the same time, and hence their RFs line up in orientation and displacement (since we receive features rel. world, which will automatically line up if the LMs learn the object at the same time). Otherwise, we could store one displacement between their RFs and apply that in addition.
The two LMs receive input from two sensors that sense different locations and orientations in space. They receive that as a pose in a common coordinate system (rel. World in the image below). Since the sensors are at different locations on the object and our hypotheses are “locations of sensor rel. Model” we can’t just vote on the hypotheses directly but have to account for the relative sensor displacement. So the sensor that is sensing the handle of the cup needs to incorporate the offset to the sensor that senses the rim to be able to use its hypotheses.
This offset can be easily calculated from the difference of the two LM’s inputs as those poses are in a common coordinate system. The sending LM attaches its sensed pose in the world to the vote message it sends out (along with its hypotheses about the locations on the mug) and the receiving LM compares it with its own sensed pose and applies the difference to the vote hypotheses.

![](../../figures/how-monty-works/voting_rf_transform.gif)

Check out our evidence LM documentation for [more details on voting](../learning-module/evidence-based-learning-module.md#voting-with-evidence).