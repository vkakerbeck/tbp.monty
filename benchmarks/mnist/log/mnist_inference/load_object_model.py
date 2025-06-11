
from pathlib import Path                       # 경로 조작
from copy import deepcopy                      # ObjectModel.copy용

from typing import Optional, Iterable, Mapping, Union
import numpy as np                             # 배열 처리
from numpy.typing import ArrayLike
import torch                                   # model.pt 로드

from scipy.spatial.transform import Rotation as R  # ObjectModel 회전

class ObjectModel:
    """Mutable wrapper for object models.

    Args:
        pos (ArrayLike): The points of the object model as a sequence of points
          (i.e., has shape (n_points, 3)).
        features (Optional[Mapping]): The features of the object model. For
          convenience, the features become attributes of the ObjectModel instance.
    """

    def __init__(
        self,
        pos: ArrayLike,
        features: Optional[Mapping[str, ArrayLike]] = None,
    ):
        self.pos = np.asarray(pos, dtype=float)
        if features:
            for key, value in features.items():
                setattr(self, key, np.asarray(value))

    @property
    def x(self) -> np.ndarray:
        return self.pos[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.pos[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.pos[:, 2]

    def copy(self, deep: bool = True) -> "ObjectModel":
        return deepcopy(self) if deep else self

    def rotated(
        self,
        rotation: Union[R, ArrayLike],
        degrees: bool = False,
    ) -> "ObjectModel":
        """Rotate the object model.

        Args:
            rotation: Rotation to apply. May be one of
              - A `scipy.spatial.transform.Rotation` object.
              - A 3x3 rotation matrix.
              - A 3-element array of x, y, z euler angles.
            degrees (bool): Whether Euler angles are in degrees. Ignored
                if `rotation` is not a 1D array.

        Returns:
            ObjectModel: The rotated object model.
        """
        if isinstance(rotation, R):
            rot = rotation
        else:
            arr = np.asarray(rotation)
            if arr.shape == (3,):
                rot = R.from_euler("xyz", arr, degrees=degrees)
            elif arr.shape == (3, 3):
                rot = R.from_matrix(arr)
            else:
                raise ValueError(f"Invalid rotation argument: {rotation}")

        pos = rot.apply(self.pos)
        out = self.copy()
        out.pos = pos

        return out

    def __add__(self, translation: ArrayLike) -> "ObjectModel":
        translation = np.asarray(translation)
        out = deepcopy(self)
        out.pos += translation
        return out

    def __sub__(self, translation: ArrayLike) -> "ObjectModel":
        translation = np.asarray(translation)
        return self + (-translation)
    
def load_object_model(
    model_name: str,
    object_name: str,
    features: Optional[Iterable[str]] = ("rgba",),
    checkpoint: Optional[int] = None,
    lm_id: int = 0,
) -> ObjectModel:
    """Load an object model from a pretraining experiment.

    Args:
        model_name (str): The name of the model to load (e.g., `dist_agent_1lm`).
        object_name (str): The name of the object to load (e.g., `mug`).
        checkpoint (Optional[int]): The checkpoint to load. Defaults to None. Most
          pretraining experiments aren't checkpointed, so this is usually None.
        lm_id (int): The ID of the LM to load. Defaults to 0.

    Returns:
        ObjectModel: The loaded object model.

    Example:
        >>> model = load_object_model("dist_agent_1lm", "mug")
        >>> model -= [0, 1.5, 0]
        >>> rotation = R.from_euler("xyz", [0, 90, 0], degrees=True)
        >>> rotated = model.rotated(rotation)
        >>> print(model.rgba.shape)
        (1354, 4)
    """
    if checkpoint is None:
        model_path = f"{model_name}"
    else:
        model_path = (
            f"pretrained/checkpoints/{checkpoint}/{model_name}"
        )
    data = torch.load(model_path)
    data = data["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]
    points = np.array(data.pos, dtype=float)
    if features:
        features = [features] if isinstance(features, str) else features
        feature_dict = {}
        for feature in features:
            if feature not in data.feature_mapping:
                print(f"WARNING: Feature {feature} not found in data.feature_mapping")
                continue
            idx = data.feature_mapping[feature]
            feature_data = np.array(data.x[:, idx[0] : idx[1]])
            if feature == "rgba":
                feature_data = feature_data / 255.0
            feature_dict[feature] = feature_data

    return ObjectModel(points, features=feature_dict)

