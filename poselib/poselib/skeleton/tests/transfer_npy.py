# Copyright (c) respective authors.
# Distributed under the repository license.

import numpy as np
import os
from pathlib import Path
from ...core import Tensor, SO3, Quaternion, Vector3D
from ..skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

default_skeleton_dir = Path(__file__).resolve().parents[5] / "data" / "skeletons"
skeleton_dir = Path(os.environ.get("HORD_SKELETONS_DIR", str(default_skeleton_dir)))

tpose = np.load(
    str(skeleton_dir / "flex_tpose.npy")
).item()

local_rotation = SO3.from_numpy(tpose["local_rotation"], dtype="float32")
root_translation = Vector3D.from_numpy(tpose["root_translation"], dtype="float32")
skeleton_tree = tpose["skeleton_tree"]
parent_indices = Tensor.from_numpy(skeleton_tree["parent_indices"], dtype="int32")
local_translation = Vector3D.from_numpy(
    skeleton_tree["local_translation"], dtype="float32"
)
node_names = skeleton_tree["node_names"]
skeleton_tree = SkeletonTree(node_names, parent_indices, local_translation)
skeleton_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=skeleton_tree, r=local_rotation, t=root_translation, is_local=True
)

skeleton_state.to_file(
    str(skeleton_dir / "flex_tpose_new.npy")
)
