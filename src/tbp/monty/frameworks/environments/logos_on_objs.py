# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""
Levels of Difficulty for Testing Compositional Models:
1. Logos on flat surfaces
2. Logos on curved surfaces
3. Logos at different rotations
4. Logos with bends
"""

LOGOS = ["021_logo_tbp", "022_logo_numenta"]

FLAT_OBJECTS_WITHOUT_LOGOS = ["001_cube", "006_disk"]

OBJECTS_WITH_LOGOS_LVL1 = [
    "002_cube_tbp_horz",
    "004_cube_numenta_horz",
    "007_disk_tbp_horz",
    "009_disk_numenta_horz",
]

PARENT_TO_CHILD_MAPPING_LVL1 = {
    "002_cube_tbp_horz": ["001_cube", "021_logo_tbp"],
    "004_cube_numenta_horz": ["001_cube", "022_logo_numenta"],
    "007_disk_tbp_horz": ["006_disk", "021_logo_tbp"],
    "009_disk_numenta_horz": ["006_disk", "022_logo_numenta"],
}