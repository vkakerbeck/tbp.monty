# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder


class LivePlotter:
    def __init__(self):
        pass

    def initialize_online_plotting(self):
        self.fig, self.ax = plt.subplots(
            1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [1, 0.8]}
        )
        self.fig.subplots_adjust(top=1.1)
        # self.colorbar = self.fig.colorbar(None, fraction=0.046, pad=0.04)
        self.setup_camera_ax()
        self.setup_sensor_ax()

    def show_observations(
        self, observation, model, step: int, is_saccade_on_image_data_loader=False
    ) -> None:
        self.fig.suptitle(f"Observation at step {step}")
        self.show_view_finder(observation, model, step, is_saccade_on_image_data_loader)
        self.show_patch(observation, model)
        plt.pause(0.00001)

    def show_view_finder(
        self,
        observation,
        model,
        step,
        is_saccade_on_image_data_loader,
        sensor_id="view_finder",
    ):
        if self.camera_image:
            self.camera_image.remove()

        view_finder_image = observation[model.motor_system._policy.agent_id][sensor_id][
            "rgba"
        ]
        if is_saccade_on_image_data_loader:
            center_pixel_id = np.array([200, 200])
            patch_size = np.array(
                observation[model.motor_system._policy.agent_id]["patch"]["depth"]
            ).shape[0]
            raw_obs = model.sensor_modules[0].raw_observations
            if len(raw_obs) > 0:
                center_pixel_id = np.array(raw_obs[-1]["pixel_loc"])
                view_finder_image = add_patch_outline_to_view_finder(
                    view_finder_image, center_pixel_id, patch_size
                )
            self.camera_image = self.ax[0].imshow(view_finder_image, zorder=-99)
        else:
            self.camera_image = self.ax[0].imshow(
                view_finder_image,
                zorder=-99,
            )
            # Show a square in the middle as a rough estimate of where the patch is
            # Note: This isn't exactly the size that the patch actually is.
            image_shape = observation[model.motor_system._policy.agent_id][sensor_id][
                "rgba"
            ].shape
            square = plt.Rectangle(
                (image_shape[1] * 4.5 // 10, image_shape[0] * 4.5 // 10),
                image_shape[1] / 10,
                image_shape[0] / 10,
                fc="none",
                ec="white",
            )
            self.ax[0].add_patch(square)
        if hasattr(model.learning_modules[0].graph_memory, "current_mlh"):
            mlh = model.learning_modules[0].get_current_mlh()
            if mlh is not None:
                self.add_text(mlh, pos=view_finder_image.shape[0], model=model)

    def show_patch(self, observation, model, sensor_id="patch"):
        if self.depth_image:
            self.depth_image.remove()
        self.depth_image = self.ax[1].imshow(
            observation[model.motor_system._policy.agent_id][sensor_id]["depth"],
            cmap="viridis_r",
        )
        # self.colorbar.update_normal(self.depth_image)

    def add_text(self, mlh, pos, model):
        if self.text:
            self.text.remove()
        new_text = r"MLH: "
        mlh_id = mlh["graph_id"].split("_")
        for word in mlh_id:
            new_text += r"$\bf{" + word + "}$ "
        new_text += f"with evidence {np.round(mlh['evidence'], 2)}\n\n"
        pms = model.learning_modules[0].get_possible_matches()
        graph_ids, evidences = model.learning_modules[
            0
        ].graph_memory.get_evidence_for_each_graph()

        # Highlight 2nd MLH if present
        if len(evidences) > 1:
            top_indices = np.flip(np.argsort(evidences))[0:2]
            second_id = graph_ids[top_indices[1]].split("_")
            new_text += "2nd MLH: "
            for word in second_id:
                new_text += r"$\bf{" + word + "}$ "
            new_text += f"with evidence {np.round(evidences[top_indices[1]], 2)}\n\n"

        new_text += r"$\bf{Possible}$ $\bf{matches:}$"
        for gid, ev in zip(graph_ids, evidences):
            if gid in pms:
                new_text += f"\n{gid}: {np.round(ev, 1)}"

        self.text = self.ax[0].text(0, pos + 30, new_text, va="top")

    def setup_camera_ax(self):
        self.ax[0].set_title("Camera image")
        self.ax[0].set_axis_off()
        self.camera_image = None
        self.text = None

    def setup_sensor_ax(self):
        self.ax[1].set_title("Sensor depth image")
        self.ax[1].set_axis_off()
        self.depth_image = None
