# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.mixins.no_reset_evidence import (
    TheoreticalLimitLMLoggingMixin,
)


class TelemetryEvidenceGraphLM(TheoreticalLimitLMLoggingMixin, EvidenceGraphLM):
    """Evidence graph learning module with detailed plotting telemetry."""

    def _add_detailed_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Add standard evidence statistics and plotting telemetry.

        Args:
            stats: Existing statistics for the current learning-module step.

        Returns:
            Statistics augmented with standard evidence details, theoretical-limit
            metrics, and hypothesis-updater telemetry.
        """
        stats = EvidenceGraphLM._add_detailed_stats(self, stats)
        return TheoreticalLimitLMLoggingMixin._add_detailed_stats(self, stats)
