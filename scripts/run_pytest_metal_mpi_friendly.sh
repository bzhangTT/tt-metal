#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run pytest with a wider per-user process cap (when the hard ulimit allows) and Open MPI
# singleton-friendly MCA defaults, before Python loads libtt_metal / libmpi.
#
# Usage (from tt-metal repo root):
#   ./scripts/run_pytest_metal_mpi_friendly.sh models/demos/mistral_small_4_119B/tests/test_embedding.py -v
#
# If this script cannot widen ulimit (policy), you still need fewer stray processes or a higher
# hard limit from your admin.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(ulimit -u)" != unlimited ]]; then
  HARD="$(ulimit -Hu 2>/dev/null || true)"
  if [[ -n "${HARD}" && "${HARD}" != unlimited ]]; then
    ulimit -u "${HARD}" 2>/dev/null || true
  fi
fi

export TT_METAL_OMPI_SINGLETON_WORKAROUND="${TT_METAL_OMPI_SINGLETON_WORKAROUND:-1}"
export OMPI_MCA_plm="${OMPI_MCA_plm:-isolated}"
export PRTE_MCA_plm="${PRTE_MCA_plm:-isolated}"
export OMPI_MCA_plm_ssh_agent="${OMPI_MCA_plm_ssh_agent:-false}"
export OMPI_MCA_plm_rsh_agent="${OMPI_MCA_plm_rsh_agent:-false}"

exec python3 -m pytest "$@"
