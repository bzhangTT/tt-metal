# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Open MPI / PRRTE environment for ``python -m pytest`` without ``mpirun`` (MPI singleton).

``libtt_metal`` may call ``MPI_Init_thread``. In singleton mode ORTE can fork a large ``prted``
process tree until the user hits ``RLIMIT_NPROC`` / max-children (often reported as
``orte_init failed`` / ``The system limit on number of children ... was reached``). When built with
MPI, Metal also bumps ``RLIMIT_NPROC`` (best effort) and applies the same MCA defaults immediately
before ``MPI_Init_thread`` (see ``tt_metal/distributed/multihost/mpi_distributed_context.cpp``).

Set ``TT_METAL_OMPI_SINGLETON_WORKAROUND=0`` to skip the Python-side MCA defaults (e.g. Slurm
``mpirun`` workflows that need a different PLM); set ``TT_METAL_SKIP_RLIM_NPROC_BUMP=1`` to skip
only the ``RLIMIT_NPROC`` raise in Python (C++ bump is separate: ``TT_METAL_SKIP_RLIM_NPROC_BUMP``).

Operational mitigations when soft and hard limits are already equal and low: raise ``ulimit -u`` in
the shell, kill stray ``prte``/``orted`` processes, reduce parallel pytest workers, or use
``scripts/run_pytest_metal_mpi_friendly.sh``.
"""

from __future__ import annotations

import os


def _try_raise_rlimit_nproc() -> None:
    """Best-effort raise soft RLIMIT_NPROC before ORTE forks (Linux only).

    Many failures with soft < hard are fixed by aligning soft to hard. When the hard limit is
    unlimited but the soft cap is low, bump soft to a moderate ceiling. Opt out:
    ``TT_METAL_SKIP_RLIM_NPROC_BUMP=1``.
    """
    if os.environ.get("TT_METAL_SKIP_RLIM_NPROC_BUMP", "") == "1":
        return
    try:
        import resource
        from resource import RLIMIT_NPROC
    except (ImportError, AttributeError):
        return
    try:
        soft, hard = resource.getrlimit(RLIMIT_NPROC)
        if hard != resource.RLIM_INFINITY:
            if soft < hard:
                resource.setrlimit(RLIMIT_NPROC, (hard, hard))
            return
        target_soft = 262144
        if soft < target_soft:
            resource.setrlimit(RLIMIT_NPROC, (target_soft, hard))
    except (ValueError, OSError):
        pass


def apply_ompi_singleton_workaround_env() -> None:
    """Apply conservative MCA defaults if not already set by the user or site."""
    _try_raise_rlimit_nproc()

    if os.environ.get("TT_METAL_OMPI_SINGLETON_WORKAROUND", "1") == "0":
        return

    # Minimal process launch manager for singleton / non-rsh launches.
    os.environ.setdefault("OMPI_MCA_plm", "isolated")
    os.environ.setdefault("PRTE_MCA_plm", "isolated")

    # Avoid forking an SSH/rsh-based remote launcher when no remote nodes are used.
    os.environ.setdefault("OMPI_MCA_plm_ssh_agent", "false")
    os.environ.setdefault("OMPI_MCA_plm_rsh_agent", "false")
