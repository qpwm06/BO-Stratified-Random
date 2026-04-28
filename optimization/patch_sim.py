"""Per-candidate / per-sequence runfile preparation.

The BO controller treats every sequence as an independent simulation job.
For each candidate, this module:

    1. Copies a user-supplied template ``runfile/<label>/`` into the
       candidate folder so each job has its own working directory.
    2. Calls ``apply_user_patch_hook`` -- a stub you fill in -- to inject
       the BO-proposed parameter vector into your simulation script.
    3. Writes a Slurm submission script tailored to your cluster.

The MD-engine-specific patching that lived here in the original pipeline
(rewriting per-bead parameter values inside an engine-specific ``production.py``,
re-targeting the restart file, collapsing a chunked ``simulation.run``
loop into a single call) has been intentionally omitted from the public
release. The hook below documents the contract; your implementation is
the only project-specific code you need to add.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def apply_user_patch_hook(
    run_dir: Path,
    config: dict[str, Any],
    parameters: dict[str, float],
) -> None:
    """Inject ``parameters`` into the simulation scripts inside ``run_dir``.

    Implement this for your engine. Typical responsibilities:

      * Open ``run_dir / "production.py"`` (or its analogue) and replace
        the parameter table / scalar tunables that the BO is sweeping.
      * Adjust run length, dump cadence, or restart filenames if the BO
        config overrides them.
      * Touch nothing else in the file -- patches should be string-level
        rewrites, not full re-generation, so the rest of the user's
        template stays exactly as written.

    The reference implementation has been removed because it leaks
    engine-specific API names; ``optimization/fake_run.py`` shows a
    minimal runnable replacement that consumes ``params.json`` directly.
    """
    # Default no-op: the fake runner reads parameters straight from the
    # candidate's ``params.json`` and does not need any source patching.
    return None


def patch_job_script(path: str | Path, config: dict, job_name: str, *, run_equilibrium: bool) -> None:
    """Write a generic SBATCH wrapper for one sequence's simulation job."""
    slurm = config["slurm"]
    simulation_python = config["paths"]["simulation_python"]
    module_block = list(slurm.get("module_block", []))
    lines = [
        "#!/bin/bash\n",
        f"#SBATCH --job-name={job_name}\n",
        f"#SBATCH --time={slurm['time']}\n",
        f"#SBATCH --mem={slurm['mem']}\n",
        "#SBATCH --output=slurm-%j-%N.out\n",
        f"#SBATCH --gres={slurm['gres']}\n",
        f"#SBATCH --partition={slurm['partition']}\n",
        f"#SBATCH --nodes={slurm['nodes']}\n",
        f"#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}\n",
        f"#SBATCH --mail-type={slurm.get('mail_type', 'NONE')}\n",
        "\n",
        "# TODO: load whichever modules / activate whichever env your\n",
        "# cluster needs. Populate ``slurm.module_block`` in the config to\n",
        "# emit the lines below.\n",
    ]
    for module_line in module_block:
        lines.append(f"{module_line}\n")
    lines.append("\n")
    if run_equilibrium:
        lines.append(f"{simulation_python} equilibrium.py\n")
    lines.append(f"{simulation_python} production.py\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def copy_existing_restart(config: dict, label: str, destination: Path) -> bool:
    """Optional: copy a pre-equilibrated restart file from the template tree."""
    template_root = Path(config["paths"]["template_run_root"])
    source = template_root / label / "restart_md.gsd"
    if not source.exists():
        return False
    shutil.copy2(source, destination / "bo_restart.gsd")
    return True


def prepare_sequence_run(
    config: dict,
    candidate_dir: Path,
    label: str,
    parameters: dict[str, float],
    candidate_name: str,
) -> Path:
    """Stage one sequence's runfile for a single BO candidate."""
    source_dir = Path(config["paths"]["template_run_root"]) / label
    if not source_dir.exists():
        raise FileNotFoundError(f"Template runfile not found: {source_dir}")
    run_dir = candidate_dir / "runfile" / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(
        source_dir,
        run_dir,
        ignore=shutil.ignore_patterns(
            "slurm-*.out",
            "rg_data.txt",
            "thermo_*.txt",
            "md.gsd",
            "nosol.gsd",
        ),
    )

    mode = config["simulation"].get("mode", "restart")
    use_restart = mode == "restart" and copy_existing_restart(config, label, run_dir)
    run_equilibrium = not use_restart or bool(
        config["simulation"].get("fresh_equilibration", True) and mode == "fresh"
    )

    apply_user_patch_hook(run_dir, config, parameters)
    patch_job_script(
        run_dir / "job.slurm",
        config,
        f"bo_{candidate_name}_{label}"[:120],
        run_equilibrium=run_equilibrium,
    )
    return run_dir
