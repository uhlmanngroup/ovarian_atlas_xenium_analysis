import tempfile
import subprocess
import os
from .config import SLURMConfig

def submit_slurm_job(
    selected_command, 
    config_file: str,
    container_path=None,
    job_name="hf_submit", 
    output="slurm-%j.out",
    ):
    """
    Submits a SLURM job with the specified parameters.
    
    :param command: The command to run within the SLURM job.
    :param job_name: Name of the SLURM job.
    :param partition: The SLURM partition to submit to.
    :param time: Max time limit (HH:MM:SS).
    :param nodes: Number of nodes to request.
    :param ntasks: Number of tasks.
    :param cpus_per_task: CPUs per task.
    :param mem: Memory per node (e.g., "4G").
    :param output: Output file for SLURM logs.
    """

    config = SLURMConfig.for_task(str(selected_command))

    if container_path is not None:
        execution_command = f"singularity exec {container_path} python -m histology_features {selected_command} --config_file={config_file}"
    else:
        execution_command = f"python -m histology_features {selected_command} --config_file={config_file}"

    if selected_command:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".sh") as script_file:
            # Generate the SLURM script
            script_file.write(f"""#!/bin/bash
#SBATCH --job-name={selected_command}
#SBATCH --output={selected_command}.out
#SBATCH --error={selected_command}.err
#SBATCH --time={config.TIME}
#SBATCH --partition={config.PARTITION}
#SBATCH --cpus-per-task={config.CPU_PER_TASK}
#SBATCH --mem={config.MEMORY}

{execution_command}
""".strip())

    script_path = script_file.name

    with open(script_path, "r") as file:  # Replace with your .sh file path
        content = file.read()
        print(content)

    try:
        subprocess.run(["sbatch", script_path], check=True)
    finally:
        os.remove(script_path)