num_tasks=$(($(wc -l < param_grid.csv) - 1))
sbatch --array=0-$(($num_tasks - 1)) run.sh
