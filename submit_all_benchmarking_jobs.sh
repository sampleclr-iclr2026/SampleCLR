#!/bin/bash

# Script to submit separate SLURM jobs for each dataset/model/model_type combination
# This will create individual jobs for all combinations

# Define datasets, models, and model types
DATASETS=("onek1k" "stephenson" "combat" "hlca")
MODELS=("large" "tiny" "middle")
MODEL_TYPES=("" "--prior" "--supervised" "--prior --supervised")

# SLURM job parameters
QUEUE="gpu_normal"
PARTITION="gpu_p"
GPU="gpu:1"
CPUS="2"
MEMORY="100G"
TIME="3:00:00"

echo "Submitting SLURM jobs for all dataset/model/model_type combinations..."
echo "Total jobs to submit: $(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#MODEL_TYPES[@]} ))"
echo ""

# Counter for job numbering
job_count=0

# Loop through all combinations
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            job_count=$((job_count + 1))
            
            # Create job name based on model type
            if [ -z "$model_type" ]; then
                job_name="sampleclr_${dataset}_${model}_unsupervised"
            elif [ "$model_type" = "--prior" ]; then
                job_name="sampleclr_${dataset}_${model}_prior"
            elif [ "$model_type" = "--supervised" ]; then
                job_name="sampleclr_${dataset}_${model}_supervised"
            else
                job_name="sampleclr_${dataset}_${model}_supervised_prior"
            fi
            
            echo "Submitting job $job_count: $dataset + $model + $model_type"
            
            # Submit the job
            sbatch \
                -q "$QUEUE" \
                -p "$PARTITION" \
                --gres="$GPU" \
                -c "$CPUS" \
                --mem="$MEMORY" \
                -t "$TIME" \
                --job-name="$job_name" \
                --wrap="./experiments/run_benchmark $dataset $model $model_type"
            
            # Small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo ""
echo "All jobs submitted! Use 'squeue -u $USER' to check job status."
echo "Use 'scancel -u $USER' to cancel all your jobs if needed."
