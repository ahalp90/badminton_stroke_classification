#!/bin/bash

set -e

echo "Creating local data directories...."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/checkpoints
mkdir -p data/logs

echo "Done!"
echo ""
echo "Local structure:"
echo "  data/raw"
echo "  data/processed"
echo "  data/checkpoints"
echo "  data/logs"
echo ""
echo "Recommended HPC structure:"
echo "  /scratch/comp320a/badminton-stroke-classifier/data/"
