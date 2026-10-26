#!/usr/bin/env bash
set -euo pipefail

ROOT=/users/sr1937/scratch/Hermes-Final-Pro-Max
BUILD=$ROOT/Hermes-3/sf-new-limiter
JOBS=$ROOT/SF-Sims/Jobs
mkdir -p "$JOBS"

for set in No_Drifts CVODE_test; do
  for run in SF-ideal SF15 SF45 SF75 SF105 SF135 SF165 SN; do
    case $run in
      SF-ideal) n=72 ;;
      SN)       n=6  ;;
      *)        n=18 ;;
    esac
    f=$JOBS/${set}_${run}.job
    cat > "$f" <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=${run}
#SBATCH --ntasks=${n}
#SBATCH --mem-per-cpu=5G
#SBATCH --time=2-00:00:00
#SBATCH --account=pet-bout-2025
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ruizgonzalez@york.ac.uk
#SBATCH --output=${ROOT}/SF-Sims/${set}/${run}/%x-%j.log

set -e

source ${ROOT}/hermes.env

cd ${BUILD}

mpiexec -n ${n} ./hermes-3 -d ${ROOT}/SF-Sims/${set}/${run} restart
EOF
    echo "wrote $f  (${n} ranks)"
  done
done
