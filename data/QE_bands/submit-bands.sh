#! /usr/bin/env bash

#SBATCH --job-name=MoS2-bands
#SBATCH --output=mos2-bands.log
#SBATCH --partition=b_standard
#SBATCH --mem=0
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --hint=nomultithread


source /nix/var/nix/profiles/default/etc/profile.d/nix.sh
source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh

export PATH_TO_FLAKE=/beegfs/lu76tel/QE/nix/

nix develop $PATH_TO_FLAKE --command bash -c "
        export OMP_NUM_THREADS=1
        mpirun -np 24 pw.x -npool 6  -in MoS2-bands.pw.in > MoS2-bands.pw.out
	mpirun -np 1 bands.x -in MoS2.bands.in > MoS2.bands.out
	"

