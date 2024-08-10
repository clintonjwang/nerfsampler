#! /bin/bash
if [ ! -d /data/vision/polina/users/clintonw/code/inrnet/results/$1 ]
then
    mkdir /data/vision/polina/users/clintonw/code/inrnet/results/$1
fi
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=${2:-gpu}
#SBATCH --time=0
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/inrnet/results/$1/err.txt
#SBATCH -o /data/vision/polina/users/clintonw/code/inrnet/results/$1/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=rosemary,sumac,anise

cd /data/vision/polina/users/clintonw/code/inrnet/nerfsampler
source .bashrc
python train.py -j=$1 -c=$1
exit()
EOT

#SBATCH --exclude=zaatar,anise,mint,clove