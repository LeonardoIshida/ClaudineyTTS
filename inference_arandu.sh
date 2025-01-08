#!/bin/bash -l

#SBATCH --output=slurm_out_%j.txt

# config de recursos
#SBACTH -n 8
#SBATCH -N 1
#SBATCH --gres=gpu:1

#SBATCH -p arandu

# nome do projeto
PROJECT=Results

hostname

# copiando para pasta output
# rm -r /output/$USER/*
cp -R /home/$USER/$PROJECT/ /output/$USER/
echo "Copiei"

# inicializando docker
# docker images
# docker build -t coquittsyour .
docker run --user "$(id -u):$(id -g)" --rm --gpus \"device=$CUDA_VISIBLE_DEVICES\" -v /output/$USER/$PROJECT:/workspace/data -w /workspace/data coquittsyour python3 your_inference.py

# echo $CUDA_VISIBLE_DEVICES

# movendo conteudo do ouput para home
mv /output/$USER/$PROJECT/teste_inf/* /home/$USER/Inferences

# limpando output
rm -r /output/$USER/$PROJECT

echo "Done"