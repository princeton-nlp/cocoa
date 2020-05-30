EXP_NAME="tom"$1
USE_GPU=$2
SEED=$3
LR=$4

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints checkpoint/language/model_best.pt checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --learning-rate ${LR} \
--agents tom pt-neural-r \
--report-every 50 --max-turns 20 --num-dialogues 10000 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 \
--model lf2lf --model-type a2c --name ${EXP_NAME} --num-cpus 5 \
--epochs 2000 ${USE_GPU} --batch-size 128 --debug ${SEED}