# A2C
for task in 'SimpleCrossingS9N1'
do
    for seed in 1 2 3 4
    do
        python3 -m scripts.train --algo a2c --env MiniGrid-$task-v0 --model $task/MiniGrid-$task-v0-original-$seed \
        --save-interval 100 --frames 1000000 --seed $seed --use_batch
    done
done