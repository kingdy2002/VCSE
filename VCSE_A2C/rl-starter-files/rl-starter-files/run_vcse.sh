# A2C + VCSE
for task in 'SimpleCrossingS9N1'
do
    for seed in 1
    do
        python3 -m scripts.train --algo a2c --env MiniGrid-$task-v0 --model $task/MiniGrid-$task-v0-vcse-$seed \
        --save-interval 100 --frames 10000 --use_entropy_reward --use_value_condition --seed $seed --beta 0.005 --use_batch
    done
done