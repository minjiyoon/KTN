set -o errexit

# computer science
SOURCE_NODES=("paper" "author" "author" "venue" "paper" "author" "author" "venue")
TARGET_NODES=("author" "paper" "venue" "author" "author" "paper" "venue" "author")
SOURCE_TASKS=("L1" "L1" "L1" "L1" "L2" "L2" "L2" "L2")
TARGET_TASKS=("L1" "L1" "L1" "L1" "L2" "L2" "L2" "L2")
TRAIN_BATCHES=(200 200 200 60 400 400 400 60)
TEST_BATCHES=(50 50 50 40 50 50 50 40)
TRAIN_BATCHES2=(2 2 2 2 10 10 10 10)
TEST_BATCHES2=(50 50 40 50 50 50 40 50)
MATCHING_HOPSS=(1 1 2 2 1 1 2 2)

length=${#SOURCE_NODES[@]}

for ((i=0;i<=$length;i++))
do
    python main.py --dataset "graph_CS/" \
         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
done


## computer network
#SOURCE_NODES=("paper" "author" "author" "venue")
#TARGET_NODES=("author" "paper" "venue" "author")
#SOURCE_TASKS=("L2" "L2" "L2" "L2")
#TARGET_TASKS=("L2" "L2" "L2" "L2")
#TRAIN_BATCHES=(250 250 250 25)
#TEST_BATCHES=(50 50 50 5)
#TRAIN_BATCHES2=(10 10 10 10)
#TEST_BATCHES2=(50 50 5 50)
#MATCHING_HOPSS=(1 1 2 2)
#
#length=${#SOURCE_NODES[@]}
#
#for ((i=0;i<=$length;i++))
#do
#    python main.py --dataset "graph_CN/" \
#         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
#done
#
#
## machine learning
#SOURCE_NODES=("paper" "author" "author" "venue")
#TARGET_NODES=("author" "paper" "venue" "author")
#SOURCE_TASKS=("L2" "L2" "L2" "L2")
#TARGET_TASKS=("L2" "L2" "L2" "L2")
#TRAIN_BATCHES=(400 400 400 35)
#TEST_BATCHES=(50 50 50 15)
#TRAIN_BATCHES2=(10 10 10 10)
#TEST_BATCHES2=(50 50 15 50)
#MATCHING_HOPSS=(1 1 2 2)
#
#length=${#SOURCE_NODES[@]}
#
#for ((i=0;i<=$length;i++))
#do
#    python main.py --dataset "graph_ML/" \
#         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
#done


