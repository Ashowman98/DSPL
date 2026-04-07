DATASET='cifar-10-10'
SAVE_PATH="./results/test/${DATASET}"
for SPLIT_IDX in 0 1 2 3 4; do
    python main.py --dataset=$DATASET --split_idx=${SPLIT_IDX} --save_path=${SAVE_PATH}\
                          --lambda1=0.01 --lambda2=0.1 --lambda3=10 --alpha1=0.4 --gpu=0 --seed=1

done

##test
#sum_auc=0
#sum_oscr=0
#sum_macrof1=0
#count=5
#for SPLIT_IDX in 0 1 2 3 4; do
#    result=$(python main.py --dataset=$DATASET --split_idx=${SPLIT_IDX} --save_path=${SAVE_PATH} --alpha1=0.4 --gpu 0 --eval | tail -n 3)
#    auc=$(echo "$result" | head -n 1)
#    oscr=$(echo "$result" | head -n 2 | tail -n 1)
#    macrof1=$(echo "$result" | tail -n 1)
#    echo "auc = $auc"
#    echo "oscr = $oscr"
#    echo "macrof1 = $macrof1"
#
#    sum_auc=$(echo "$sum_auc + $auc" | bc)
#    sum_oscr=$(echo "$sum_oscr + $oscr" | bc)
#    sum_macrof1=$(echo "$sum_macrof1 + $macrof1" | bc)
#done
#avg_auc=$(echo "scale=4; $sum_auc / $count" | bc)
#avg_oscr=$(echo "scale=4; $sum_oscr / $count" | bc)
#avg_macrof1=$(echo "scale=4; $sum_macrof1 / $count" | bc)
#echo "Average auc: $avg_auc"
#echo "Average oscr: $avg_oscr"
#echo "Average macrof1: $avg_macrof1"