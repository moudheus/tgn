# lastfm mooc wikipedia reddit
# q1, q2 = 0.70 0.85

e=9095

for dataset in lastfm mooc wikipedia reddit
do
    for q1 in 0.90
    do
        for q2 in 0.95
        do
            #python utils/preprocess_data.py --data ${dataset} --bipartite
            python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --prefix ${e}-tgn
            python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --prefix ${e}-tgn 
            
#            python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --embedding_module time --prefix ${e}-jodie
#            python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --embedding_module time --prefix ${e}-jodie

#            python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --n_layer 2 --n_degree 20 --prefix ${e}-tgat
#            python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --n_layer 2 --n_degree 20 --prefix ${e}-tgat

#            python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix ${e}-dyrep
#            python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix ${e}-dyrep
        done
    done
done