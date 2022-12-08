# Datasets : lastfm mooc wikipedia reddit

# Parameters
# e  : experiment name
# q1 : quantile for end of train, default 0.70 
# q2 : quantile for end of validation, default 0.85

e=7085
q1=0.70
q2=0.85


for dataset in lastfm mooc wikipedia reddit
do
    python utils/preprocess_data.py --data ${dataset} --bipartite
    
    python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --prefix ${e}-tgn
    python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --prefix ${e}-tgn 
            
    python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --embedding_module time --prefix ${e}-jodie
    python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --embedding_module time --prefix ${e}-jodie

    python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --n_layer 2 --n_degree 20 --prefix ${e}-tgat
    python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --n_layer 2 --n_degree 20 --prefix ${e}-tgat

    python train_self_supervised.py --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix ${e}-dyrep
    python predict.py               --data ${dataset} --q1 ${q1} --q2 ${q2} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix ${e}-dyrep
done