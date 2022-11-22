# lastfm mooc wikipedia reddit

for dataset in lastfm mooc wikipedia reddit
do
    for q in xxx
    do
        for model in xxx
        do
            python utils/preprocess_data.py --data ${dataset} --bipartite
            python train_self_supervised.py --data ${dataset} --use_memory --prefix tgn
            python predict.py               --data ${dataset} --use_memory --prefix tgn
            
            python train_self_supervised.py --data ${dataset} --use_memory --memory_updater rnn --embedding_module time --prefix jodie
            python predict.py               --data ${dataset} --use_memory --memory_updater rnn --embedding_module time --prefix jodie

            python train_self_supervised.py --data ${dataset} --n_layer 2 --n_degree 20 --prefix tgat
            python predict.py               --data ${dataset} --n_layer 2 --n_degree 20 --prefix tgat

            python train_self_supervised.py --data ${dataset} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep
            python predict.py               --data ${dataset} --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep
        done
    done
done