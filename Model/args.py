import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Optimization
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait before early stop.')
    parser.add_argument('--n_pool', type=int, default=4,
                        help='Number of process to sample subgraph')

    # Dataset
    parser.add_argument('--data_dir', type=str, default="../Data/",
                        help='Dataset location.')
    parser.add_argument('--dataset', type=str, default="graph_CS/",
                        help='Dataset to use.')
    parser.add_argument('--ranking', dest='ranking', action='store_true')
    parser.add_argument('--classification', dest='ranking', action='store_false')
    parser.set_defaults(ranking=True)
    parser.add_argument('--save_model', type=str, default="./save/",
                        help='Saved models.')
    parser.add_argument('--source_node', type=str, default="paper",
                        help='Source node type.')
    parser.add_argument('--target_node', type=str, default="author",
                        help='Target node type.')
    parser.add_argument('--source_task', type=str, default="L1",
                        help='Source node task.')
    parser.add_argument('--target_task', type=str, default="L1",
                        help='Target node task.')

    # Model
    parser.add_argument('--model_name', type=str, default="HGNN",
                        help='GNN model')
    parser.add_argument('--matching_coeff', type=float, default=1.,
                        help='Coefficient of matching loss.')
    parser.add_argument('--matching_hops', type=int, default=1,
                        help='Number of propagating steps in matching loss')
    parser.add_argument('--matching_loss', dest='matching_loss', action='store_true')
    parser.add_argument('--no_matching_loss', dest='matching_loss', action='store_false')
    parser.set_defaults(matching_loss=True)

    # Batchsize
    parser.add_argument('--train_batch', type=int, default=200,
                        help='Number of train batches.')
    parser.add_argument('--val_batch', type=int, default=10,
                        help='Number of validation batches.')
    parser.add_argument('--test_batch', type=int, default=50,
                        help='Number of test batches.')

    parser.add_argument('--train_batch2', type=int, default=2,
                        help='Number of train batches.')
    parser.add_argument('--val_batch2', type=int, default=1,
                        help='Number of validation batches.')
    parser.add_argument('--test_batch2', type=int, default=50,
                        help='Number of test batches.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch.')

    # Architecture
    parser.add_argument('--step_num', type=int, default=4,
                        help='Number of propagating steps')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='Dimension of transformation layers')
    parser.add_argument('--rel_dim', type=int, default=128,
                        help='Dimension of message layers')

    # Sampling
    parser.add_argument('--sample_num', type=int, default=1,
                        help='Number of sampled neighbors')
    parser.add_argument('--sample_depth', type=int, default=4,
                        help='Depth of sampled subgraphs')

    args, _ = parser.parse_known_args()
    return args
