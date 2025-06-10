import argparse

def arg_parser():

    parser = argparse.ArgumentParser()

    # =========== Optimizer settings ======================
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'],
                        help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of optimizer.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay.')
    
    parser.add_argument('--epochs', type=int,  default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--eval_freq', type=int,  default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--onlyUnlabel', type=str,
                        choices=['yes', 'no'],
                        default='yes')
    
    # =========== Consistency Regularization settings ======================
    parser.add_argument('--cr_loss', type=str,
                        choices=['kl', 'l2'],
                        default='kl')
    parser.add_argument('--cr_tem', type=float, default=1,
                        help='Temperature for CR.')
    parser.add_argument('--cr_conf', type=float, default=0.5,
                        help='Confidence level for CR.')
    
    parser.add_argument('--lambda_cr', type=float, default=0.5,
                        help='lambda cr')
    
    parser.add_argument('--lambda_ce', type=float, default=0.5,
                        help='lambda ce.')
    
    parser.add_argument('--lambda_g', type=float, default=0.5,
                        help='lambda g.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta for prediction.')

    # =========== RGCN settings ===========================
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_relations', type=int, default=6)  # 6 relations: 3 types x 2 directions

    # =========== Dataset settings ======================
    parser.add_argument('--dataset', type=str, default="MM COVID",
                        choices=['MM COVID', 'ReCOVery','MC Fake', 'LIAR', 'PAN2020', 'Random_test', 'PolitiFact', 'Knowledge_PolitiFact', 'GossipCop', 'Knowledge_GossipCop', 'GossipCop_4000_news', 'Knowledge_GossipCop_4000_news', 'Knowledge_more5_GossipCop', 'Knowledge_more15_GossipCop', 'Knowledge_more5_PolitiFact', 'Knowledge_more15_PolitiFact', 'Knowledge_more15_PolitiFact_nontrun', 'Knowledge_more15_PolitiFact_nontrun_Roberta', 'Knowledge_more15_GossipCop_nontrun', 'GossipCop_nontrun', 'PolitiFact_nontrun', 'Knowledge_llmchoose_PolitiFact', 'Knowledge_llmchoose_GossipCop', 'Knowledge_llmchoose_PolitiFact-50-n', 'Knowledge_llmchoose_PolitiFact-50-15', 'Knowledge_llmchoose_PolitiFact-30-15', 'Knowledge_llmchoose_PolitiFact-50-n-true', 'Knowledge_llmchoose_PolitiFact-test', 'Knowledge_llmchoose_GossipCop-test', 'Knowledge_llmchoose_GossipCop-30-15', 'Knowledge_llmchoose_GossipCop-50-15', 'Knowledge_llmchoose_GossipCop-50-n', 'KDD2020', 'Liar'], help='dataset')
    parser.add_argument("--tr", type=float, default=0.8,
                        help='rate of training data')
    parser.add_argument("--vr", type=float, default=0.1,
                        help='rate of validation data')
    parser.add_argument("--num_topics", type=int,
                        help='number of topic nodes')

    
    # =========== General settings ======================
    parser.add_argument('--verbose', choices=["True", "False"], default="False",
                        help='show tqdm bar?')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--patience', type=int, default=80)

    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--local_layers', type=int, default=3, help='Number of layers for local view')
    parser.add_argument('--global_layers', type=int, default=5, help='Number of layers for global view')

    parser.add_argument('--lambda_gating_div', type=float, default=0.1,
                        help='lambda for gating diversity regularization (KL)')

    args = parser.parse_args()
    
    return args