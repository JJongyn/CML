import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from torchmeta.utils.data import BatchMetaDataLoader
from maml.utils import load_dataset, load_model, update_parameters, get_accuracy

def main_val(args, mode, iteration=None):
    dataset = load_dataset(args, mode)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model.to(device=args.device)
    model.train()
    
    optimizer = torch.optim.Adam(model.meta_parameters(), args.meta_lr)
    
    if args.meta_train:
        total = args.train_batches
    elif args.meta_val:
        total = args.valid_batches
    elif args.meta_test:
        total = args.test_batches
        
    loss_logs, accuracy_logs = [], []
    
    # Training loop
    with tqdm(dataloader, total=total, leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            
            model.zero_grad()                
            
            support_inputs, support_targets = batch['train']
            support_inputs = support_inputs.to(device=args.device)
            support_targets = support_targets.to(device=args.device)

            query_inputs, query_targets = batch['test']
            query_inputs = query_inputs.to(device=args.device)
            query_targets = query_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
           
            for task_idx, (support_input, support_target, query_input, query_target) in enumerate(zip(support_inputs, support_targets, query_inputs, query_targets)):
                params = None
                # task-adaptation
                for _ in range(args.inner_update_num):
                    support_features, support_logit = model(support_input, params=params)
                    inner_loss = F.cross_entropy(support_logit, support_target)
                    
                    params = update_parameters(model=model,
                                               loss=inner_loss,
                                               params=params,
                                               step_size=args.extractor_step_size,
                                               first_order=args.first_order)
                
                meta_query_features, meta_query_logit = model(query_input, params=params)
    
                outer_loss += F.cross_entropy(meta_query_logit, query_target)
                
                with torch.no_grad():
                    accuracy += get_accuracy(meta_query_logit, query_target)
                    
            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            loss_logs.append(outer_loss.item())
            accuracy_logs.append(accuracy.item())
            
            postfix = {'mode': mode, 'iter': iteration, 'acc': round(accuracy.item(), 5)}
            pbar.set_postfix(postfix)
            if batch_idx+1 == total:
                break

                
    return loss_logs, accuracy_logs

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str, help='Dataset: miniimagenet, tieredimagenet, cub, cars, cifar_fs, fc100, aircraft, vgg_flower')
    parser.add_argument('--model', type=str, help='Model: 4conv, resnet')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--inner-update-num', type=int, default=1, help='The number of inner updates (default: 1).')
    parser.add_argument('--extractor-step-size', type=float, default=0.5, help='Extractor step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--classifier-step-size', type=float, default=0.5, help='Classifier step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--blocks-type', type=str, default=None, help='Resnet block type (optional).')
    
    parser.add_argument('--output-folder', type=str, default='./output/', help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=300, help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 300).')
    parser.add_argument('--train-batches', type=int, default=100, help='Number of batches the model is trained over (i.e., validation save steps) (default: 100).')
    parser.add_argument('--valid-batches', type=int, default=25, help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=2500, help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    
    parser.add_argument('--centering', action='store_true', help='Parallel shift operation in the head.')
    parser.add_argument('--ortho-init', action='store_true', help='Use the head from the orthononal model.')
    parser.add_argument('--outer-fix', action='store_true', help='Fix the head during outer updates.')
    
    # CML
    parser.add_argument('--loss-scaling', type=float, default=1.0, help='Loss scaling factor for Co-learner (default: 1.0).')

    
    args = parser.parse_args()
    args.save_dir = '{}_{}shot_{}_{}'.format(args.dataset,
                                             args.num_shots,
                                             args.model,
                                             args.save_name)


    args.device = torch.device(args.device)
    model = load_model(args)

    base_error = []
    base_acc = []

    result = pd.DataFrame()

    best_Basemodel = os.path.join(args.output_folder, args.save_dir, 'models',
                                    'best_Basemodel.pt')

    model.load_state_dict(
        torch.load(best_Basemodel))

    base_valid_loss_logs, base_valid_accuracy_logs = main_val(args=args, mode='meta_test')
    
    base_test_acc = np.mean(base_valid_accuracy_logs)
    base_test_acc = round(np.mean(base_valid_accuracy_logs)*100, 2)
    base_acc.append(base_test_acc)
    
    result["Base_model_Acc"]= base_acc
    result=result.transpose()
    result.to_csv(os.path.join(args.output_folder, args.save_dir, 'logs', 'Classification_result.csv'))