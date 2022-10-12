import numpy as np
import random
import torch
import torch.optim as optim

from model import Classifier, HGNN


def move(device, feats, adjs, labels, ids):
    if device == torch.device("cpu"):
        return feats, adjs, labels, ids
    for type_ in feats.keys():
        feats[type_] = feats[type_].to(device)
    for target_type_ in adjs.keys():
        for source_type_ in adjs[target_type_].keys():
            for rel_type_ in adjs[target_type_][source_type_].keys():
                adjs[target_type_][source_type_][rel_type_]['source'] = adjs[target_type_][source_type_][rel_type_]['source'].to(device)
                adjs[target_type_][source_type_][rel_type_]['target'] = adjs[target_type_][source_type_][rel_type_]['target'].to(device)
                adjs[target_type_][source_type_][rel_type_]['adj'] = adjs[target_type_][source_type_][rel_type_]['adj'].to(device)
                adjs[target_type_][source_type_][rel_type_]['agg_adj'] = adjs[target_type_][source_type_][rel_type_]['agg_adj'].to(device)
    return feats, adjs, labels.to(device), ids


def evaluate(gnn, classifier, batch_start, batch_num, target_node, target_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stack_output = []
    stack_label = []
    total_matching_loss = 0.
    for batch in range(batch_start, batch_start + batch_num):
        target_feats, target_adjs, target_labels, target_ids = target_data[batch]
        target_feats, target_adjs, target_labels, target_ids = move(device, target_feats, target_adjs, target_labels, target_ids)

        with torch.no_grad():
            gnn.eval()
            classifier.eval()
            target_message, target_cmd = gnn(target_feats, target_adjs)
            if target_node == gnn.source_node:
                target_output = classifier(target_message[target_node][target_ids])
                # Matching loss
                h_S = target_message[gnn.source_node].clone().detach().to(device)
                h_T = target_message[gnn.target_node].clone().detach().to(device)
                matching_loss = gnn.get_matching_loss(target_adjs, h_S, h_T).item()
                total_matching_loss = total_matching_loss + matching_loss
            else:
                transform_message = gnn.transform(target_message[target_node][target_ids])
                target_output = classifier(transform_message)
            stack_output.append(target_output)
            stack_label.append(target_labels[target_ids])

    stack_output = torch.cat(stack_output, dim=0)
    stack_label = torch.cat(stack_label, dim=0)
    loss = classifier.calc_loss(stack_output, stack_label).item()
    micro_acc, macro_acc = classifier.calc_acc(stack_output, stack_label)
    return loss, micro_acc, macro_acc, total_matching_loss

def finetuning(args, gnn, classifier, target_data):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ml = list()
    ml.extend(gnn.get_parameters())
    ml.extend(classifier.get_parameters())
    optimizer = optim.Adam(ml, lr=args.lr)

    min_loss = np.inf
    patient = 0
    #print("Training starts...")
    for epoch in range(args.epochs):
        for batch in range(args.train_batch2 - args.val_batch2):
            target_feats, target_adjs, target_labels, target_ids = target_data[batch]
            target_feats, target_adjs, target_labels, target_ids = move(device, target_feats, target_adjs, target_labels, target_ids)

            gnn.train()
            classifier.train()
            optimizer.zero_grad()

            target_message, target_cmd = gnn(target_feats, target_adjs)
            transform_message = gnn.transform(target_message[args.target_node][target_ids])
            target_output = classifier(transform_message)
            loss = classifier.calc_loss(target_output, target_labels[target_ids])
            loss.backward()
            optimizer.step()

        loss, micro_acc, macro_acc, _ = evaluate(gnn, classifier, args.train_batch2 - args.val_batch2, args.val_batch2, args.target_node, target_data)
        if loss >= min_loss:
            patient = patient + 1
        else:
            patient = 0
            min_loss = loss
        if patient == args.early_stopping:
            break
        #if epoch % 10 == 0:
        #    print("{}th epoch".format(epoch))
        #    print("Loss: ", loss, ",\t Micro acc: ", micro_acc, ",\t Macro acc: ", macro_acc)

    loss, micro_acc, macro_acc, _ = evaluate(gnn, classifier, args.train_batch2, args.test_batch2, args.target_node, target_data)
    #print("Evaluation result....")
    #print("Loss: ", loss, ",\t Micro acc: ", micro_acc, ",\t Macro acc: ", macro_acc)
    return loss, micro_acc, macro_acc


def scratch(args, types, relations, feat_dims, label_dim, target_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnn = HGNN(device, args.source_node, args.target_node, args.matching_loss, args.matching_hops, types, relations, feat_dims, \
            args.hid_dim, args.rel_dim, args.step_num).to(device)
    classifier = Classifier(args.hid_dim, label_dim[args.target_node][args.target_task], args.ranking).to(device)

    #print(">>>>>>>>>Target accuracy from scratch<<<<<<<<<<")
    return finetuning(args, gnn, classifier, target_data)


def pretrain(args, types, relations, feat_dims, label_dim, source_data, target_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnn = HGNN(device, args.source_node, args.target_node, args.matching_loss, args.matching_hops, types, relations, feat_dims, \
            args.hid_dim, args.rel_dim, args.step_num).to(device)
    classifier = Classifier(args.hid_dim, label_dim[args.source_node][args.source_task], args.ranking).to(device)

    ml = list()
    ml.extend(gnn.get_parameters())
    ml.extend(classifier.get_parameters())
    optimizer = optim.Adam(ml, lr=args.lr)


    min_loss = np.inf
    patient = 0
    #print("Pre-training starts...")
    for epoch in range(args.epochs):
        for batch in range(args.train_batch - args.val_batch):
            source_feats, source_adjs, source_labels, source_ids = source_data[batch]
            source_feats, source_adjs, source_labels, source_ids = move(device, source_feats, source_adjs, source_labels, source_ids)

            gnn.train()
            classifier.train()
            optimizer.zero_grad()

            # Pretrain on source computation graphs
            message, cmd = gnn(source_feats, source_adjs)
            output = classifier(message[args.source_node][source_ids])
            loss = classifier.calc_loss(output, source_labels[source_ids])

            # Matching loss
            h_S = message[args.source_node] #.clone().detach().to(device)
            h_T = message[args.target_node] #.clone().detach().to(device)
            matching_loss = gnn.get_matching_loss(source_adjs, h_S, h_T)

            # Gradient update
            loss = loss + args.matching_coeff * matching_loss
            #loss = loss + cmd
            loss.backward()
            optimizer.step()

         # Evaluation
        loss, micro_acc, macro_acc, match_loss = evaluate(gnn, classifier, args.train_batch - args.val_batch, args.val_batch, args.source_node, source_data)
        if loss >= min_loss:
            patient = patient + 1
        else:
            patient = 0
            min_loss = loss
        if patient == args.early_stopping:
            break
        """
        if epoch % 5 == 0:
            print("{}th epoch".format(epoch))
            print("[SOURCE] Loss: ", loss, ",\t Micro acc: ", micro_acc, ",\t Macro acc: ", macro_acc, ",\t Matching loss: ", match_loss)
            if args.source_task == args.target_task:
                target_loss, target_micro_acc, target_macro_acc, _ = evaluate(gnn, classifier, args.train_batch2, args.test_batch2, args.target_node, target_data)
                print("[TARGET] Loss: ", target_loss, ",\t Micro acc: ", target_micro_acc, ",\t Macro acc: ", target_macro_acc)
        """
    #print(">>>>>>>>>Source accuracy before fine-tuning<<<<<<<<<<")
    src_loss, src_micro_acc, src_macro_acc, src_match_loss = evaluate(gnn, classifier, args.train_batch, args.test_batch, args.source_node, source_data)
    #print("Loss: ", src_loss, ",\t Micro acc: ", src_micro_acc, ",\t Macro acc: ", src_macro_acc, ",\t Matching loss: ", src_match_loss)

    del source_data
    torch.cuda.empty_cache()

    if args.source_task != args.target_task:
        classifier = Classifier(args.hid_dim, label_dim[args.target_node][args.target_task], args.ranking).to(device)

    #print(">>>>>>>>>Target accuracy before fine-tuning<<<<<<<<<<")
    trg_loss1, trg_micro_acc1, trg_macro_acc1, _ = evaluate(gnn, classifier, args.train_batch2, args.test_batch2, args.target_node, target_data)
    #print("Loss: ", trg_loss1, ", Micro acc: ", trg_micro_acc1, ", Macro acc: ", trg_macro_acc1)

    #print(">>>>>>>>>Target accuracy after fine-tuning<<<<<<<<<<")
    #trg_loss2, trg_micro_acc2, trg_macro_acc2 = finetuning(args, gnn, classifier, target_data)

    return src_loss, src_micro_acc, src_macro_acc, src_match_loss, \
        trg_loss1, trg_micro_acc1, trg_macro_acc1

