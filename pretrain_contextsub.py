import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tensorboardX import SummaryWriter

# from tensorboardX import SummaryWriter

# from model import GNN
from .model import GNN
from .loader import MoleculeDataset
from .dataloader import (
    # DataLoaderSubstructContext,
    ExtractPubchemSubstructs,
    DataLoaderPubchemContext,
)


def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()


def sqrt_norm(tensor):
    sqrt = torch.sqrt(torch.sum(torch.square(tensor), dim=1, keepdim=True))
    return tensor / sqrt


def train(
    args,
    model_substruct,
    model_context,
    loader,
    optimizer_substruct,
    optimizer_context,
    device,
    epoch,
    logger,
):
    model_substruct.train()
    model_context.train()

    if args.norm_output:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        # creating substructure representation
        substruct_rep = model_substruct(
            batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct
        )[batch.center_substruct_idx]
        substruct_rep = pool_func(
            substruct_rep, batch.batch_center_substruct, mode=args.context_pooling
        )
        if args.norm_output:
            substruct_rep = sqrt_norm(substruct_rep)

        # creating context representations
        overlapped_node_rep = model_context(
            batch.x_context, batch.edge_index_context, batch.edge_attr_context
        )[batch.overlap_context_substruct_idx]

        # Contexts are represented by
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(
                overlapped_node_rep,
                batch.batch_overlapped_context,
                mode=args.context_pooling,
            )
            if args.norm_output:
                context_rep = sqrt_norm(context_rep)

            # negative contexts are obtained by shifting the indicies of context
            # embeddings
            neg_context_rep = torch.cat(
                [
                    context_rep[cycle_index(len(context_rep), i + 1)]
                    for i in range(args.neg_samples)
                ],
                dim=0,
            )
            if args.norm_output:
                neg_context_rep = sqrt_norm(neg_context_rep)

            pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
            pred_neg = torch.sum(
                substruct_rep.repeat((args.neg_samples, 1)) * neg_context_rep, dim=1
            )
            if args.norm_output:
                # change the value range from [-1, 1] to [0, 1]
                pred_pos = 0.5 * (pred_pos + 1.0)
                pred_neg = 0.5 * (pred_neg + 1.0)

        elif args.mode == "skipgram":

            expanded_substruct_rep = torch.cat(
                [
                    substruct_rep[i].repeat((batch.overlapped_context_size[i], 1))
                    for i in range(len(substruct_rep))
                ],
                dim=0,
            )
            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim=1)

            # shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = substruct_rep[
                    cycle_index(len(substruct_rep), i + 1)
                ]
                shifted_expanded_substruct_rep.append(
                    torch.cat(
                        [
                            shifted_substruct_rep[i].repeat(
                                (batch.overlapped_context_size[i], 1)
                            )
                            for i in range(len(shifted_substruct_rep))
                        ],
                        dim=0,
                    )
                )

            shifted_expanded_substruct_rep = torch.cat(
                shifted_expanded_substruct_rep, dim=0
            )
            pred_neg = torch.sum(
                shifted_expanded_substruct_rep
                * overlapped_node_rep.repeat((args.neg_samples, 1)),
                dim=1,
            )

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(
            pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double()
        )
        loss_neg = criterion(
            pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double()
        )

        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples * loss_neg
        loss.backward()
        # To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        logger.add_scalar(
            "train_loss_step",
            loss.detach().cpu().item(),
            len(loader) * (epoch - 1) + step + 1,
        )

        balanced_loss_accum += float(
            loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item()
        )
        if args.norm_output:
            threshold = 0.5
        else:
            threshold = 0
        acc_accum += 0.5 * (
            float(torch.sum(pred_pos > threshold).detach().cpu().item()) / len(pred_pos)
            + float(torch.sum(pred_neg < threshold).detach().cpu().item())
            / len(pred_neg)
        )

    return balanced_loss_accum / (step + 1), acc_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--csize", type=int, default=3, help="context size (default: 3)."
    )
    parser.add_argument(
        "--emb_dim", type=int, default=300, help="embedding dimensions (default: 300)"
    )
    parser.add_argument(
        "--norm_output",
        action="store_true",
        help="normalize the GNN output to unit vectors before applying dot "
        "multiplication",
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--neg_samples",
        type=int,
        default=1,
        help="number of negative contexts per positive context (default: 1)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features are combined across layers. last, sum, max or "
        "concat",
    )
    parser.add_argument(
        "--context_pooling",
        type=str,
        default="mean",
        help="how the contexts are pooled (sum, mean, or max)",
    )
    parser.add_argument("--mode", type=str, default="cbow", help="cbow or skipgram")
    parser.add_argument(
        "--dataset",
        type=str,
        default="zinc_standard_agent",
        help="root directory of dataset for pretraining",
    )
    parser.add_argument(
        "--output_model_file", type=str, default="", help="filename to output the model"
    )
    parser.add_argument(
        "--logpath", type=str, default="", help="path for tensorboard log"
    )
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for splitting dataset."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers for dataset loading",
    )
    parser.add_argument(
        "--partial_charge",
        action="store_true",
        help="If to use atom partial charge as property.",
    )
    parser.add_argument(
        "--input_mlp",
        action="store_true",
        help="use MLP instead of Embedding layer for input",
    )
    parser.add_argument(
        "--node_feat_dim",
        type=int,
        default=3,
        help="the dimension of the node features",
    )
    parser.add_argument(
        "--edge_feat_dim", type=int, default=2, help="dimension of the edge features"
    )
    parser.add_argument(
        "--pattern_path",
        type=str,
        default="contextSub/resources/pubchemFPKeys_to_SMARTSpattern.csv",
        help="path to the csv file saves substructure patterns.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize

    print(args.mode)
    print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))

    # set up dataset and transform function.
    dataset = MoleculeDataset(
        "contextSub/dataset/" + args.dataset,
        dataset=args.dataset,
        transform=ExtractPubchemSubstructs(
            args.num_layer, l1, l2, partial_charge=args.partial_charge
        ),
        partial_charge=args.partial_charge,
        pattern_path=args.pattern_path,
    )
    loader = DataLoaderPubchemContext(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # set up models, one for pre-training and one for context embeddings
    model_substruct = GNN(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
        partial_charge=args.partial_charge,
        input_mlp=args.input_mlp,
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
    ).to(device)
    model_context = GNN(
        int(l2 - l1),
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
        partial_charge=args.partial_charge,
        input_mlp=args.input_mlp,
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
    ).to(device)

    # set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(
        model_substruct.parameters(), lr=args.lr, weight_decay=args.decay
    )
    optimizer_context = optim.Adam(
        model_context.parameters(), lr=args.lr, weight_decay=args.decay
    )
    scheduler_substruct = optim.lr_scheduler.StepLR(
        optimizer_substruct, step_size=50, gamma=0.1
    )
    scheduler_context = optim.lr_scheduler.StepLR(
        optimizer_context, step_size=50, gamma=0.1
    )

    if os.path.exists(args.logpath):
        shutil.rmtree(args.logpath)
    writer = SummaryWriter(args.logpath)
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(
            args,
            model_substruct,
            model_context,
            loader,
            optimizer_substruct,
            optimizer_context,
            device,
            epoch,
            writer,
        )
        scheduler_substruct.step()
        scheduler_context.step()

        print()
        print(f"train loss: {train_loss}, train acc: {train_acc}")
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("substruct_lr", scheduler_substruct.get_last_lr(), epoch)
        writer.add_scalar("context_lr", scheduler_context.get_last_lr(), epoch)

    if not args.output_model_file == "":
        os.makedirs(os.path.dirname(args.output_model_file), exist_ok=True)
        torch.save(model_substruct.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10,2)
    main()
