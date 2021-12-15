import argparse
import os
import shutil
import json

from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    average_precision_score,
)
from tensorboardX import SummaryWriter
import pandas as pd

from .model import GNN_graphpred, ContextSubDouble
from .loader import MoleculeDataset
from .splitters import scaffold_split, random_split, random_scaffold_split
from .dataloader import DataLoaderPooling


criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizers, schedulers, epoch=None):
    model.train()

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
        )
        if args.freeze:
            if args.two_step_freeze:
                if epoch < args.freeze:
                    optimizers[2].zero_grad()
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                    loss.backward()
                    optimizers[2].step()
                elif args.freeze <= epoch < 2 * args.freeze:
                    [opt.zero_grad() for opt in optimizers[1:]]
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                    loss.backward()
                    [opt.step() for opt in [optimizers[0], optimizers[2]]]
                else:
                    [opt.zero_grad() for opt in optimizers]
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                    loss.backward()
                    [opt.step() for opt in optimizers]
            else:
                if epoch < args.freeze:
                    optimizers[1].zero_grad()
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                    loss.backward()
                    optimizers[1].step()
                else:
                    [opt.zero_grad() for opt in optimizers]
                    loss = torch.sum(loss_mat) / torch.sum(is_valid)
                    loss.backward()
                    [opt.step() for opt in optimizers]
        else:
            [opt.zero_grad() for opt in optimizers]
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss.backward()
            [opt.step() for opt in optimizers]

    # update lr after the whole epoch is done
    if args.freeze:
        if args.two_step_freeze:
            if epoch < args.freeze:
                schedulers[2].step()
            elif args.freeze <= epoch < 2 * args.freeze:
                [sch.step() for sch in [schedulers[0], schedulers[2]]]
            else:
                [sch.step() for sch in schedulers]
        else:
            if epoch < args.freeze:
                schedulers[1].step()
            else:
                [sch.step() for sch in schedulers]
    else:
        [sch.step() for sch in schedulers]


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(batch))

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    precision_list = []
    recall_list = []
    ap_list = []
    specificity_list = []
    f1_list = []
    acc_list = []
    for i in range(y_true.shape[1]):
        # metrics are only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            y_true_01 = (y_true[is_valid, i] + 1) / 2
            y_pred = np.round(y_scores[is_valid, i]).astype(int)
            roc_list.append(
                roc_auc_score(y_true_01, y_scores[is_valid, i], average="micro")
            )
            precision_list.append(precision_score(y_true_01, y_pred, average="binary"))
            recall_list.append(recall_score(y_true_01, y_pred, average="binary"))
            ap_list.append(average_precision_score(y_true_01, y_pred))
            specificity_list.append(
                recall_score((1 - y_true_01), (1 - y_pred), average="binary")
            )
            f1_list.append(f1_score(y_true_01, y_pred, average="binary"))
            acc_list.append(accuracy_score(y_true_01, y_pred))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    metrics = {
        "roc_auc": sum(roc_list) / len(roc_list),
        "precision": sum(precision_list) / len(precision_list),
        "recall": sum(recall_list) / len(recall_list),
        "average precision": sum(ap_list) / len(ap_list),
        "specificity": sum(specificity_list) / len(specificity_list),
        "f1": sum(f1_list) / len(f1_list),
        "accuracy": sum(acc_list) / len(acc_list),
    }

    return metrics


def parse_args():
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
        default=32,
        help="input batch size for training (default: 32)",
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
        "--lr_scale",
        type=float,
        default=1,
        help="relative learning rate for the feature extraction layer (default: 1)",
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
        "--emb_dim", type=int, default=300, help="embedding dimensions (default: 300)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        help="graph level pooling (sum, mean, max, set2set, attention)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features across layers are combined. last, sum, max or "
        "concat",
    )
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tox21",
        help="root directory of dataset. For now, only classification.",
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default="",
        help="filename to read the model (if there is any)",
    )
    parser.add_argument("--filename", type=str, default="", help="output filename")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for splitting the dataset."
    )
    parser.add_argument(
        "--runseed",
        type=int,
        default=0,
        help="Seed for minibatch selection, random initialization.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="scaffold",
        help="random or scaffold or random_scaffold",
    )
    parser.add_argument(
        "--eval_train", action="store_true", help="evaluating training or not"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataset loading",
    )
    parser.add_argument(
        "--partial_charge",
        action="store_true",
        help="to use partial charge atom property",
    )
    parser.add_argument(
        "--sub_level",
        action="store_true",
        help="GNN_graphpred model output substructure based embeddings and the model is"
        " fine-tuned based on that embedding.",
    )
    parser.add_argument(
        "--sub_input",
        action="store_true",
        help="Input graphs contain substructure information. For fine-tune only.",
    )
    parser.add_argument(
        "--context", action="store_true", help="The input is in context format"
    )
    parser.add_argument(
        "--pooling_indicator",
        action="store_true",
        help="data includes pooling indicator attribute",
    )
    parser.add_argument(
        "--separate_output",
        action="store_true",
        help="Separate the molecule embedding and substructrue embedding before feeding"
        " into the MLP classifier.",
    )
    parser.add_argument(
        "--contextpred",
        action="store_true",
        help="Use pretrained weights from contextPred to compute molecule level "
        "embedding, and use contextSub to compute substructure level embeddings.",
    )
    parser.add_argument(
        "--contextpred_model_file",
        type=str,
        help="Path to the pretrained contextPred weights file.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze the weigthts of GNN at the beggining of fine-tuning for n steps."
        " Default is 0.",
    )
    parser.add_argument(
        "--two_step_freeze",
        action="store_true",
        help="Use 2-step freezing strategy to fine-tune the model.",
    )
    parser.add_argument(
        "--use_lightbbb_test",
        action="store_true",
        help="If present, use the external testing set from LightBBB. "
        "Otherwise, generate testing set with scaffold splitting method.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Save finetuned model to given path.",
    )
    args = parser.parse_args()
    return args


def get_num_tasks(task):
    if task == "tox21":
        num_tasks = 12
    elif task == "hiv":
        num_tasks = 1
    elif task == "pcba":
        num_tasks = 128
    elif task == "muv":
        num_tasks = 17
    elif task == "bace":
        num_tasks = 1
    elif task == "bbbp":
        num_tasks = 1
    elif task == "toxcast":
        num_tasks = 617
    elif task == "sider":
        num_tasks = 27
    elif task == "clintox":
        num_tasks = 2
    elif task == "lightbbb":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")
    return num_tasks


def decide_split(dataset, args, pattern_path):
    if args.dataset == "lightbbb":
        if args.split == "scaffold":
            if args.use_lightbbb_test:
                print("scaffold")
                smiles_list = pd.read_csv(
                    "contextSub/dataset/" + args.dataset + "/processed/smiles.csv",
                    header=None,
                )[0].tolist()
                train_dataset, valid_dataset, _ = scaffold_split(
                    dataset,
                    smiles_list,
                    null_value=0,
                    frac_train=0.9,
                    frac_valid=0.1,
                    frac_test=0.0,
                )
                print("Use LightBBB testing set.")
            else:
                smiles_list = pd.read_csv(
                    "contextSub/dataset/" + args.dataset + "/processed/smiles.csv",
                    header=None,
                )[0].tolist()
                train_dataset, valid_dataset, test_dataset = scaffold_split(
                    dataset,
                    smiles_list,
                    null_value=0,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                )
                print("Not using LightBBB testing set.")
        elif args.split == "random":
            train_dataset, valid_dataset, _ = random_split(
                dataset,
                null_value=0,
                frac_train=0.9,
                frac_valid=0.1,
                frac_test=0.0,
                seed=args.seed,
            )
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv(
                "contextSub/dataset/" + args.dataset + "/processed/smiles.csv",
                header=None,
            )[0].tolist()
            train_dataset, valid_dataset, _ = random_scaffold_split(
                dataset,
                smiles_list,
                null_value=0,
                frac_train=0.9,
                frac_valid=0.1,
                frac_test=0.0,
                seed=args.seed,
            )
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        if args.use_lightbbb_test:
            test_dataset = MoleculeDataset(
                "contextSub/dataset/lightbbb_test",
                dataset="lightbbb_test",
                partial_charge=args.partial_charge,
                substruct_input=args.sub_input,
                pattern_path=pattern_path,
                context=args.context,
                hops=args.num_layer,
                pooling_indicator=args.pooling_indicator,
            )
    else:
        if args.split == "scaffold":
            smiles_list = pd.read_csv(
                "contextSub/dataset/" + args.dataset + "/processed/smiles.csv",
                header=None,
            )[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(
                dataset,
                smiles_list,
                null_value=0,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
            )
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset,
                null_value=0,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
                seed=args.seed,
            )
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv(
                "contextSub/dataset/" + args.dataset + "/processed/smiles.csv",
                header=None,
            )[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(
                dataset,
                smiles_list,
                null_value=0,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
                seed=args.seed,
            )
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

    print(train_dataset[0])
    return train_dataset, valid_dataset, test_dataset


def main():
    args = parse_args()
    arg_outpath = os.path.join("contextSub", "runs", args.filename, "args.txt")
    if not os.path.exists(arg_outpath):
        os.makedirs(os.path.dirname(arg_outpath))
        with open(arg_outpath, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = get_num_tasks(args.dataset)

    # set up dataset
    # if args.sub_input:
    #     pattern_path = os.path.join(
    #         "contextSub", "resources", "pubchemFPKeys_to_SMARTSpattern.csv"
    #     )
    # else:
    #     pattern_path = os.path.join(
    #         "contextSub", "resources", "pubchemFPKeys_to_SMARTSpattern_filtered.csv"
    #     )
    pattern_path = os.path.join(
        "contextSub", "resources", "pubchemFPKeys_to_SMARTSpattern_filtered.csv"
    )
    dataset = MoleculeDataset(
        "contextSub/dataset/" + args.dataset,
        dataset=args.dataset,
        partial_charge=args.partial_charge,
        substruct_input=args.sub_input,
        pattern_path=pattern_path,
        context=args.context,
        hops=args.num_layer,
        pooling_indicator=args.pooling_indicator,
    )

    print(dataset)
    train_dataset, valid_dataset, test_dataset = decide_split(
        dataset, args, pattern_path
    )

    if args.pooling_indicator:
        DataLoaderClass = DataLoaderPooling
    else:
        DataLoaderClass = DataLoader
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # set up model
    if args.contextpred:
        model = ContextSubDouble(
            args.num_layer,
            args.emb_dim,
            num_tasks,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            graph_pooling=args.graph_pooling,
            gnn_type=args.gnn_type,
            partial_charge=args.partial_charge,
        )
        model.from_pretrained(args.contextpred_model_file, args.input_model_file)
    else:
        model = GNN_graphpred(
            args.num_layer,
            args.emb_dim,
            num_tasks,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            graph_pooling=args.graph_pooling,
            gnn_type=args.gnn_type,
            partial_charge=args.partial_charge,
            sub_level=args.sub_level,
            separate=args.separate_output,
        )
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)

    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    gnn_param_group = []
    mlp_param_group = []
    if args.contextpred:
        gnn_param_group.append({"params": model.gnn1.parameters()})
        gnn_param_group.append({"params": model.gnn2.parameters()})
    else:
        gnn_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        gnn_param_group.append(
            {"params": model.pool.parameters(), "lr": args.lr * args.lr_scale}
        )
    mlp_param_group.append({"params": model.graph_pred_linear.parameters()})

    if args.freeze:
        print("freeze")
        if args.two_step_freeze:
            print("two step freeze")
            pred_optimizer = optim.Adam(
                gnn_param_group[:1], lr=args.lr, weight_decay=args.decay
            )
            sub_optimizer = optim.Adam(
                gnn_param_group[1:], lr=args.lr, weight_decay=args.decay
            )
            mlp_optimizer = optim.Adam(
                mlp_param_group, lr=args.lr, weight_decay=args.decay
            )
            pred_lr_scheduler = optim.lr_scheduler.StepLR(
                pred_optimizer, step_size=20, gamma=0.1
            )
            sub_lr_scheduler = optim.lr_scheduler.StepLR(
                sub_optimizer, step_size=20, gamma=0.1
            )
            mlp_lr_scheduler = optim.lr_scheduler.StepLR(
                mlp_optimizer, step_size=20, gamma=0.1
            )
            print(pred_optimizer)
            print(sub_optimizer)
            print(mlp_optimizer)
        else:
            print("one step freeze")
            gnn_optimizer = optim.Adam(
                gnn_param_group, lr=args.lr, weight_decay=args.decay
            )
            mlp_optimizer = optim.Adam(
                mlp_param_group, lr=args.lr, weight_decay=args.decay
            )
            gnn_lr_scheduler = optim.lr_scheduler.StepLR(
                gnn_optimizer, step_size=20, gamma=0.1
            )
            mlp_lr_scheduler = optim.lr_scheduler.StepLR(
                mlp_optimizer, step_size=20, gamma=0.1
            )
            print(gnn_optimizer)
            print(mlp_optimizer)
    else:
        print("no freeze")
        print(gnn_param_group)
        print(mlp_param_group)
        optimizer = optim.Adam(
            gnn_param_group + mlp_param_group, lr=args.lr, weight_decay=args.decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        print(optimizer)
    # train_acc_list = []
    # val_acc_list = []
    # test_acc_list = []

    if not args.filename == "":
        fname = os.path.join(
            "contextSub", "runs", args.filename, f"finetune_cls_runseed{args.runseed}"
        )
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs + 1):
        print(f"\n====epoch {epoch}====")

        if args.freeze:
            if args.two_step_freeze:
                train(
                    args,
                    model,
                    device,
                    train_loader,
                    [pred_optimizer, sub_optimizer, mlp_optimizer],
                    [pred_lr_scheduler, sub_lr_scheduler, mlp_lr_scheduler],
                    epoch,
                )
            else:
                train(
                    args,
                    model,
                    device,
                    train_loader,
                    [gnn_optimizer, mlp_optimizer],
                    [gnn_lr_scheduler, mlp_lr_scheduler],
                    epoch,
                )
        else:
            train(args, model, device, train_loader, [optimizer], [scheduler], epoch)

        print("\n====Evaluation====")
        if args.eval_train:
            train_metrics = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
        val_metrics = eval(args, model, device, val_loader)
        test_metrics = eval(args, model, device, test_loader)

        result_str = f"\nval: {val_metrics}\ntest: {test_metrics}\n"
        if args.eval_train:
            result_str = f"\ntrain: {train_metrics}" + result_str

        print(result_str)

        # val_acc_list.append(val_auc)
        # test_acc_list.append(test_auc)
        # train_acc_list.append(train_auc)

        if not args.filename == "":
            if args.eval_train:
                for metric, value in train_metrics.items():
                    writer.add_scalar(f"train/{metric}", value, epoch)
            for metric, value in val_metrics.items():
                writer.add_scalar(f"val/{metric}", value, epoch)
            for metric, value in test_metrics.items():
                writer.add_scalar(f"test/{metric}", value, epoch)

        print("")

    if args.save_model is not None:
        os.makedirs(args.save_model, exist_ok=True)
        if hasattr(model, "gnn"):
            torch.save(model.gnn.state_dict(), os.path.join(args.save_model, "gnn.pth"))
        else:
            torch.save(
                model.gnn1.state_dict(), os.path.join(args.save_model, "gnn1.pth")
            )
            torch.save(
                model.gnn2.state_dict(), os.path.join(args.save_model, "gnn2.pth")
            )

    if not args.filename == "":
        writer.close()


if __name__ == "__main__":
    main()
