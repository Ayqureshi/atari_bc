from time import perf_counter
from argparse import ArgumentParser
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.dataset import CustomDataset
from torch.utils.data import DataLoader

# from utils.atari_dataloader import MultiprocessAtariDataLoader
# from utils.atari_head_dataloader import MultiprocessAtariHeadDataLoader

from utils.bc import Mnih2015

if __name__ == "__main__":
    parser = ArgumentParser("Train PyTorch models to do imitation learning.")
    # parser.add_argument("input_directory", type=str,
    #                     help="Path to directory with recorded gameplay.")
    # parser.add_argument("game", type=str,
    #                     help="Name of the game to use for training.")
    parser.add_argument("model", nargs="?", type=str,
                        help="Path of the file where model will be saved.") 
    parser.add_argument("--actions", type=int, default=6,
                        help="Number of actions")       
    parser.add_argument("--framestack", type=int, default=4,
                        help="Number of frames to stack")
    parser.add_argument("--merge", action="store_true",
                        help="Merge stacked frames into one image.")
    parser.add_argument("--width", "-x", type=int, default=84,
                        help="Width of the image")
    parser.add_argument("--height", "-y", type=int, default=84,
                        help="Height of the image")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes to use for the dataloader.")
    parser.add_argument("--l2", type=float, default="0.00001",
                        help="L2 regularization weight.")
    parser.add_argument("--percentile", type=int,
                        help="The top q-percentile of samples to use for training.")
    parser.add_argument("--top-n", type=int,
                        help="The top n number of samples to use for training.")
    parser.add_argument("--save-freq", type=int, default=1,
                        help="Number of epochs between checkpoints.")
    parser.add_argument("--augment", action="store_true",
                        help="Use image augmentation.")
    parser.add_argument("--preload", action="store_true",
                        help="Preload image data to memory.")
    parser.add_argument("--atari-head", action="store_true",
                        help="Use the Atari-HEAD dataloader.")
    parser.add_argument("--action-delay", type=int, default=0,
                        help="How many frames to delay the actions by.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Don't use CUDA")
    parser.add_argument("--json", action="store_true",
                        help="Dataset is stored as JSON")
    parser.add_argument("--fileformat", type=str, default="png",
                        help="Postfix of the image files to be loaded")
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Directory of datasets")
    parser.add_argument("--use_wb", action="store_true",
                        help="Use wandb for logging.")
    parser.add_argument("--wandb_project_name", type=str, default = "atari_bc",
                        help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default = "matthewh6",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--run_name", type=str, default = "bc",
                        help="the name of the run")
    parser.add_argument("--num_trajs", type=int, default = 500,
                        help="number of trajectories to use for training")
    parser.add_argument("--seed", type=int, default = 10,
                        help="seed")

    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #import ipdb; ipdb.set_trace()

    #random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Mnih2015(
        (args.width, args.height),
        1*args.framestack,
        args.actions
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2)
    
    # dataloader_args = {
    #     "directory": args.input_directory,
    #     "game": args.game,
    #     "stack": args.framestack,
    #     "batch_size": args.batch,
    #     "size": (args.width, args.height),
    #     "percentile": args.percentile,
    #     "top_n": args.top_n,
    #     "augment": args.augment,
    #     "preload": args.preload,
    #     "merge": args.merge,
    #     "json": args.json,
    #     "action_delay": args.action_delay,
    #     "fileformat": args.fileformat
    # }

    # Note: if new dataloader arguments are added, make sure they work with
    #       both loaders, or if they don't, remove them with 'del' below
    # if args.atari_head:
    #     del dataloader_args["game"]
    #     del dataloader_args["json"]
    #     del dataloader_args["fileformat"]
    #     gen = MultiprocessAtariHeadDataLoader(dataloader_args, args.workers)
    # else:
    #     gen = MultiprocessAtariDataLoader(dataloader_args, args.workers)
    
    if args.use_wb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("train/bc_loss", step_metric="epoch")
        wandb.define_metric("train/bc_accuracy", step_metric="epoch")



    data_file = args.data_dir + "/data.pkl"
    data = np.load(data_file, allow_pickle=True)
    #shape = data.shape

    history = dict()
    history["loss"] = []
    history["accuracy"] = []

    #dataset = CustomDataset(data["observations"], data["actions"])
    # 50 trajectories
    
    dataset = CustomDataset(data["observations"][:args.num_trajs], data["actions"][:args.num_trajs])
    #dataset = CustomDataset(data["observations"], data["actions"])
    
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        print("Starting epoch {}".format(epoch))
        model.train()
        start = perf_counter()

        # Accuracy
        correct = 0
        total = 0

        # Loss
        loss_sum = 0
        loss_num = 0

        for batch, data in enumerate(dataloader):
            # Convert data to correct format
            # 128 x 1 x 4 x 84 x 84 -> 128 x 4 x 84 x 84
            x = torch.Tensor(data[0].squeeze()).to(device) / 255
            # 128 x 1 -> 128
            y = torch.Tensor(data[1]).to(device)[:, 0]
            
            optimizer.zero_grad()

            # Get model output
            output = model(x)

            # Calculate loss
            loss = F.cross_entropy(output, y)

            # Add loss to epoch statistics
            loss_sum += loss
            loss_num += 1

            # Calculate accuracy and add to epoch statistics
            correct += output.argmax(1).eq(y).sum()

            total += len(y)
            
            # Backpropagate loss
            loss.backward()
            optimizer.step()

            # Print statistics
            if batch % 100 == 0:
                end = perf_counter()
                accuracy = float(correct) / float(total)
                loss = loss_sum / loss_num
                print("Epoch {} - {}/{}: loss: {}, acc: {} ({} s/batch)".format(
                    epoch,
                    batch,
                    len(dataset)//args.batch,
                    loss,
                    accuracy,
                    (end - start) / 100)
                )
                start = perf_counter()

        # Save statistics
        accuracy = float(correct) / float(total)
        loss = loss_sum / loss_num

        # history["accuracy"].append(float(accuracy))
        # history["loss"].append(float(loss))


        # with open(args.model + "-history.json", "w") as f:
        #     json.dump(history, f)

        if args.use_wb:
            wandb.log({
                "train/bc_accuracy": accuracy,
                "train/bc_loss": loss,
                "epoch": epoch,
            }, step=epoch)

        print("Finished epoch {}".format(epoch))
        # Save model
    if args.model is not None:
        filename = "{}_bc.pt".format(args.run_name)
        print("Saving {}".format(filename))
        torch.save(model, filename)
    
    #gen.stop()