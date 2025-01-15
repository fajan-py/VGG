from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Callable

from typing import Callable

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epoch_number: int = 0,
    device: str = "cuda",
) -> None:
    """
     Trains the model for a single epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader
                                                  containing training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (Callable[torch.Tensor, torch.Tensor]): Loss function.
        epoch_number (int, optional): The current epoch number. Defaults to 0.
        device (str, optional): The device to use for computation.
                 Defaults to 'cuda'.

    """
    dataloader_iterator = tqdm(dataloader, colour="green", leave=True)

    model.train()
    model.to(device)

    for batch, (data, targets) in enumerate(dataloader_iterator):
        data = data.to(device)
        targets = targets.to(device)

        loss = perform_training_step(model, optimizer, loss_fn, data, targets)

        dataloader_iterator.set_description(
            f"[EPOCH {epoch_number}]"
        )
        dataloader_iterator.set_postfix(batch_loss=loss.item())


def perform_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Performs a single training step for the model."""
    predictions = model(data)
    if isinstance(loss_fn, nn.BCELoss):
        loss = loss_fn(predictions, targets.unsqueeze(1).float())
    else:
        loss = loss_fn(predictions, targets.long())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss