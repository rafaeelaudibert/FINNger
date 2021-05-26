from threading import Thread
import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
import PySimpleGUI as sg
import click
from tqdm import tqdm
import matplotlib.pyplot as plt


from model import FINNger
from dataset import FINNgerDataset
from calculator import Calculator


from default_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_WEIGHT_DECAY,
)


key_pressed: bool = False


def detect_key_press():
    global key_pressed
    didnt_press_before = False

    while True:
        input("Press anything to take a screenshot and use this number in the calculator")
        key_pressed = True
        if didnt_press_before:
            print("Thank you!")
            didnt_press_before = True


def add_text(image, text, position, scale=1):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
    )


def test(network, test_loader, epoch='', should_save_test_acc=False):
    network.eval()
    test_loss = 0
    correct = 0

    if should_save_test_acc:
        f = open(f"data/input_{epoch}.txt", "w")

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Test validation"):
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            if should_save_test_acc:
                for tgt, prd in zip(target.data.view_as(pred), pred):
                    f.write(f"{tgt[0]} {prd[0]}\n")

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100. * correct / len(test_loader.dataset)):.0f}%)\n')

    # If we opened the file, we close it here
    if should_save_test_acc:
        f.close()

    return test_loss


def train_epoch(epoch_number, network, train_loader, save_id=''):
    losses, counter = [], []

    network.train()
    with tqdm(
            total=len(train_loader.dataset),
            desc=f'Train Epoch: 0 | Loss: 0'
    ) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            network.optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            network.optimizer.step()
            pbar.update(len(data))

            pbar.set_description_str(
                f'Train Epoch: {epoch_number} | Loss: {loss.item():.6f}')
            losses.append(loss.item())
            counter.append((batch_idx*DEFAULT_BATCH_SIZE) + ((epoch_number-1)
                                                             * len(train_loader.dataset)))
            network.save(save_id)

    return losses, counter


def train(network, n_epochs, train_loader, test_loader, model_id, should_save_test_acc=False):
    train_losses_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_losses.append(test(network, test_loader, -1, should_save_test_acc))
    for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs"):
        train_losses_counter.append(train_epoch(
            epoch,
            network,
            train_loader,
            save_id=model_id
        ))
        test_losses.append(
            test(network, test_loader, epoch, should_save_test_acc))

    train_losses, train_counter = list(map(
        lambda l: np.array(l).reshape(-1),
        zip(*train_losses_counter),
    ))

    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


@click.command()
@click.option('--train/--load', 'should_train', default=True, help='Should train or load model', show_default=True)
@click.option('--save_output', 'should_save_model_output', default=False, help="Should save the output from every validation epoch in a file", show_default=True)
@click.option("-i", "--model_id", help="The model id it should use to load/save")
@click.option("-e", "--epochs", default=5, help="How many epochs we will train the model for", show_default=True)
@click.option("-l", "--learning_rate", default=DEFAULT_LEARNING_RATE, help="The learning rate used to train the model", show_default=True)
@click.option("-w", "--weight_decay", default=DEFAULT_WEIGHT_DECAY, help="The weight decay rate used to train the model", show_default=True)
@click.option("--train_dataset", default=DEFAULT_TRAIN_DATASET, help="Regex used with `glob` to fetch the train dataset", show_default=True)
@click.option("--test_dataset", default=DEFAULT_TEST_DATASET, help="Regex used with `glob` to fetch the test dataset", show_default=True)
def main(
    model_id,
    epochs,
    learning_rate,
    weight_decay,
    should_train,
    train_dataset,
    test_dataset,
    should_save_model_output
):
    global key_pressed

    if not model_id:
        model_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"Model ID is {model_id}")
    print(f"Learning rate is {learning_rate}")

    model = FINNger(FINNgerDataset.NUM_CLASSES, learning_rate, weight_decay)

    # We Guarantee the 128x128 size, even though it probably already is
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((96, 96)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.CenterCrop((96, 96)),
            transforms.ToTensor(),
        ]),
    }

    if should_train:
        train_dataset = FINNgerDataset(train_dataset, data_transforms['train'])
        test_dataset = FINNgerDataset(test_dataset, data_transforms['test'])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=DEFAULT_BATCH_SIZE,
        )

        print(f"Starting to train for {epochs} epochs...")
        train(model, epochs, train_dataloader,
              test_dataloader, model_id, should_save_model_output)
        print("Finished training!")
    else:
        # We can load the model and test it
        model.load(model_id)

        # Initialize the enter detection thread to be used to display the values in the calculator
        thread = Thread(target=detect_key_press)
        thread.start()

        window = sg.Window(
            'FINNger',
            [[sg.Image(filename='', key='image')], ],
            location=(800, 400),
        )

        calculator = Calculator()
        cap = cv2.VideoCapture(0)  # Setup the camera as a capture device
        while True:  # The PSG "Event Loop"
            # get events for the window with 20ms max wait
            event, _values = window.Read(timeout=20, timeout_key='timeout')
            if event is None:  # if user closed window, quit
                break

            image = cap.read()[1]

            model_image = np.stack(
                (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),)*3,
                axis=-1
            )
            model_image = data_transforms['test'](model_image)
            model_image = model_image.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(model_image)
                pred = output.data.max(1, keepdim=True)[1]

            # This value is set by a background thread
            if key_pressed:
                calculator.add_number(pred[0][0].item())
                key_pressed = False

            # Update image in window
            image = np.array(model_image[0]).transpose((1, 2, 0)) * 255
            image = cv2.resize(image, (640, 480))

            add_text(image, f'Detecting: {pred[0][0]}', (10, 25))
            add_text(image, str(calculator), (10, 60))
            add_text(
                image,
                "[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                    *output.data.tolist()[0]),
                (10, 460),
                scale=0.4,
            )

            window_image = window.FindElement('image')
            encoded_image = cv2.imencode('.png', image)[1]
            window_image.Update(data=encoded_image.tobytes())


if __name__ == "__main__":
    main()
