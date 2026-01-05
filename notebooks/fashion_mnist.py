import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from pyprojroot import here

    import torchvision
    import torch
    return torch, torchvision


@app.cell
def _(torchvision):
    torchvision.datasets.FashionMNIST(root="./data/", download=True)
    return


@app.cell
def _(torchvision):
    dataset = torchvision.datasets.FashionMNIST(root="./data/")
    return (dataset,)


@app.cell
def _(dataset):
    vars(dataset)
    return


@app.cell
def _(dataset):
    dataset.data.size()
    return


@app.cell
def _(dataset):
    import matplotlib.pyplot as plt

    image = dataset.data[0]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {dataset.targets[0]}")
    plt.axis('off')
    plt.gca()
    return


@app.cell
def _(dataset, torch):
    data_loader = torch.utils.data.DataLoader(
        dataset.data,
        batch_size=4,
        shuffle=True,
    )
    return (data_loader,)


@app.cell
def _(data_loader):
    data_loader
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
