"""Initialize the CLI module."""

from typer import Typer
from .train import train_model

app = Typer(help="Command-line interface for mycfg.")
app.command("train")(train_model)
