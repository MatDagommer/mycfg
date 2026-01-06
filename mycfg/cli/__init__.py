"""Initialize the CLI module."""

from typer import Typer
from .train import train_model
from .test_config import test_config

from ..version import __version__

app = Typer(
    help="Command-line interface for mycfg.",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

app.command("train")(train_model)
app.command("test-config")(test_config)

@app.command()
def version():
    """Print the current version of pyml."""
    print(__version__)