import typer

import tomllib

from ..io import spatialdata_concat

app = typer.Typer()

@app.command()
def concatenate(
    config_file: str
):

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    sdata_paths = config.get("concatenate").get("file_paths")
    save_path = config.get("concatenate").get("save_path")

    spatialdata_concat(
        sdata_paths,
        save_path
    )

@app.command()
def _(
    input
):
    """Fake function to force Typer to 
    have sub-commands in the CLI when only one command is 
    added"""