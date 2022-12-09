import click



@click.command()
def download() -> None:
    """
    download the Crop Listing table from the url below into ./data/external
    """
    url = "https://www.avemariasongs.org/games/FarmVille/FV-crops.htm"
    ...
