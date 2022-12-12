"""
Download the Crop Listing table from the url below into "./data/external"

    "https://www.avemariasongs.org/games/FarmVille/FV-crops.htm"

Once that's done, this function extracts the table data into a dataframe
and saves it to "./data/processed".
"""

import os
import re
import string

import click
import numpy as np
import requests

from bs4 import BeautifulSoup
from bs4.element import Tag
from matplotlib import colors as mpcolors, image as mpimage

from rlville.constants import DATA


@click.group()
def main():
    pass


@main.command()
def download():
    root = "https://www.avemariasongs.org/games/FarmVille/"

    # Prepare the target
    os.makedirs("f{DATA}/images/seeds/", exist_ok=True)

    # An empty green pasture: (85 x 85) icon of 3-channel yellowgreen
    empty = np.full((85, 85, 3), fill_value=mpcolors.to_rgb("yellowgreen"))
    mpimage.imsave("./data/images/seeds/empty.jpg", empty)

    def sluggify(s: str):
        """
        Turns a string into a slug by replacing non-ASCII characters with whitespace
        and then replacing whitespace with dashes.

        :param s:
        :return:
        """
        slug, _ = re.subn(f"[{string.punctuation}]", " ", s)
        slug, _ = re.subn(r"\s+", "-", slug)
        return slug.lower()

    def get_row(tag: Tag):
        """
        Given an <img> tag pointing to the icon for the crop, walk up the tree
        until we have the row from the crops table. It's easiest to get the
        images we want than to get the table we want given the HTML of the page.

        :param tag:
        :return:
        """
        parent = tag.parent
        while parent.name != "tr":
            parent = parent.parent
        return parent.find_all("td")[:7]

    def download_image(src: str, tgt: str):
        """
        Download the image in the img tag, and name the file after the alt text.

        :param tag:
        :param name:
        :return:
        """
        r = requests.get(root + src, stream=True)

        if r.status_code == 200:
            with open(tgt, 'wb') as f:
                for chunk in r: f.write(chunk)

    # The root data source is this url for both the table and the images
    soup = BeautifulSoup(requests.get(root + "FV-crops.htm").text, "html.parser")

    # Get all seed file names from the sources of <img> tags
    images = soup.find_all("img", src=re.compile("images/seeds/FV-cro.*"))
    rows = map(get_row, images)

    # Download the images
    header = ("id", "icon", "name", "revenue", "cost", "growtime")
    noop = ("0", "./data/images/seeds/empty.jpg", "empty", "0", "0", "1")
    lines = ["\t".join(header), "\t".join(noop)]

    for i, row in enumerate(rows):
        image = row[0].select_one("img").attrs["src"]
        slug = sluggify(row[1].text.strip())
        filepath = f"{DATA}/images/seeds/{slug}.jpg"
        download_image(image, filepath)

        # Unpack the rest of the relevant data
        revenue, cost, growtime = (
            row[2].text.strip(),
            row[3].text.strip(),
            row[-1].text.strip(),
        )

        row = tuple(map(str, (i + 1, filepath, slug, revenue, cost, growtime)))
        click.echo(str(row))
        lines.append("\t".join(row))

    # Write the data to a tsv
    with open(f"{DATA}/processed/market.tsv", "w+") as target:
        target.write("\n".join(lines))


if __name__ == "__main__":
    main()
