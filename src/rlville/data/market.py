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
import pandas as pd
import requests

from bs4 import BeautifulSoup
from bs4.element import Tag
from matplotlib import colors as mpcolors, image as mpimage


@click.group()
def main():
    pass


@main.command()
def download():
    root = "https://www.avemariasongs.org/games/FarmVille/"

    # Prepare the target
    os.makedirs("./data/images/seeds/", exist_ok=True)

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
        filepath = f"./data/images/seeds/{slug}.jpg"
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
    with open("./data/processed/market.tsv", "w+") as target:
        target.write("\n".join(lines))


@main.command()
def process() -> None:
    """
    Process the HTML table in the "FV-crops.htm" link. NOTE: I did a bunch of
    manual massaging to get the data quickly. Need to implement this a little
    more cleanly.
    """
    def html_table_to_rows(soup: BeautifulSoup, has_header: bool = True) -> tuple[list[tuple], tuple | None]:
        re_multi_whitespace = re.compile("\s{2,}")

        lines = (line.text.strip() for line in soup.find_all("tr"))
        rows = [tuple(re_multi_whitespace.split(line))[:8] for line in lines]
        header = rows.pop() if has_header else None

        return rows, header

    with open("./data/raw/crops.html", "r") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    data, columns = html_table_to_rows(soup)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("./data/processed/market.csv")


if __name__ == "__main__":
    main()
