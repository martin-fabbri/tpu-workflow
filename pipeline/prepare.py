from zipfile import ZipFile

from config import config

with ZipFile(config.DOWNLOADED_DATA) as zipOjb:
    zipOjb.extractall()
