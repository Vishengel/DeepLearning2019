import os
import sqlite3
import urllib.request
from urllib.error import HTTPError

def get_urls_from_database():
    # Set this to how you named the database in the data folder
    database_name = "bam1m.sqlite"

    # Get the path of the sqlite database using the os package so it works on any OS
    dir_path = os.path.dirname(os.path.realpath(__file__))
    database = os.path.join(dir_path, os.pardir, "data", database_name)

    # Connect to the database
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Get the urls of the images with the label 'watercolor' from the table with human-determined labels
    # BAM sqlite database docs: https://gist.github.com/gcr/c0e13bd205ed593f022ae0ad863e4ee2
    query = "select src from modules, crowd_labels where modules.mid = crowd_labels.mid and attribute = 'media_watercolor' and label='positive' limit 100;"

    cursor.execute(query)
    url_list = cursor.fetchall()

    conn.close()

    # cursor.fetchall() returns a list of tuples. The first element of each tuple contains the image url
    return [url[0] for url in url_list]

def download_imgs_from_urls(url_list):
    img_counter = 0
    error_counter = 0

    # Get the destination folder for the watercolor imgs
    dir_path = os.path.dirname(os.path.realpath(__file__))
    watercolor_img_folder = os.path.join(dir_path, os.pardir, "data", "watercolor_imgs")

    for idx in range(len(url_list)):
        file_destination = os.path.join(watercolor_img_folder, str(img_counter) + ".png")
        try:
            urllib.request.urlretrieve(url_list[idx], file_destination)
            img_counter += 1

            # Keep track of progress
            if img_counter % 10 == 0:
                print("Images downloaded: ", img_counter)
        except HTTPError:
            error_counter += 1

    print("Total amount of images downloaded: ", img_counter)
    print("Total amount of images not found: ", error_counter)

def main():
    url_list = get_urls_from_database()
    download_imgs_from_urls(url_list)

if __name__ == "__main__":
    main()