import os
from PIL import Image
import sys

def resize_imgs(source_dir, target_dim):
    target_w = target_dim
    target_h = target_dim

    target_dir_name = str(target_dim) + "x" + str(target_dim)

    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    source_dir_path = os.path.join(cur_dir_path, os.pardir, "data", source_dir)
    target_dir_path = os.path.join(source_dir_path, target_dir_name)

    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)

    with os.scandir(source_dir_path) as entries:
        for imageFile in entries:
            if (imageFile.name.endswith(".png") or imageFile.name.endswith(".jpg")):
                im = Image.open(os.path.join(source_dir_path, imageFile.name))
                im = im.resize((target_w, target_h), Image.ANTIALIAS)
                im.save(os.path.join(target_dir_path, imageFile.name))

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Provide the the name of the image source directory and the target dimension size as arguments"

    resize_imgs(str(sys.argv[1]), int(sys.argv[2]))