import os

IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "list_attr_celeba.txt"

def __is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def __find_images_and_annotation(dir):
    images = {}
    attr = None
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if __is_image(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lower() == ATTR_ANNO:
                attr = os.path.join(root, fname)

    assert attr is not None, "Failed to find `list_attr_celeba.txt`"
    final = []
    with open(attr, "r") as fin:
        image_total = 0
        attrs = []
        for i_line, line in enumerate(fin):
            line = line.strip()
            if i_line == 0:
                image_total = int(line)
            elif i_line == 1:
                attrs = line.split(" ")
            else:
                line = line.split(" ")
                fname = os.path.splitext(line[0])[0]
                onehot = [int(int(d) > 0) for d in line[1:]]
                assert len(onehot) == len(attrs), "{} only has {} attrs < {}".format(
                    fname, len(onehot), len(attrs))
        final.append({
            "path": images[fname],
            "attr": onehot
        })
    return final, attrs
