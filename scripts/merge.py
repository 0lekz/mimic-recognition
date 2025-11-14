import os
import shutil

# source (new images) and destination (old dataset)
src_dir = "dataset"
dst_dir = "../dataset"

# loop over emotion classes
for cls in ["happy", "sad", "neutral"]:
    src_cls_dir = os.path.join(src_dir, cls)
    dst_cls_dir = os.path.join(dst_dir, cls)

    # make sure destination class dir exists
    os.makedirs(dst_cls_dir, exist_ok=True)

    # copy images, avoid overwriting by renaming
    for fname in os.listdir(src_cls_dir):
        src_path = os.path.join(src_cls_dir, fname)
        base, ext = os.path.splitext(fname)
        i = 0
        new_fname = fname
        while os.path.exists(os.path.join(dst_cls_dir, new_fname)):
            i += 1
            new_fname = f"{base}_{i}{ext}"
        shutil.copy(src_path, os.path.join(dst_cls_dir, new_fname))

print("Datasets merged successfully.")
