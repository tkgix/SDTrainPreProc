# Apache License 2.0

import argparse
import csv
import glob
import os
import shutil

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# from wd14 tagger
IMAGE_SIZE = 448

WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-convnext-tagger'
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]


def main(args):

  if not os.path.exists(args.model_dir) or args.force_download:
    print("downloading wd14 tagger model from hf_hub")
    for file in FILES:
      hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True, force_filename=file)
    for file in SUB_DIR_FILES:
      hf_hub_download(args.repo_id, file, subfolder=SUB_DIR, cache_dir=os.path.join(
          args.model_dir, SUB_DIR), force_download=True, force_filename=file)

  image_paths = glob.glob(os.path.join(args.dir_in, "*.jpg")) + \
      glob.glob(os.path.join(args.dir_in, "*.png")) + glob.glob(os.path.join(args.dir_in, "*.webp"))
  print(f"found {len(image_paths)} images.")

  print("loading model and labels")
  model = load_model(args.model_dir)

  with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    header = l[0]             # tag_id,name,category,count
    rows = l[1:]
  assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"unexpected csv format: {header}"

  tags = [row[1] for row in rows[1:] if row[2] == '0']      # categoryが0、つまり通常のタグのみ

  if args.dir_out is None :
    path_out = os.path.join( args.dir_out, "out" )  
  else:
    path_out = args.dir_out

  def run_batch(path_imgs):
    imgs = np.array([im for _, im in path_imgs])

    probs = model(imgs, training=False)
    probs = probs.numpy()

    arr_tags = args.tags.split(',')
    for i, s in enumerate(arr_tags):
      arr_tags[i] = s.strip()
      
    
    for (image_path, _), prob in zip(path_imgs, probs):
      b_move = False
      for i, p in enumerate(prob[4:]):
        if p >= args.thresh:
          s_tag = tags[i]            
          for idx_test, s_test in enumerate(arr_tags):
            if s_tag == s_test:
              if os.path.exists(os.path.join( path_out, s_test )) == False:
                os.makedirs( os.path.join( path_out, s_test ), exist_ok=True)

              temp_path_out = os.path.join( path_out , s_test )
              shutil.move(image_path, os.path.join( temp_path_out, os.path.basename(image_path))) 
              b_move = True
              break
        if b_move == True:
          break
      
  b_imgs = []
  for image_path in tqdm(image_paths, smoothing=0.0):
    img = Image.open(image_path)                  
    if img.mode != 'RGB':
      img = img.convert("RGB")
    img = np.array(img)
    img = img[:, :, ::-1]

    # pad to square
    size = max(img.shape[0:2])
    pad_x = size - img.shape[1]
    pad_y = size - img.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    img = np.pad(img, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    img = img.astype(np.float32)
    b_imgs.append((image_path, img))

    if len(b_imgs) >= args.batch_size:
      run_batch(b_imgs)
      b_imgs.clear()

  if len(b_imgs) > 0:
    run_batch(b_imgs)

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir_in", type=str, default=None, help="Directory for check", required=True)
  parser.add_argument("--tags", type=str, default="sketch,monochrome,no_humans", help="Tags for move")
  parser.add_argument("--thresh", type=float, default=0.1, help="threshold of confidence to add a tag")
  parser.add_argument("--dir_out", type=str, default=None, help="Root Directory to move")

  parser.add_argument("--repo_id", type=str, default=WD14_TAGGER_REPO, help="repo id for wd14 tagger on Hugging Face")
  parser.add_argument("--model_dir", type=str, default="wd14_tagger_model", help="directory to store wd14 tagger model")
  parser.add_argument("--force_download", action='store_true', help="force downloading wd14 tagger models")  
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference")      
  
  args = parser.parse_args()

  main(args)
