# Tools for removing background

Tested on WSL in windows 10:

## Installed miniconda 

Download the installed script

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

sudo chmod +x Miniconda3-py39_4.12.0-Linux-x86_64.sh

sh ./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

## Create a development environement (dev for my example)

After installed miniconda exit and relaod a new console for using conda

```
conda create -n dev python=3.10

#for activate dev env :
conda activate dev

#For exited dev env :
conda deactivate devconda 
```

## Installed dependancy

Before running the python script for removing background, you must installed these dependancy
```
pip install --upgrade pip
pip install rembg pillow tqdm
pip install onnxruntime
```

If your PC have Nvidia GPU run these command :
```
pip install --upgrade rembg pillow tqdm onnxruntime
# or GPU NVIDIA :
# pip install --upgrade rembg pillow tqdm onnxruntime-gpu
```

note for uninstalled pip package :
```
pip uninstall -y onnxruntime
```

## create the python script

Create a directory "notebook"
enter in notebook and create a directory "remove_background"

```
mkdir ~/notebook
cd ~/notebook
mkdir remove_background
cd remove_background
```

create file the python script file : "remove_background_batch_iset_gen.py"

```
nano remove_background_batch_iset_gen.py
```

Copy and paste the code on the remove_background_batch_iset_gen.py :

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(input_dir: Path, recursive: bool):
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)

def output_path_for(input_file: Path, input_dir: Path, output_dir: Path, keep_structure: bool):
    if keep_structure:
        rel = input_file.relative_to(input_dir).with_suffix(".png")
        return (output_dir / rel).with_suffix(".png")
    else:
        return (output_dir / (input_file.stem + ".png"))

def process_one(input_file: Path, output_file: Path, session, force: bool = False, max_size: int | None = None):
    try:
        if output_file.exists() and not force:
            return (input_file, True, None)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_file) as im:
            if max_size:
                im.thumbnail((max_size, max_size), Image.LANCZOS)

            if im.mode != "RGBA":
                im = im.convert("RGBA")

            out = remove(im, session=session)
            if isinstance(out, (bytes, bytearray)):
                out_im = Image.open(BytesIO(out)).convert("RGBA")
            else:
                out_im = out.convert("RGBA")

            out_im.save(output_file, format="PNG", optimize=True)

        return (input_file, True, None)

    except Exception as e:
        return (input_file, False, str(e))

def main():
    parser = argparse.ArgumentParser(
        description="Retire l’arrière-plan de toutes les images d’un dossier et génère des PNG avec transparence."
    )
    parser.add_argument("input_dir", type=Path, help="Dossier d’entrée")
    parser.add_argument("output_dir", type=Path, help="Dossier de sortie")
    parser.add_argument("-r", "--recursive", action="store_true", help="Inclure les sous-dossiers")
    parser.add_argument("-k", "--keep-structure", action="store_true", help="Conserver l’arborescence")
    parser.add_argument("-f", "--force", action="store_true", help="Écraser les fichiers existants")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 4), help="Threads")
    parser.add_argument("--max-size", type=int, default=None, help="Redimension max avant traitement (px)")

    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        parser.error(f"Dossier d’entrée invalide: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Utilisation fixe du modèle isnet-general-use
    try:
        session = new_session("isnet-general-use")
    except Exception as e:
        print("⚠️ Impossible d’initialiser la session rembg avec isnet-general-use.")
        print(f"Détail: {e}")
        return

    files = list_images(args.input_dir, args.recursive)
    if not files:
        print("Aucune image trouvée.")
        return

    print(f"Trouvé {len(files)} image(s). Traitement…")

    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for f in files:
            out = output_path_for(f, args.input_dir, args.output_dir, args.keep_structure)
            tasks.append(ex.submit(process_one, f, out, session, args.force, args.max_size))

        ok = 0
        fails = 0
        with tqdm(total=len(tasks), unit="img") as bar:
            for fut in as_completed(tasks):
                _in, success, err = fut.result()
                if success:
                    ok += 1
                else:
                    fails += 1
                    tqdm.write(f"[ERREUR] {_in}: {err}")
                bar.update(1)

    print(f"Terminé. OK: {ok} | Échecs: {fails} | Sortie: {args.output_dir}")

if __name__ == "__main__":
    main()

```

## execute the script for removing background

Create a image directory with two sub directory for example "original" and "rmbg"

```
mkdir  ~/notebook/remove_background/images
mkdir  ~/notebook/remove_background/images/original
mkdir  ~/notebook/remove_background/images/rmbg

```
Copy some images in the "original" directory for testing.


Run the script for testing. 

Test with one image for the first run because the script download the model on the directory : .u2net
The models are :
 - isnet-general-use.onnx  
 - u2net.onnx

For our script the best models seams isnet-general-use.onnx.

The command to run the script is described below, and the processed images are in the rmbg folder.
```
conda activate dev
cd ~/notebook/remove_background
python remove_background_batch_iset_gen.py images/original/ images/rmbg/
```

***Note: I had a few WSL crashes because the script loads the ‘isnet-general-use.onnx’ model, which uses a lot of memory, and with a Docker that uses quite a few containers.
To get the script to work, I stopped Docker to free up some memory.***