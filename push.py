import os
from huggingface_hub import HfApi, HfFolder
from pathlib import Path

repository_id = "mt5-small-prompted-germanquad"
src_dir = "tmp/mt5-small-prompted-germanquad"

huggingface_token = HfFolder.get_token()
api = HfApi()

user = api.whoami(huggingface_token)

# change into src dir
os.chdir(src_dir)

for subdir, dirs, files in os.walk("."):
    for file in files:
        full_path = os.path.join(subdir, file)[len(".") + 1 :]
        print(os.path.join(os.getcwd(), full_path))
        print(full_path)
        try:
            api.upload_file(
                token=huggingface_token,
                repo_id=f"{user['name']}/{repository_id}",
                path_or_fileobj=os.path.join(os.getcwd(), full_path),
                path_in_repo=full_path,
            )
        except KeyError:
            pass
        except NameError:
            pass
