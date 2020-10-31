import os
from utils.datasets import create_dataloader
import glob


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


hyp = check_file('data/hyp.scratch.yaml') 
dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=False, rect=False, local_rank=-1,
                                            world_size=1, mask=True)