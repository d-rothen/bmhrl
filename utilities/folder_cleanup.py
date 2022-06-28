import os
import pathlib
import glob
import shutil

def del_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print("Error: %s : %s" % (folder, e.strerror))

def deleteNonPyFiles(parent_dir):
    is_train_dir_kw = 'events.out.tfevents.'
    total = 0
    for tp in next(os.walk(parent_dir))[1]:
        folder = os.path.join(parent_dir, tp)
        path = os.path.join(folder, "*")
        
        files = glob.glob(path)
        if len(files) == 1:
            filename = os.path.basename(files[0])
            if(filename.startswith(is_train_dir_kw)):
                total += 1
                #print(folder)
                del_folder(folder)
                #print(path)
    print(total)
deleteNonPyFiles('./log/train_rl_cap')