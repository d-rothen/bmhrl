import pandas as pd
import json
import csv 

charades_file = './data/CharadeCaptions/charades_captions.json'
captioning_file = './data/train.json'

def write_captions_csv():
    captioning_df = pd.read_json(captioning_file)
    charades_df = pd.read_json(charades_file)

    captioning_list = captioning_df.transpose().sentences.tolist()
    charades_list = charades_df.captions.tolist()
    combined_captions = [*captioning_list, *charades_list]

    header = ["caption"]

    with open('./data/combined_captions.csv', 'w') as csvfile:
        vid_id = 0
        seg_id = 0
        writer = csv.writer(csvfile)

        writer.writerow(header)

        for captions in combined_captions:
            for caption in captions:
                writer.writerow([caption.lstrip()])
                #f"{str(vid_id)} {caption.lstrip()} {seg_id}\n")
                #writer.writerow([str(vid_id), caption, seg_id])
                seg_id += 1
            vid_id += 1
        

def write_captions_json():
    captioning_df = pd.read_json(captioning_file)
    charades_df = pd.read_json(charades_file)

    captioning_list = captioning_df.transpose().sentences.tolist()
    charades_list = charades_df.captions.tolist()
    combined_captions = [*captioning_list, *charades_list]


    res = {}
    vid_id = 0
    for captions in combined_captions:
        res[str(vid_id)] = captions 
        vid_id += 1
    
    with open('./data/combined_captions.json', 'w') as cc:
        json.dump(res, cc)



write_captions_csv()
