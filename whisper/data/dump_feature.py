import sys, os
import json

import torch
import whisper


setpath = "/home/zhaoqiuming_p/project/whisper/data"

setname = "train_clean_360_10spk"
# dsets = ["train", "dev", "test"]
# dsets = ["train"]
# dsets = ["dev"]
dsets = ["test"]
tokenizer = whisper.tokenizer.get_tokenizer(True, language="en")

features = {}
for speaker in os.listdir(setname):
    spkpath = os.path.join(setname, speaker)
    for dset in dsets:
        features = {}
        fullpath = os.path.join(spkpath, dset)
        with open(os.path.join(fullpath, "{}.trans.txt".format(dset))) as fin:
            for line in fin:
                uttname = line.split()[0]
                print(uttname)
                utt = " " + ' '.join(line.split()[1:])
                utttokens = tokenizer.encode(utt.lower())
                audiopath = os.path.join(setpath, fullpath, "{}.flac".format(uttname))
                audio = whisper.load_audio(audiopath)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio)
                dumppath = os.path.join(setpath, fullpath, "{}_fbank.pt".format(uttname))
                torch.save(mel, dumppath)
                datapiece = {"fbank": dumppath, "words": utt}
                features[uttname] = datapiece

        with open(os.path.join(setpath, spkpath) + f"/{speaker}.{dset}.json", "w") as fout:
            json.dump(features, fout, indent=4)
