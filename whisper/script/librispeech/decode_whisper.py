import sys, os
import re
import time
import argparse
import json

import torch
import whisper
import editdistance
from dataloader import get_dataloader
from whisper.normalizers.english import EnglishTextNormalizer

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--test_json', type=str, default="data/LibriSpeech/test_clean.json")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--beamsize', type=int, default=3)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--save_nbest', action="store_true")
parser.add_argument('--modeltype', type=str, default="base.en")
parser.add_argument('--normalise', action="store_true")
parser.add_argument('--logfile', type=str, default="")
parser.add_argument('--quantize_nf4', action="store_true")
parser.add_argument('--lora_mode', type=int, default=0)
args = parser.parse_args()


def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


logging(f"beamsize: {args.beamsize}", args.logfile)
logging(f"loadfrom: {args.loadfrom}", args.logfile)

normaliser = EnglishTextNormalizer()
logfile = args.logfile if args.logfile != "" else os.path.join(args.expdir, "log.txt")


if args.loadfrom != "":
    model = torch.load(args.loadfrom)
    model.eval()
else:
    model = whisper.load_model(args.modeltype).eval()
model.to(args.device)

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")

####################
# Data Loader
####################
testloader = get_dataloader(
    args.test_json,
    args.eval_batch_size,
    loadtarget=False,
    tokenizer=tokenizer,
    shuffle=False,
)

totalwords = 0
totalwer = 0
total_hyp = []
total_ref = []
nbest_dict = {}

print("Start of decoding")
start = time.time()
for idx, data in enumerate(testloader):
    uttnames, fbank, tgt = data
    fbank = fbank.to(model.device)

    options = whisper.DecodingOptions(
        language="en",
        without_timestamps=True,
        beam_size=args.beamsize,
        fp16=False,
    )
    result = whisper.decode(model, fbank, options)

    for i, utt in enumerate(tgt):
        uttname = uttnames[i]

        if args.normalise:
            text = normaliser(result[i].text).split()
            refwords = normaliser(utt.lower()).split()
        else:
            text = result[i].text.lower()
            text = re.sub("[^a-zA-Z0-9\' ]+", "", text).split()
            refwords = utt.lower().split()
        we = editdistance.eval(text, refwords)
        totalwords += len(refwords)
        totalwer += we
        fulltext = "{} ({})\n".format(' '.join(text), uttname)
        fullref = "{} ({})\n".format(normaliser(utt.lower()) if args.normalise else utt.lower(), uttname)
        total_hyp.append(fulltext)
        total_ref.append(fullref)
        if args.save_nbest:
            text_nbest = [text_nbest_i.lower() for text_nbest_i in result[i].text_nbest]
            text_nbest = [re.sub("[^a-zA-Z\' ]+", "", text_nbest_i) for text_nbest_i in text_nbest]
            sum_logprob_nbest = result[i].sum_logprob_nbest
            token_nbest = result[i].token_nbest
            nbest_dict[uttname] = [
                {"text": t, "token": token, "whisper_slp": slp}
                for t, slp, token in zip(text_nbest, sum_logprob_nbest, token_nbest)
            ]

    if idx % 10 == 0 and idx > 0:
        print("{} out of {} finished | time elapsed {}".format(idx, len(testloader), time.time()-start))
        print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))
        logging("{} out of {} finished | time elapsed {} | WER: {}".format(
            idx, len(testloader), time.time()-start, totalwer/totalwords), logfile)

print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))

with open(os.path.join(args.expdir, "hyp.wrd.trn"), "w") as fout:
    for line in total_hyp:
        fout.write(line + '\n')
with open(os.path.join(args.expdir, "ref.wrd.trn"), "w") as fout:
    for line in total_ref:
        fout.write(line + '\n')

if args.save_nbest:
    with open(os.path.join(args.expdir, "nbest.json"), "w") as fout:
        json.dump(nbest_dict, fout)
