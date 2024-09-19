import sys, os
import re
import time

import numpy as np
import random
import whisper
import editdistance
from dataloader import get_dataloader
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam
from transformers import WhisperTokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--modeltype', type=str, default="base.en")
parser.add_argument('--train_json', type=str, default="data/LibriSpeech/train_clean_100.json")
parser.add_argument('--dev_json', type=str, default="data/LibriSpeech/dev.json")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--decay_pct', type=float, default=1)
parser.add_argument('--warmup_pct', type=float, default=0.0)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--logfile', type=str, default="log")
parser.add_argument('--accumgrad', type=int, default=1)
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--device', type=str, default="cuda")

parser.add_argument('--n_expert', type=int, default=10)
parser.add_argument('--lora_alpha', type=int, default=64)
parser.add_argument('--lora_r', type=int, default=4)
parser.add_argument('--scheduler', type=str, default="warmuplr")
parser.add_argument('--is_pretrain', type=int, default=0)
parser.add_argument('--grad_mode', type=int, default=0)
parser.add_argument('--save_snapshot', action="store_true")

parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--lr2', type=float, default=0.00001)
parser.add_argument('--interleave_interval', type=int, default=1)
args = parser.parse_args()

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

# trainer
logging(f"batch_size: {args.batch_size}", args.logfile)

# lora
logging(f"is_pretrain: {args.is_pretrain}", args.logfile)
logging(f"lora_r: {args.lora_r}, lora_alpha: {args.lora_alpha}", args.logfile)
# scheduler
logging(f"scheduler: {args.scheduler}, lr: {args.lr}, lr2: {args.lr2}, max_epoch: {args.nepochs}, interleave_interval: {args.interleave_interval}", args.logfile)
if args.scheduler == "warmuplr":
    logging(f"warmup_pct: {args.warmup_pct}, decay_pct: {args.decay_pct}", args.logfile)
elif args.scheduler == "steplr":
    logging(f"gamma: {args.gamma}", args.logfile)
elif args.scheduler == "fixlr":
    pass
else:
    raise ValueError
# loadfrom
logging(f"loadfrom: {args.loadfrom}", args.logfile)
logging(f"temperature: {args.temperature}", args.logfile)

##################
# Model
##################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


if args.loadfrom != "":
    model = whisper.load_moe_model(args.loadfrom, args.is_pretrain, args.lora_r, args.lora_alpha, 0.2, args.temperature, args.device)
else:
    model = whisper.load_model(args.modeltype)

model.to(args.device)
model.train()

options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")
decodetask = whisper.decoding.DecodingTask(model, options)
logit_filters = decodetask.logit_filters
sot_sequence = decodetask.sot_sequence
sotlen = len(sot_sequence)

##################
# Lora
##################
# freeze
if args.grad_mode == 3:
    for n, p in model.named_parameters():  # router and lora
        if "router" in n or "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
elif args.grad_mode == 2:
    for n, p in model.named_parameters():  # router
        if "router" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
elif args.grad_mode == 1:
    for n, p in model.named_parameters():  # lora
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
else:
    raise ValueError
    

grad_list = []
for n, p in model.named_parameters():
    if p.requires_grad == True:
        print(f"{n}: {p.requires_grad}")
        grad_list.append(f"{n}")
logging(f"grad_list: {grad_list}", args.logfile)

##################
# Data Loader
##################
trainloader = get_dataloader(args.train_json, args.batch_size, loadtarget=True, tokenizer=tokenizer)
devloader = get_dataloader(args.dev_json, args.batch_size, loadtarget=True, tokenizer=tokenizer)

##################
# Training
##################
criterion = torch.nn.NLLLoss()
router_params = [param for name, param in model.named_parameters() if "router" in name]
expert_params = [param for name, param in model.named_parameters() if "lora_" in name]

router_optimiser = Adam(router_params, lr=args.lr)
expert_optimiser = Adam(expert_params, lr=args.lr2)

##################
# Start Training
##################
logging("Start of training", args.logfile)
bestacc = 0
for epoch in range(args.nepochs):
    start = time.time()
    totalloss = 0
    for idx, data in enumerate(trainloader):
        uttnames, fbank, tgt = data
        fbank = fbank.to(model.device)
        origtarget = [torch.tensor(list(sot_sequence) + y, dtype=torch.long, device=model.device) for y in tgt]
        target = pad_sequence(origtarget, batch_first=True, padding_value=-100).to(model.device)
        targetmask = target != -100

        router_optimiser.zero_grad()
        expert_optimiser.zero_grad()

        # temperature decay
        # if idx == 0 or (idx + 1) % 10 == 0:
        #     current_ratio = (epoch + idx / len(trainloader)) / args.nepochs
        #     temp = 1 - current_ratio * (1 - args.temperature)
        #     model.set_temperature(temp)

        # forward
        logits = model(fbank, target * targetmask)
        output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)
        loss = F.nll_loss(output.view(-1, output.size(-1)), target[:, sotlen:].reshape(-1))

        loss = loss / args.accumgrad
        loss.backward()

        totalloss += loss.item()

        if (idx + 1) % args.accumgrad == 0:
            # LR scheduler
            if args.scheduler == "warmuplr":
                currentstep = epoch * len(trainloader) + idx + 1
                totalstep = args.nepochs * len(trainloader)
                if currentstep > int(args.decay_pct * totalstep):
                    factor = max(0, (totalstep - currentstep) / (totalstep - int(args.decay_pct * totalstep)))
                elif currentstep < int(args.warmup_pct * totalstep):
                    factor = currentstep / int(args.warmup_pct * totalstep)
                else:
                    factor = 1

                router_optimiser.param_groups[0]['lr'] = args.lr * factor
                expert_optimiser.param_groups[0]['lr'] = args.lr2 * factor
            elif args.scheduler == "fixlr":
                pass
            elif args.scheduler == "steplr":
                router_optimiser.param_groups[0]['lr'] = args.lr * (args.gamma ** epoch)
                expert_optimiser.param_groups[0]['lr'] = args.lr2 * (args.gamma ** epoch)
            
            # Interleave training strategy
            if (idx // args.interleave_interval) % 2 == 0:
                router_optimiser.step()
                logging("router-parameter update.", args.logfile)
            else:
                expert_optimiser.step()
                logging("expert-parameter update.", args.logfile)

        if (idx + 1) % args.log_interval == 0:
            logging("{} / {} steps finished in {} | Loss: {} | wp_lr: {} | lora_lr: {}".format(
                idx + 1, len(trainloader), time.time()-start, totalloss/args.log_interval, router_optimiser.param_groups[0]['lr'], expert_optimiser.param_groups[0]['lr']),
                 args.logfile)
            totalloss = 0

    # validation
    totalvalset = 0
    totalvalacc = 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(devloader):
            uttnames, fbank, tgt = data
            fbank = fbank.to(model.device)
            target = [torch.tensor(list(sot_sequence) + y, dtype=torch.long, device=model.device) for y in tgt]
            # target = [torch.tensor(y, dtype=torch.long) for y in tgt]
            target = pad_sequence(target, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100

            # forward
            logits = model(fbank, target * targetmask)
            output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)

            target = target[:, sotlen:]
            output = output.view(target.size(0), target.size(1), -1).max(dim=-1)[1]
            totalvalacc += ((output == target) * targetmask[:, sotlen:]).sum()
            totalvalset += targetmask[:, sotlen:].sum()

            if (idx + 1) % args.accumgrad == 0:
                logging("{} out of {} finished | time elapsed {} | ACC: {}".format(
                    idx + 1, len(devloader), time.time()-start, totalvalacc/totalvalset), args.logfile)
        logging("[epoch {}] Total ACC: {}".format(epoch+1, totalvalacc/totalvalset), args.logfile)

        totalacc = totalvalacc / totalvalset
    if totalacc > bestacc:
        torch.save(model, os.path.join(args.expdir, "model.acc.best"))
        bestacc = totalacc
        logging("Saving best model at epoch {}".format(epoch+1), args.logfile)

    if args.save_snapshot:
        torch.save(model, os.path.join(args.expdir, "snapshot.ep.{}".format(epoch+1)))
