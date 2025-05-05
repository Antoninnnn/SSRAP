import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from Bio import SeqIO
import torch.nn as nn
import torch.optim as optim
import itertools
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = "cuda" if torch.cuda.is_available() else "cpu"



def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )


class DMSDataset(Dataset):
    def __init__(self, csv_file, sequence, structure_sequence, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.sequence = sequence
        self.structure_sequence = structure_sequence
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutant = self.data.iloc[idx]['mutant']
        dms_score = self.data.iloc[idx]['DMS_score']

        # Tokenize sequence
        # tokenized = self.tokenizer(self.sequence, return_tensors="pt")
        # input_ids = tokenized["input_ids"].squeeze(0)
        # attention_mask = tokenized["attention_mask"].squeeze(0)

        # Process structure sequence
        # ss_input_ids = torch.tensor(self.structure_sequence, dtype=torch.long)

        return {
            # 'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            # 'ss_input_ids': ss_input_ids,
            'mutant': mutant,
            'dms_score': torch.tensor(dms_score, dtype=torch.float)
        }



model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)


residue_sequence_dir = "/home/yining_yang/Documents/lm/SSRAP/VenusREM/data/proteingym_v1/aa_seq"
structure_sequence_dir = "/home/yining_yang/Documents/lm/SSRAP/VenusREM/data/proteingym_v1/struc_seq/2048"
# name = "PAI1_HUMAN_Huttinger_2021"
name = "NCAP_I34A1_Doud_2015" # set your few shot DMS assay for fine-

residue_fasta = Path(residue_sequence_dir) / f"{name}.fasta"
structure_fasta = Path(structure_sequence_dir) / f"{name}.fasta"

sequence = read_seq(residue_fasta)
structure_sequence = read_seq(structure_fasta)

structure_sequence = [int(i) for i in structure_sequence.split(",")]
ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
tokenized_results = tokenizer([sequence], return_tensors="pt")
input_ids = tokenized_results["input_ids"].to(device)
attention_mask = tokenized_results["attention_mask"].to(device)

fewshot_DMS_csv = '/home/yining_yang/Documents/lm/SSRAP/data/fewshot/'+name+'.csv'
# dataset = load_dataset('csv', data_files=fewshot_DMS_csv)
# dataset = dataset['train'].train_test_split(test_size=0.1)



dataset = DMSDataset(fewshot_DMS_csv, sequence, structure_sequence, tokenizer)
train_size = int(0.9 * len(dataset))  # 90% for training
val_size = len(dataset) - train_size  # 10% for validation

# Set a seed for reproducibility
torch.manual_seed(42)

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 5
# loss_fn = nn.MarginRankingLoss(margin=0.0)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
tokenized_results = tokenizer([sequence], return_tensors="pt")
input_ids = tokenized_results["input_ids"].to(device)
attention_mask = tokenized_results["attention_mask"].to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # ss_input_ids = batch['ss_input_ids'].to(device)
        # print("the "+str(epoch)+"th epoch")
        dms_scores = batch['dms_score'].to(device)
        mutants = batch['mutant']
        # print(mutants)
        # break

        batch_size = len(mutants)
        # print(input_ids.size())
        # print(batch_size)
        # pred_scores = torch.tensor([0]*batch_size).to(device)
        pred_scores=[]
        # print(pred_scores.size())
        # break
        # print(ss_input_ids)

        outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ss_input_ids=ss_input_ids,
                    labels=input_ids,
                    # output_hidden_states=True,
                    # # output_attentions=True,  # Make sure to get attention outputs
                    # return_dict=True
                )

        # logits = outputs.logits
        logits = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)
        # print(logits)
        # break

        for i in range(batch_size):

            pred_score = 0
            for sub_mutant in mutants[i].split(":"):
                wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                score = logits[0, idx, tokenizer.convert_tokens_to_ids(mt)] - logits[0, idx, tokenizer.convert_tokens_to_ids(wt)]
                pred_score += score
                # print(score)

                # print(pred_score)
            # pred_scores[i] = pred_score
            pred_scores.append(pred_score)

        # pred_scores = pred_scores.clone().detach().requires_grad_(True)
        # print(pred_scores)
        pred_scores = torch.stack(pred_scores)
        # print(pred_scores)
        # break
        # # Prepare pairs for MarginRankingLoss
        # pairs = list(itertools.combinations(range(batch_size), 2))
        # if not pairs:
        #     continue  # Skip if less than 2 samples in batch

        # pred1 = torch.stack([pred_scores[i] for i, j in pairs])
        # pred2 = torch.stack([pred_scores[j] for i, j in pairs])
        # dms1 = torch.stack([dms_scores[i] for i, j in pairs])
        # dms2 = torch.stack([dms_scores[j] for i, j in pairs])

        # # Determine target: 1 if dms1 > dms2, -1 if dms1 < dms2
        # target = torch.sign(dms1 - dms2)
        # non_zero_indices = target != 0
        # if non_zero_indices.sum() == 0:
        #     continue  # Skip if all targets are zero

        # pred1 = pred1[non_zero_indices]
        # pred2 = pred2[non_zero_indices]
        # target = target[non_zero_indices]

                # Compute predicted scores for each sample in the batch
        # This assumes you have a way to map logits to predicted scores
        # For example, summing log-probabilities of the correct tokens

        # batch_size = pred_scores.size(0)

        if batch_size > 1:
            pred_mean = pred_scores.mean()
            pred_std = pred_scores.std(unbiased=False)
            pred_scores_std = (pred_scores - pred_mean) / (pred_std + 1e-8)

            dms_mean = dms_scores.mean()
            dms_std = dms_scores.std(unbiased=False)
            dms_scores_std = (dms_scores - dms_mean) / (dms_std + 1e-8)

            loss = loss_fn(pred_scores_std.view(-1), dms_scores_std.view(-1))
        else:
            # Handle batch size of 1 appropriately
            # For example, you might skip the update or accumulate gradients over multiple batches
            continue


        # loss = loss_fn(pred1, pred2, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Validation phase
    model.eval()
    val_losses = []

    # break
    with torch.no_grad():
        for batch in val_loader:
            dms_scores = batch['dms_score'].to(device)
            mutants = batch['mutant']

            batch_size = len(mutants)
            pred_scores = []

            for i in range(batch_size):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ss_input_ids=ss_input_ids,
                    labels=input_ids
                )

                logits = outputs.logits
                logits = torch.log_softmax(logits[:, 1:-1, :], dim=-1)

                pred_score = 0
                for sub_mutant in mutants[i].split(":"):
                    wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                    score = logits[0, idx, tokenizer.convert_tokens_to_ids(mt)] - logits[0, idx, tokenizer.convert_tokens_to_ids(wt)]
                    pred_score += score.item()
                pred_scores.append(pred_score)

            pred_scores = torch.tensor(pred_scores, device=device)

            # Ensure batch size is greater than 1 to compute standard deviation
            if pred_scores.size(0) > 1:
                pred_mean = pred_scores.mean()
                pred_std = pred_scores.std(unbiased=False)
                pred_scores_std = (pred_scores - pred_mean) / (pred_std + 1e-8)

                dms_mean = dms_scores.mean()
                dms_std = dms_scores.std(unbiased=False)
                dms_scores_std = (dms_scores - dms_mean) / (dms_std + 1e-8)

                # Reshape tensors to ensure matching dimensions
                pred_scores_std = pred_scores_std.view(-1)
                dms_scores_std = dms_scores_std.view(-1)



                loss = loss_fn(pred_scores_std, dms_scores_std)
                
                # print(loss.item())
                val_losses.append(loss.item())
            else:
                # Skip this batch if batch size is 1
                continue

    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
    print(f"Validation Loss: {avg_val_loss:.4f}")



model.save_pretrained("../model/prosst_finetuned_"+name+"_epoch"+str(num_epochs)+"/model")
tokenizer.save_pretrained("../model/prosst_finetuned_"+name+"_epoch"+str(num_epochs)+"/model")