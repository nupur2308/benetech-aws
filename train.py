from __future__ import print_function
import argparse, os, json, traceback, sys, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
from torch import autocast


from PIL import Image
from copy import deepcopy
import albumentations as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from IPython.display import Image, display, HTML
from operator import itemgetter
from tqdm.auto import tqdm
import shutil

from tokenizers import AddedToken
from transformers import Pix2StructProcessor
from transformers import DataCollatorWithPadding
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration
from transformers import GenerationConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator


#Set parallelism to false -- https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#---------------------------------------------------
# UTILS
#---------------------------------------------------

from utils.data_utils import process_annotations
from utils.metric_utils import compute_metrics
from utils.train_utils import (EMA, AverageMeter, as_minutes, get_lr,
                                   print_line, init_wandb
                                  )
from utils.dotdict import dotdict

#---------------------------------------------------
# DATASET, MODEL, AND DATALOADER
#---------------------------------------------------


from model.benetech_dataset import (TOKEN_MAP, BenetechDataset,
                                     create_train_transforms)
from model.benetech_dataloader import BenetechCollator
from model.benetech_model import BenetechModel

    
#---------------------------------------------------  
#SHOW BATCH
#---------------------------------------------------
    
    
def run_sanity_check(hyperparams, batch, tokenizer, num_examples=8):
    print("generating sanity check results for a training batch...")
    
    num_examples = min(num_examples, len(batch['images']))
    print(f"num_examples={num_examples}")

    for i in range(num_examples):
        image = batch['images'][i]
        text = tokenizer.decode(
            batch['decoder_input_ids'][i], skip_special_tokens=True)

        text = "\n".join(wrap(text, width=128))
        
        
        #display image and its corresponding text label ---
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.xlabel(text)
        plt.close()


    print("done!")
    
    
    
#---------------------------------------------------  
#EVALUATION
#---------------------------------------------------

def post_process(pred_string, token_map, delimiter="|"):
    # get chart type ---
    chart_options = [
        "horizontal_bar",
        "dot",
        "scatter",
        "vertical_bar",
        "line",
        "histogram",
    ]

    chart_type = "line"  # default type

    for ct in chart_options:
        if token_map[ct] in pred_string:
            chart_type = ct
            break

    if chart_type == "histogram":
        chart_type = "vertical_bar"

    # get x series ---
    x_start_tok = token_map["x_start"]
    x_end_tok = token_map["x_end"]

    try:
        x = pred_string.split(x_start_tok)[1].split(x_end_tok)[0].split(delimiter)
        x = [elem.strip() for elem in x if len(elem.strip()) > 0]
    except IndexError:
        x = []

    # get y series ---
    y_start_tok = token_map["y_start"]
    y_end_tok = token_map["y_end"]

    try:
        y = pred_string.split(y_start_tok)[1].split(y_end_tok)[0].split(delimiter)
        y = [elem.strip() for elem in y if len(elem.strip()) > 0]
    except IndexError:
        y = []

    return chart_type, x, y


def run_evaluation(hyperparams, model, valid_dl, label_df, tokenizer, token_map):
    
    # # config for text generation ---
    conf_g = {
        "max_new_tokens": hyperparams.max_length_generation,  # 256,
        "do_sample": False,
        "top_k": 1,
        "use_cache": True,
    }
    
    generation_config = GenerationConfig(**conf_g)
    
    device = hyperparams.device

    # put model in eval mode ---
    model.eval()

    all_ids = []
    all_texts = []

    progress_bar = tqdm(range(len(valid_dl)))
    for batch in valid_dl:
        with torch.no_grad():
    
            batch_ids = batch.pop("id")
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            
            generated_ids = model.backbone.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_ids.extend(batch_ids)
            all_texts.extend(generated_texts)

        progress_bar.update(1)
    progress_bar.close()
    
    # prepare output dataframe ---
    preds = []
    extended_preds = []
    for this_id, this_text in zip(all_ids, all_texts):
        id_x = f"{this_id}_x"
        id_y = f"{this_id}_y"
        pred_chart, pred_x, pred_y = post_process(this_text, token_map)

        preds.append([id_x, pred_x, pred_chart])
        preds.append([id_y, pred_y, pred_chart])

        extended_preds.append([id_x, pred_x, pred_chart, this_text])
        extended_preds.append([id_y, pred_y, pred_chart, this_text])

    pred_df = pd.DataFrame(preds)
    pred_df.columns = ["id", "data_series", "chart_type"]

    eval_dict = compute_metrics(label_df, pred_df)

    result_df = pd.DataFrame(extended_preds)
    result_df.columns = ["id", "pred_data_series", "pred_chart_type", "pred_text"]
    result_df = pd.merge(label_df, result_df, on="id", how="left")
    result_df['score'] = eval_dict['scores']  # individual scores

    results = {
        "oof_df": pred_df,
        "result_df": result_df,
    }

    for k, v in eval_dict.items():
        if k != 'scores':
            results[k] = v

    print_line()
    print("Evaluation Results:")
    print(results)
    print_line()

    return results


    
#---------------------------------------------------  
#MAIN FUNCTION
#---------------------------------------------------
    

def main():
    # Read hyper parameters passed by SageMaker
    # https://github.com/aws/sagemaker-python-sdk/issues/613
    hyperparams = dotdict(json.loads(os.environ["SM_TRAINING_ENV"])["hyperparameters"])
    
    #https://nono.ma/sagemaker-model-dir-output-dir-and-output-data-dir-parameters
    hyperparams['input_path'] = os.environ['SM_CHANNEL_TRAINING']
    hyperparams['output_path'] = os.environ['SM_OUTPUT_DIR']
    hyperparams['output_data_path'] = os.environ['SM_OUTPUT_DATA_DIR']
    hyperparams['model_path'] = os.environ['SM_MODEL_DIR']
    
    seed = 42
    torch.manual_seed(seed)

    hyperparams['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    #-----GET TRAIN AND VALID IDS------#
    fold = hyperparams.fold
    train_folds = [i for i in range(hyperparams.n_folds) if i != fold]
    valid_folds = [fold]
    train_folds.append(99)
    fold_df = pd.read_parquet(os.path.join(hyperparams.input_path, 'cv_map_4_folds.parquet'))
    
    extracted_ids = fold_df[fold_df["kfold"] != 99]["id"].unique().tolist()
    train_ids = fold_df[fold_df["kfold"].isin(train_folds)]["id"].unique().tolist()
    print(f'# images in original train: {len(train_ids)}')
    
    #repeat original train ids 
    #train_ids = train_ids * hyperparams.original_multiplier
    #print(f'# images in original train after multiplier: {len(train_ids)}')
    
    extracted_train_ids = deepcopy(extracted_ids)
    #extracted_train_ids = extracted_train_ids * max(hyperparams.extracted_multiplier - 1, 1)
    
    if len(extracted_train_ids) > 0:
        train_ids.extend(extracted_train_ids)
    
    print(f'# images in original train after extracted multiplier: {len(train_ids)}')
    
    # valid ids
    valid_ids = fold_df[fold_df["kfold"].isin(valid_folds)]["id"].unique().tolist()
    
    
    # labels
    label_df = process_annotations(hyperparams)
    label_df["original_id"] = label_df["id"].apply(lambda x: x.split("_")[0])
    label_df = label_df[label_df["original_id"].isin(valid_ids)].copy()
    label_df = label_df.drop(columns=["original_id"])
    label_df = label_df.sort_values(by="source")
    label_df = label_df.reset_index(drop=True)

    # show labels 
    print("labels:")
    print(label_df.head())
    print_line()

    print(f"# of graphs in train: {len(train_ids)}")
    print(f"# of graphs in valid: {len(valid_ids)}")
    print_line()

       
    #-----AUGMENTATIONS------#
    
    if hyperparams.use_augmentations:
        print_line()
        print("using augmentations...")
        train_transforms = create_train_transforms()
        print_line()
        
    #-----DATASET------#
        
    train_ds = BenetechDataset(hyperparams, train_ids, transform=train_transforms)
    valid_ds = BenetechDataset(hyperparams, valid_ids)

    tokenizer = train_ds.processor.tokenizer
    hyperparams.len_tokenizer = len(tokenizer)
    
    BOS_TOKEN = TOKEN_MAP["bos_token"]

    hyperparams.pad_token_id = tokenizer.pad_token_id
    hyperparams.decoder_start_token_id = tokenizer.convert_tokens_to_ids([BOS_TOKEN])[0]
    hyperparams.bos_token_id = tokenizer.convert_tokens_to_ids([BOS_TOKEN])[0]
        
    #-----DATALOADERS------#
    
    collate_fn = BenetechCollator(tokenizer=tokenizer)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=hyperparams.train_bs,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
    )
    

    valid_dl = DataLoader(
        valid_ds,
        batch_size=hyperparams.valid_bs,
        collate_fn=collate_fn,
        shuffle=False,
    )
    
    #-----WANDB LOGIN------#
    
    print_line()
    if hyperparams.use_wandb:
        print("initializing wandb run...")
        init_wandb(hyperparams)
    print_line()
    
    #-----SHOW BATCH------#
    
    print_line()
    for idx, b in enumerate(train_dl):
        if idx == 16:
            break
        run_sanity_check(hyperparams, b, tokenizer)

    for idx, b in enumerate(valid_dl):
        if idx == 4:
            break
        run_sanity_check(hyperparams, b, tokenizer)
        
    #-----LOAD MODEL------#   
    
    print_line()
    print("creating the Benetech model...")
    model = BenetechModel(hyperparams)  # get_model(cfg)
    model.to(hyperparams.device)
    print_line()

    #-----OPTIMIZER------#   
    
    print_line()
    print("creating the optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.optimizer_lr,
        weight_decay=hyperparams.optimizer_weight_decay,

    )
    
    #-----SCHEDULER------#
    
    print_line()
    print("creating the scheduler...")

    num_epochs = hyperparams.num_epochs
    grad_accumulation_steps = hyperparams.grad_accumulation
    warmup_pct = hyperparams.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    
    #-----SETUP------#
        
    best_lb = -1.
    save_trigger = hyperparams.save_trigger

    patience_tracker = 0
    current_iteration = 0
        
        
    #-----EMA------#
    
    '''
    if hyperparams.use_ema:
        print_line()
        decay_rate = hyperparams.decay_rate
        ema = EMA(model, decay=decay_rate)
        ema.register()

        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")
        print_line()
       
    '''
    
    #-----TRAINING------#
    
    start_time = time.time()
    num_vbar = 0
    num_hbar = 0
    num_histogram = 0
    num_dot = 0
    num_line = 0
    num_scatter = 0
    
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    gc.collect()
    for epoch in range(num_epochs):
        epoch_progress = 0
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()

        model.train()
        
        for step, batch in enumerate(train_dl):
            num_vbar += len([ct for ct in batch['chart_type'] if ct == 'vertical_bar'])
            num_hbar += len([ct for ct in batch['chart_type'] if ct == 'horizontal_bar'])
            num_histogram += len([ct for ct in batch['chart_type'] if ct == 'histogram'])
            num_dot += len([ct for ct in batch['chart_type'] if ct == 'dot'])
            num_line += len([ct for ct in batch['chart_type'] if ct == 'line'])
            num_scatter += len([ct for ct in batch['chart_type'] if ct == 'scatter'])
            
            #https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            with autocast(device_type='cuda', dtype=torch.float16):
            
                loss = model(
                    flattened_patches = batch['flattened_patches'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
            scaler.scale(loss).backward()
            epoch_progress += 1
            
            #https://wandb.ai/wandb_fc/tips/reports/How-To-Implement-Gradient-Accumulation-in-PyTorch--VmlldzoyMjMwOTk5
            if ((step + 1) % grad_accumulation_steps == 0) or (step + 1 == len(train_dl)):
                #nn.utils.clip_grad_norm_(model.parameters(), hyperparams.grad_clip_value)  
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                loss_meter.update(loss.item())


                # ema 
                #if hyperparams.use_ema:
                #    ema.update()
               

                progress_bar.set_description(
                    f"STEP: {epoch_progress+1:5}/{len(train_dl):5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )
                progress_bar.update(1)
                current_iteration += 1
    
            #-----EVALUATION------#

            if (epoch_progress + 1) % hyperparams.eval_frequency == 0:
                model.eval()

                # apply ema if it is used ---
                #if hyperparams.use_ema:
                #    ema.apply_shadow()

                result_dict = run_evaluation(
                    hyperparams,
                    model=model,
                    valid_dl=valid_dl,
                    label_df=label_df,
                    tokenizer=tokenizer,
                    token_map=TOKEN_MAP,
                )
                
                
                lb = result_dict["lb"]
                oof_df = result_dict["oof_df"]
                result_df = result_dict["result_df"]

                print_line()
                et = as_minutes(time.time()-start_time)
                print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # ---
                    best_dict = dict()
                    for k, v in result_dict.items():
                        if "df" not in k:
                            best_dict[f"{k}_at_best"] = v

                else:
                    patience_tracker += 1

                print_line()
                print(f">>> Current LB = {round(lb, 4)}")
                for k, v in result_dict.items():
                    if ("df" not in k) & (k != "lb"):
                        print(f">>> Current {k}={round(v, 4)}")
                print_line()
                
                if is_best:
                    oof_df.to_csv(os.path.join(hyperparams.output_data_path, f"oof_df_fold_{fold}_best.csv"), index=False)
                    result_df.to_csv(os.path.join(hyperparams.output_data_path, f"result_df_fold_{fold}_best.csv"), index=False)
                else:
                    print(f">>> patience reached {patience_tracker}/{hyperparams.patience}")
                    print(f">>> current best score: {round(best_lb, 4)}")
                    
                oof_df.to_csv(os.path.join(hyperparams.output_data_path, f"oof_df_fold_{fold}.csv"), index=False)
                result_df.to_csv(os.path.join(hyperparams.output_data_path, f"result_df_fold_{fold}.csv"), index=False)

                # save pickle for analysis
                result_df.to_pickle(os.path.join(hyperparams.output_data_path, f"result_df_fold_{fold}.pkl"))
                
                #-----SAVING------#
                              
                if best_lb > save_trigger:
                    print("saving model...")
                    name = f"benetech_model_fold_{hyperparams['fold']}"
                    path = os.path.join(hyperparams.model_path, name+".pt")
                    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
                    torch.save(model.cpu().state_dict(), path)
                    if is_best:
                        best_name = f"{name}_best.pt"
                        shutil.copyfile(path, os.path.join(hyperparams.model_path, best_name))

                
                # -- post eval
                model.train()
                model.to(hyperparams.device)
                torch.cuda.empty_cache()
                gc.collect()

                
                # ema ---
                #if hyperparams.use_ema:
                #   ema.restore()
                
                
                print_line()

                # early stopping ----
                if patience_tracker >= hyperparams.patience:
                    print("stopping early")
                    model.eval()
                    return
        
                
if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(os.environ['SM_OUTPUT_DIR'], 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)