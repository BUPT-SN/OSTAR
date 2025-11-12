import os
import torch
import argparse
import logging
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler, Subset
from tqdm import tqdm, trange
from torch.optim import AdamW

from transformers import (
    set_seed,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from apex import amp
except ImportError:
    amp = None

from util import glue_compute_metrics as compute_metrics
from util import glue_output_modes as output_modes
from util import glue_processors as processors

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    ray = None

from Modeling import RobertaForMultiFacetedContrastiveLearning

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data"), type=str, help="The input data dir.")
parser.add_argument("--model_type", default="roberta", type=str, help="Base model")
parser.add_argument("--model_name_or_path", default="/data/Content_Moderation/model/roberta-base", type=str, help="Path to pretrained model")
parser.add_argument("--task_name", default="deepfake", type=str)
parser.add_argument("--output_dir", default=os.path.join(os.getcwd(), "test_deepfake_attack_robust_0.02"), type=str, required=False)
parser.add_argument("--config_name", default="", type=str)
parser.add_argument("--train_file", default="train_all.jsonl", type=str)
parser.add_argument("--dev_file", default="test_attack_Robust.jsonl", type=str)
parser.add_argument("--test_file", default="test_attack_Robust.jsonl", type=str)
parser.add_argument("--tokenizer_name", default="", type=str)
parser.add_argument("--cache_dir", default="", type=str)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--do_train", default=True, help="Whether to run training.")
parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=True, help="Whether to run test on the dev set.")
parser.add_argument("--evaluate_during_training", action="store_true")
parser.add_argument("--do_lower_case", action="store_true")
parser.add_argument("--per_gpu_train_batch_size", default=4, type=int)
parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--num_train_epochs", default=10, type=float)
parser.add_argument("--max_steps", default=-1, type=int)
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--logging_steps", type=int, default=125)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--eval_all_checkpoints", action="store_true")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--overwrite_cache", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--fp16_opt_level", type=str, default="O1")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--server_ip", type=str, default="")
parser.add_argument("--server_port", type=str, default="")
parser.add_argument("--dataset_name", default="gpt3.5_mixed", type=str)
parser.add_argument("--do_ray", action='store_true')
parser.add_argument("--wandb_log", action="store_true")
parser.add_argument("--wandb_note", default="CoCo_rf", type=str)
parser.add_argument("--contrastive_weight", type=float, default=0, help="Weight for the contrastive loss.")
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--use_triplet_loss", action="store_true")
parser.add_argument("--triplet_margin", type=float, default=1.0)

args = parser.parse_args()

class InputFeaturesWithManipulation(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label, manipulation):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.manipulation = manipulation

    def __repr__(self):
        return str({
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
            'label': self.label,
            'manipulation': self.manipulation
        })

def convert_examples_to_features_with_manipulation(
    examples,
    tokenizer,
    label_list=None,
    output_mode="classification",
    max_length=512,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    MANIPULATION_MAP = {
        "human_orginal": 0,
        "human_attack": 1,
        "human_paraphrase": 2,
        "machine_orginal": 3,
        "machine_attack": 4,
        "machine_paraphrase": 5
    }
    
    if output_mode == "classification":
        label_list = sorted(set(ex.label for ex in examples)) if label_list is None else label_list
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None
    
    if not examples:
        raise ValueError("输入示例列表为空！请检查数据加载逻辑")

    features = []
    for ex_index, ex in enumerate(examples):
        required_attrs = ['text_a', 'label', 'manipulation']
        for attr in required_attrs:
            if not hasattr(ex, attr):
                raise ValueError(f"示例{ex_index}缺少必要属性: {attr}")

        inputs = tokenizer.encode_plus(
            ex.text_a,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )

        manip_str = getattr(ex, 'manipulation', 'human_orginal').lower().strip()
        manip_label = MANIPULATION_MAP.get(manip_str, 0)
        
        feature = InputFeaturesWithManipulation(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            label=label_map[ex.label] if label_map else ex.label,
            manipulation=manip_label
        )
        features.append(feature)

        if ex_index < 3:
            logger.info(f"样本 {ex_index} 转换详情:")
            logger.info(f"原始文本: {ex.text_a[:50]}...")
            logger.info(f"标签: {ex.label} -> {feature.label}")
            logger.info(f"Manipulation: {manip_str} -> {manip_label}")
            logger.info(f"Input IDs长度: {len(feature.input_ids)}")

    if not features:
        raise ValueError("未生成任何特征，请检查数据处理逻辑！")
    
    logger.info(f"成功生成 {len(features)} 个特征样本")
    return features

class DynamicContrastiveDataset(Dataset):
    MANIPULATION_MAP = {
        "human_orginal": 0,
        "human_attack": 1,
        "human_paraphrase": 2,
        "machine_orginal": 3,
        "machine_attack": 4,
        "machine_paraphrase": 5
    }

    def __init__(self, dataset, seed=42):
        if not isinstance(dataset, (Subset, TensorDataset)):
            raise TypeError(f"Invalid dataset type: {type(dataset)}")
            
        self.valid_indices = [i for i in range(len(dataset)) if len(dataset[i]) == 5]
        
        if not self.valid_indices:
            raise ValueError("没有符合5字段的有效样本")
            
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                assert len(sample) == 5, f"索引 {idx} 的样本字段数为 {len(sample)}"
                assert isinstance(sample[4], torch.Tensor), f"manipulation 字段类型错误: {type(sample[4])}"
                
                manip_code = sample[4].item()
                assert 0 <= manip_code <= 5, f"非法manipulation代码: {manip_code}"
                
                if idx not in self.valid_indices:
                    self.valid_indices.append(idx)
            except Exception as e:
                logger.error(f"无效样本 {idx}: {str(e)}")
                continue
                
        self.dataset = Subset(dataset, self.valid_indices)
        logger.info(f"动态数据集初始化成功，总样本: {len(self.dataset)}")
        self.rng = random.Random(seed)
        
        self.human_original = []
        self.human_attack = []
        self.human_paraphrase = []
        self.machine_original = []
        self.machine_attack = []
        self.machine_paraphrase = []
        
        for idx in range(len(self.dataset)):
            input_ids, attention_mask, token_type_ids, label, manipulation = self.dataset[idx]
            manip_str = self._code_to_manipulation(manipulation.item())
            
            if manip_str == "human_orginal":
                self.human_original.append(idx)
            elif manip_str == "human_attack":
                self.human_attack.append(idx)
            elif manip_str == "human_paraphrase":
                self.human_paraphrase.append(idx)
            elif manip_str == "machine_orginal":
                self.machine_original.append(idx)
            elif manip_str == "machine_attack":
                self.machine_attack.append(idx)
            elif manip_str == "machine_paraphrase":
                self.machine_paraphrase.append(idx)
            else:
                raise ValueError(f"无效的manipulation代码: {manipulation.item()}")

    def _code_to_manipulation(self, code):
        mapping = {
            0: "human_orginal",
            1: "human_attack",
            2: "human_paraphrase",
            3: "machine_orginal",
            4: "machine_attack",
            5: "machine_paraphrase"
        }
        return mapping.get(code, "unknown")

    def __len__(self):
        return len(self.human_original) + len(self.machine_original)

    def __getitem__(self, idx):
        if idx < len(self.human_original):
            anchor_idx = self.human_original[idx]
            attack_idx = self.rng.choice(self.human_attack)
            positive_candidates = [i for i in self.human_original if i != anchor_idx]
            positive_idx = self.rng.choice(positive_candidates) if positive_candidates else anchor_idx
            negative_idx = self.rng.choice(self.human_paraphrase)
        else:
            machine_idx = idx - len(self.human_original)
            anchor_idx = self.machine_original[machine_idx]
            attack_idx = self.rng.choice(self.machine_attack)
            positive_idx = self.rng.choice(self.machine_paraphrase)
            negative_idx = self.rng.choice(self.human_original)
            
        return (
            self.dataset[anchor_idx],
            self.dataset[attack_idx],
            self.dataset[positive_idx],
            self.dataset[negative_idx]
        )

def dynamic_group_collate_fn(batch):
    def process_sample(sample):
        return {
            'input_ids': sample[0],
            'attention_mask': sample[1],
            'token_type_ids': sample[2],
            'labels': sample[3]
        }
    
    anchors, attacks, positives, negatives = zip(*batch)
    
    return {
        'anchor': [process_sample(s) for s in anchors],
        'attack': [process_sample(s) for s in attacks],
        'positive': [process_sample(s) for s in positives],
        'negative': [process_sample(s) for s in negatives]
    }

def train(args, train_dataset, model, tokenizer):
    st_logger = logging.getLogger('sentence_transformers')
    st_logger.setLevel(logging.WARNING)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        force=True 
    )
    
    def save_checkpoint(args, model, optimizer, scheduler, epoch, metric_name, metric_value):
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"best-by-{metric_name}",
            f"epoch{epoch}-{metric_name}_{metric_value:.4f}"
        )
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(checkpoint_dir)
        
        torch.save(args, os.path.join(checkpoint_dir, "training_args.bin"))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        logger.info(f"Saved {metric_name} best model to {checkpoint_dir}")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=dynamic_group_collate_fn
    )

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=t_total
    )

    if args.fp16:
        if amp is None:
            raise ImportError("Please install apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step, tr_loss = 0, 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    
    best_f1 = -np.inf
    best_acc = -np.inf
    best_acc_f1 = -np.inf
    best_f1_acc = -np.inf
    results = {}
    
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        model.train()
        
        for step, batch in enumerate(epoch_iterator):
            def merge_group(group):
                return {
                    'input_ids': torch.stack([f['input_ids'] for f in group]),
                    'attention_mask': torch.stack([f['attention_mask'] for f in group]),
                    'token_type_ids': torch.stack([f['token_type_ids'] for f in group]),
                    'labels': torch.stack([f['labels'] for f in group])
                }
            
            anchor = merge_group(batch['anchor'])
            attack = merge_group(batch['attack'])
            positive = merge_group(batch['positive'])
            negative = merge_group(batch['negative'])

            combined_inputs = {
                'input_ids': torch.cat([g['input_ids'] for g in [anchor, attack, positive, negative]]),
                'attention_mask': torch.cat([g['attention_mask'] for g in [anchor, attack, positive, negative]]),
                'token_type_ids': torch.cat([g['token_type_ids'] for g in [anchor, attack, positive, negative]]),
                'labels': torch.cat([g['labels'] for g in [anchor, attack, positive, negative]])
            }
            
            combined_inputs = {k: v.to(args.device) for k, v in combined_inputs.items()}

            outputs = model(**combined_inputs)
            loss = outputs[0].mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer) if args.fp16 else model.parameters(),
                    args.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    logs = {
                        "loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "step": global_step
                    }
                    logging_loss = tr_loss
                    logger.info(f"Training Log: {logs}")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info(f"Saved checkpoint to {output_dir}")
                
        if args.local_rank in [-1, 0] and args.do_eval:
            epoch_results = evaluate(args, model, tokenizer, prefix=f"epoch{epoch}", mode="dev")
            logger.info(f"Epoch {epoch} Evaluation Results - Accuracy: {epoch_results['acc']:.4f}, F1: {epoch_results['f1']:.4f}")

            current_acc = epoch_results.get("acc", 0)
            current_f1 = epoch_results.get("f1", 0)
            
            if current_acc > best_acc:
                best_acc = current_acc
                best_acc_f1 = current_f1
                save_checkpoint(
                    args, model, optimizer, scheduler, epoch, 
                    metric_name="acc", metric_value=current_acc
                )
                results = epoch_results.copy()

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_f1_acc = current_acc
                save_checkpoint(
                    args, model, optimizer, scheduler, epoch,
                    metric_name="f1", metric_value=current_f1
                )
                results = epoch_results.copy()

            logger.info(f"Current Best Models => "
                        f"Accuracy: {best_acc:.4f} (F1@{best_acc_f1:.4f}) | "
                        f"F1: {best_f1:.4f} (Acc@{best_f1_acc:.4f})")

    final_output_dir = os.path.join(args.output_dir, "final_model")
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(final_output_dir)
    torch.save(args, os.path.join(final_output_dir, "training_args.bin"))
    logger.info(f"Training completed. Final model saved to {final_output_dir}")

    return global_step, tr_loss / global_step if global_step > 0 else 0, results, final_output_dir

def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode="dev"):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True, mode=mode
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds, out_label_ids = None, None
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            
            with torch.no_grad():
                input_ids, attention_mask, token_type_ids, labels, _ = batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss, logits = outputs[:2]

                eval_loss += loss.mean().item()
                
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )
                
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        os.makedirs(os.path.dirname(output_eval_file), exist_ok=True)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False, mode="train", dataset_name=""):
    processor = processors[task]()
    output_mode = output_modes[task]

    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{mode}_{args.model_type}_{args.max_seq_length}_{task}_with_manipulation_author"
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info(f"加载缓存特征文件 {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"从数据集文件创建特征: {args.data_dir}")
        if mode == "train":
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif mode == "test":
            examples = processor.get_test_examples(args.data_dir, args.test_file)

        features = convert_examples_to_features_with_manipulation(
            examples,
            tokenizer,
            label_list=processor.get_labels(),
            output_mode=output_mode,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        torch.save(features, cached_features_file)
        for f in features:
            if not hasattr(f, 'manipulation'):
                raise ValueError(f"特征缺少manipulation字段，数据集模式: {mode}")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], 
        dtype=torch.long if output_mode == "classification" else torch.float)
    all_manipulation = torch.tensor([f.manipulation for f in features], dtype=torch.long)
    
    assert len(all_manipulation) == len(all_input_ids), "字段数量不一致"
    logger.info(f"成功加载数据集，总样本数: {len(all_input_ids)}")
    logger.info(f"首个样本manipulation值: {all_manipulation[0].item()}")

    return TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
        all_manipulation
    )

def run(conf, data_dir=None):
    args.seed = conf["seed"]
    args.data_dir = data_dir

    res = {}

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    print(device)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args.seed)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    train_dataset = load_and_cache_examples(
        args,
        args.task_name,
        tokenizer,
        evaluate=False,
        mode="train",
    )
    contrastive_train_dataset = DynamicContrastiveDataset(train_dataset, seed=args.seed)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    model = RobertaForMultiFacetedContrastiveLearning.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        ignore_mismatched_sizes=True 
    )
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        global_step, tr_loss, train_res, output_dir = train(
            args, contrastive_train_dataset, model, tokenizer
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        res.update(train_res)

    return res

def main():
    data_dir = os.path.abspath("/data/Content_Moderation/paper_data/deepfake_final")
    if args.do_ray:
        import ray
        ray.init()
        config = {
            "seed": tune.choice([10, 11, 12, 13, 14, 15]),
        }
        scheduler = ASHAScheduler(metric="test_accuracy", mode="max")
        reporter = CLIReporter(
            metric_columns=[
                "test_accuracy",
                "test_f1",
            ]
        )

        result = tune.run(
            partial(run, data_dir=data_dir),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            num_samples=8,
            scheduler=scheduler,
            progress_reporter=reporter,
        )
        best_trial = result.get_best_trial("test_accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print(
            "Best trial final validation accuracy: {}".format(
                best_trial.last_result["test_accuracy"]
            )
        )
    else:
        for seed in [10, 11, 12, 13, 14, 15]:
            config = {
                "seed": seed,
            }
            run(config, data_dir)

if __name__ == "__main__":
    main()

