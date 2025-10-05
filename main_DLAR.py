from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from collections import defaultdict
import torch
import numpy as np
import random
import os
import copy
from tqdm import tqdm
from metrics_cluster1 import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim,clusterSim_jsd
import time
import argparse
from datetime import timedelta
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_DLAR1 import DLAR
from modules.optimization import BertAdam
from scipy.optimize import minimize
import torch.nn.functional as F
from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
import tqdm
from sklearn.cluster import KMeans
import skfuzzy as fuzz
torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=4000))
global logger

def get_args(description='DLAR on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--eval_epoch', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_opt", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--cache_dir", default="", type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp16", help="Floating point precition.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")   ## 不同的数据集
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=1, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")
    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")
    parser.add_argument("--tau", default=20, type=float, help="the scale parameter of the exponential function")
    parser.add_argument("--lambda1", default=1.0, type=float, help="the weight of sim_loss")
    parser.add_argument("--lambda2", default=1.0, type=float, help="the origin weight of cluster_loss")
    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")
    parser.add_argument('--num_clusters', type=int, default=15, help="Number of clusters for KMeans")
    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--extract_feature', action='store_true', help='if extract feature')
    parser.add_argument('--fuzzy_index', default=1.1, type=float, help='fcm')
    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    return args

def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = DLAR.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)
    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]
    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir,
        "pytorch_model_{}{}.bin.{}".format(f"{args.num_clusters}_" if hasattr(args, 'num_clusters') and args.num_clusters is not None else "", f"{type_name}." if type_name else "",epoch))
    optimizer_state_file = os.path.join( args.output_dir,
        "pytorch_opt_{}{}.bin.{}".format(f"{args.num_clusters}_" if hasattr(args, 'num_clusters') and args.num_clusters is not None else "",f"{type_name}." if type_name else "",epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({'epoch': epoch,'optimizer_state_dict': optimizer.state_dict(),'loss': tr_loss,}, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = DLAR.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        model.to(device)
    else:
        model = None
    return model

def train_epoch_all(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    if epoch <= 2:  
        print(f'load zero prototype {epoch}') 
        v_prototype = torch.zeros(args.num_clusters, 512)
        t_prototype = torch.zeros(args.num_clusters, 512)
    else:
        print(f'load non-zero prototype {epoch}')  
        v_prototype, t_prototype=load_cluster_centers(args, epoch)
        if isinstance(v_prototype, np.ndarray):  
            v_prototype = torch.from_numpy(v_prototype)
        if isinstance(t_prototype, np.ndarray):   
            t_prototype = torch.from_numpy(t_prototype)

    for step, batch in enumerate(train_dataloader):
        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss, loss_set, _, _ = model(epoch, input_ids, segment_ids, input_mask, video, v_prototype.detach(), t_prototype.detach(), video_mask)
        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()
           
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, feature_Loss: %f, cluster_loss_jsd: %f, Time/step: %f", epoch,
                    args.epochs-1, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss), float(loss_set['feature_loss']),  float(loss_set['cluster_loss_jsd']),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    if epoch != args.epochs - 1 and epoch !=0 and epoch !=1:
        extract_clip_features_dist(model, train_dataloader,v_prototype, t_prototype, device, epoch, args, name=f'membership_contours_{epoch}')
    return total_loss, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list, v_prototype, t_prototype):
    sim_feature_matrix = []
    Sim_cluster_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        seq_features = batch_seq_features_list[idx1]
        each_row_feature = []
        each_row_cluster = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            feature_logits, ret, _, _, _ = model.get_similarity_logits(sequence_output, seq_features, visual_output, input_mask, video_mask, v_prototype, t_prototype,
                                                                     loose_type=model.loose_type)
            feature_logits = feature_logits.detach()
            each_row_feature.append(feature_logits)
            sim_cluster= clusterSim_jsd(ret['v_alpha'],ret['t_alpha']).detach()
            each_row_cluster.append(sim_cluster)
        each_row_feature = torch.cat(tuple(each_row_feature), axis=-1)
        sim_feature_matrix.append(each_row_feature)
        each_row_cluster = torch.cat(tuple(each_row_cluster), axis=-1)
        Sim_cluster_matrix.append(each_row_cluster)
    return sim_feature_matrix, Sim_cluster_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu, epoch):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_seq_features_list = []
        total_video_num = 0
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        print('eval:load v_prototype and t_prototype')
        v_prototype, t_prototype=load_cluster_centers(args,epoch)
        if isinstance(v_prototype, np.ndarray):  
            v_prototype = torch.from_numpy(v_prototype)
        if isinstance(t_prototype, np.ndarray):   
            t_prototype = torch.from_numpy(t_prototype)

        for bid, batch in enumerate(test_dataloader): # Maybe something went wrong here!!!
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            if multi_sentence_:
                b, *_t = video.shape
                sequence_output, seq_features = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))
                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]
                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                (sequence_output, seq_features), visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_seq_features_list.append(seq_features)
                batch_list_t.append((input_mask, segment_ids,))
                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
            print("{}/{}\r".format(bid, len(test_dataloader)), end="")
        print('finish enumerate(test_dataloader)')
        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if epoch <= 2:
            # print('Test................................... ONLY SIM_feature epoch:',epoch)
            sim_feature, _= _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list, v_prototype, t_prototype)
            sim_feature = torch.cat(tuple(sim_feature), axis=0)
            sim_cluster = torch.zeros_like(sim_feature)
            sim_all = args.lambda1*sim_feature + args.lambda2*sim_cluster
        else:  
            # print('Test................................... SIM_all epoch:',epoch)
            sim_feature, sim_cluster= _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_seq_features_list, batch_visual_output_list, v_prototype, t_prototype)
            sim_feature = torch.cat(tuple(sim_feature), axis=0)
            sim_cluster = torch.cat(tuple(sim_cluster), axis=0)
            sim_all = args.lambda1*sim_feature + 0.1*sim_cluster




    #####################################################
    if multi_sentence_:
        ############sim_feature
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_feature.shape[0], sim_feature.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(torch.cat((sim_feature[s_:e_].to(device),
                                                  torch.full((max_length-e_+s_, sim_feature.shape[1]), -float('inf')).to(device)), axis=0))
        sim_feature = torch.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_feature.shape[0], sim_feature.shape[1], sim_feature.shape[2])) 

        ############################################sim_matrix
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_all.shape[0], sim_all.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_all_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(torch.cat((sim_all[s_:e_].to(device),
                                                  torch.full((max_length-e_+s_, sim_all.shape[1]), -float('inf')).to(device)), axis=0))
        sim_all = torch.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_all.shape[0], sim_all.shape[1], sim_all.shape[2])) 
        #####################################################
        tv_metrics_sim_feature, _ = compute_metrics(sim_feature)
        vt_metrics_sim_feature, _ = compute_metrics(sim_feature.T)

        tv_metrics_sim_all, _ = compute_metrics(sim_all)
        vt_metrics_sim_all, _ = compute_metrics(sim_all.T)       
    else:
        logger.info("sim matrix size: {}, {}".format(sim_feature.shape[0], sim_feature.shape[1]))
        tv_metrics_sim_feature, _ = compute_metrics(sim_feature)
        vt_metrics_sim_feature, _ = compute_metrics(sim_feature.T)

        tv_metrics_sim_cluster, _ = compute_metrics(sim_cluster)
        vt_metrics_sim_cluster, _ = compute_metrics(sim_cluster.T)

        tv_metrics_sim_all, _ = compute_metrics(sim_all)
        vt_metrics_sim_all, _ = compute_metrics(sim_all.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_feature), len(sim_feature[0])))
    
    if multi_sentence_ == False:
        logger.info(f'\nT2V-feature-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_feature['R1'], tv_metrics_sim_feature['R5'], tv_metrics_sim_feature['R10'], tv_metrics_sim_feature['MR'], tv_metrics_sim_feature['MeanR']))
        logger.info("V2T-feature-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_feature['R1'], vt_metrics_sim_feature['R5'], vt_metrics_sim_feature['R10'], vt_metrics_sim_feature['MR'], vt_metrics_sim_feature['MeanR']))       

        logger.info(f'\nT2V-cluster-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_cluster['R1'], tv_metrics_sim_cluster['R5'], tv_metrics_sim_cluster['R10'], tv_metrics_sim_cluster['MR'], tv_metrics_sim_cluster['MeanR']))
        logger.info("V2T-cluster-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_cluster['R1'], vt_metrics_sim_cluster['R5'], vt_metrics_sim_cluster['R10'], vt_metrics_sim_cluster['MR'], vt_metrics_sim_cluster['MeanR']))       

        logger.info(f'\nT2V-all-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_all['R1'], tv_metrics_sim_all['R5'], tv_metrics_sim_all['R10'], tv_metrics_sim_all['MR'], tv_metrics_sim_all['MeanR']))
        logger.info("V2T-all-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_all['R1'], vt_metrics_sim_all['R5'], vt_metrics_sim_all['R10'], vt_metrics_sim_all['MR'], vt_metrics_sim_all['MeanR']))   
    else:
        logger.info(f'\nT2V-feature-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_feature['R1'], tv_metrics_sim_feature['R5'], tv_metrics_sim_feature['R10'], tv_metrics_sim_feature['MR'], tv_metrics_sim_feature['MeanR']))
        logger.info("V2T-feature-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_feature['R1'], vt_metrics_sim_feature['R5'], vt_metrics_sim_feature['R10'], vt_metrics_sim_feature['MR'], vt_metrics_sim_feature['MeanR']))       

        logger.info(f'\nT2V-cluster-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_cluster['R1'], tv_metrics_sim_cluster['R5'], tv_metrics_sim_cluster['R10'], tv_metrics_sim_cluster['MR'], tv_metrics_sim_cluster['MeanR']))
        logger.info("V2T-cluster-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_cluster['R1'], vt_metrics_sim_cluster['R5'], vt_metrics_sim_cluster['R10'], vt_metrics_sim_cluster['MR'], vt_metrics_sim_cluster['MeanR']))       

        logger.info(f'\nT2V-all-u:-{args.num_clusters}_{args.fuzzy_index}')
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_sim_all['R1'], tv_metrics_sim_all['R5'], tv_metrics_sim_all['R10'], tv_metrics_sim_all['MR'], tv_metrics_sim_all['MeanR']))
        logger.info("V2T-all-u:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics_sim_all['R1'], vt_metrics_sim_all['R5'], vt_metrics_sim_all['R10'], vt_metrics_sim_all['MR'], vt_metrics_sim_all['MeanR']))        
    R1 = tv_metrics_sim_all['R1']
    return R1

def extract_clip_features_dist(model, dataloader,v_prototype, t_prototype, device, epoch, args, name):
    model.eval()
    text_features_list = []
    video_features_list = []
    # if args.local_rank == 0:
    #     save_path = f"./log-{args.datatype}/cluster-results_{args.num_clusters}_{args.fuzzy_index}"
    #     os.makedirs(save_path, exist_ok=True)
    #     print(f"[Rank {args.local_rank}] Save path created: {save_path}")
    if args.n_gpu > 1:
        torch.distributed.barrier()
        print(f"[Rank {args.local_rank}] Passed initial barrier.")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids, input_mask, segment_ids, video, video_mask = batch
            text_features, video_features = model(
                1, input_ids, segment_ids, input_mask, video, v_prototype, t_prototype, video_mask
            )
            if args.n_gpu > 1:
                gathered_text = [torch.zeros_like(text_features) for _ in range(torch.distributed.get_world_size())]
                gathered_video = [torch.zeros_like(video_features) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_text, text_features)
                torch.distributed.all_gather(gathered_video, video_features)
                if args.local_rank == 0:
                    text_features = torch.cat(gathered_text, dim=0)
                    video_features = torch.cat(gathered_video, dim=0)
            text_features_list.append(text_features.cpu())
            video_features_list.append(video_features.cpu())
        if args.local_rank == 0:
            text_features_ = torch.cat(text_features_list, dim=0)
            video_features_ = torch.cat(video_features_list, dim=0)
            # concatenated_features = torch.cat((text_features_, video_features_), dim=1)
            # pooled_features = F.adaptive_avg_pool1d(concatenated_features.unsqueeze(1), 512).squeeze(1)
            concatenated_features = torch.cat((text_features_, video_features_), dim=0)
            print("[Rank 0] Using Fuzzy C-Means clustering...", flush=True)
            cluster_centers, membership_matrix, _, d, _, _, _ = fuzz.cmeans(
                data=concatenated_features.T.cpu().numpy(),
                c=args.num_clusters,
                m=args.fuzzy_index, 
                error=0.0001,
                maxiter=100000,
                metric='cosine'
            )       
            logger.info("fuzzy_index: %s", args.fuzzy_index)    
            save_path = f"./log-{args.datatype}/cluster-results_{args.num_clusters}_{args.fuzzy_index}"
            print(f"[Rank {args.local_rank}] Save path created: {save_path}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            membership_sums = membership_matrix.sum(axis=0)
            is_valid = np.allclose(membership_sums, 1, atol=1e-6)
            cluster_epoch_path = os.path.join(args.output_dir, f'cluster_centers_{args.datatype}_{args.num_clusters}_{args.fuzzy_index}_{epoch}.pt')
            torch.save(cluster_centers, cluster_epoch_path)
            logger.info(f"[Rank 0] Cluster centers saved to: {cluster_epoch_path}")
            if args.n_gpu > 1:
                cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32).to(device)
                torch.distributed.broadcast(cluster_centers_tensor, src=0)
        if args.n_gpu > 1 and args.local_rank != 0:
            cluster_centers_tensor = torch.zeros((args.num_clusters, 512), dtype=torch.float32).to(device)
            torch.distributed.broadcast(cluster_centers_tensor, src=0)
            cluster_centers = cluster_centers_tensor.cpu().numpy()
        if args.n_gpu > 1:
            torch.distributed.barrier()
            print(f"[Rank {args.local_rank}] Passed final barrier.")
    model.train()

def load_cluster_centers(args, epoch):
    import os
    epoch1 = epoch-1 
    cluster_path = os.path.join(f'./log-{args.datatype}/cluster_centers_{args.datatype}_{args.num_clusters}_{args.fuzzy_index}_{epoch1}.pt') 
    if os.path.exists(cluster_path):
        v_prototype = torch.load(cluster_path)
        t_prototype = torch.load(cluster_path)
        print(f'Cluster centers loaded from {cluster_path}')
    else:
        logger.warning("Cluster centers file not found at: %s", cluster_path)
        v_prototype = torch.zeros(args.num_clusters, 512)
        t_prototype = torch.zeros(args.num_clusters, 512)
    return  v_prototype, t_prototype    

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    tokenizer = ClipTokenizer()
    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue
            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                param.requires_grad = False
  
    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        
        
        ##########   这里可能不需要事先提取聚类中心了
        if args.extract_feature:
            v_prototype, t_prototype=load_cluster_centers(args,0)
            if isinstance(v_prototype, np.ndarray):  
                v_prototype = torch.from_numpy(v_prototype)
            if isinstance(t_prototype, np.ndarray):   
                t_prototype = torch.from_numpy(t_prototype)
            extract_clip_features_dist(model, train_dataloader,v_prototype, t_prototype, device, -1 , args, name='membership_contours_eval')
            exit()

        best_score = 0.00001
        best_output_model_file = "None"
        resumed_epoch = 0
        if args.resume_opt:
            checkpoint = torch.load(args.resume_opt, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            
            ########................................................................第一步 train...........................................
            tr_loss, global_step = train_epoch_all(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")
                logger.info(f'Eval on val dataset-{args.num_clusters}_{args.fuzzy_index}')    
            
            ########................................................................第一步 test...........................................    
                R1 = eval_epoch(args, model, val_dataloader, device, n_gpu, epoch)
                print(f'=================================')
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
                # cluster_path = os.path.join(args.output_dir, f'cluster_centers_{args.datatype}_{args.num_clusters}_{args.fuzzy_index}.pt')
                # if os.path.exists(cluster_path):
                #     os.remove(cluster_path)
                #     print(f"{cluster_path} remove")
                # else:
                #     print(f"{cluster_path} not")
                ##############################
                # if epoch <= 0:
                #     continue
                # else:
                #     cluster_epoch_path = os.path.join(args.output_dir, f'cluster_centers_{args.datatype}_{args.num_clusters}_{args.fuzzy_index}_{epoch}.pt')
                #     clusters_epoch_center=torch.load(cluster_epoch_path)
                #     logger.info(f"eval_epoch fininsh load: {cluster_epoch_path}")
                #     torch.save(clusters_epoch_center, cluster_path)
                #     logger.info(f"fininsh Cluster centers saved to: {cluster_path}")
    elif args.do_eval:
        if args.local_rank == 0:
            model.eval() 
            eval_epoch(args, model, test_dataloader, device, n_gpu, args.eval_epoch)
if __name__ == "__main__":
    main()
