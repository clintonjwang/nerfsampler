"""INR classification"""
import os
import pdb
import time
import torch
import wandb

# from nerfsampler.inn.nets.consistent import LearnedSampler
from nerfsampler.utils import jobs as job_mgmt, util
from nerfsampler.utils import args as args_module
osp = os.path
nn = torch.nn
F = nn.functional

from nerfsampler import inn, RESULTS_DIR
from nerfsampler.data import dataloader
from nerfsampler.inn import point_set
import nerfsampler.baselines.classifier
import nerfsampler.inn.nets.classifier
import nerfsampler.baselines.hypernets
import nerfsampler.baselines.nuft
from nerfsampler.utils.losses import contrastive_loss as contrastive_loss_fxn
import yaml

def train_classifier(args: dict) -> None:
    wandb.init(project="nerfsampler", name=args["job_id"],
        config=wandb.helper.parse_config(args, exclude=['job_id']))
    args = args_module.get_wandb_train_config()
    paths = args["paths"]

    entropy_loss = args['optimizer'].get('discretization entropy loss', None)
    contrastive_loss = args['optimizer'].get('discretization contrastive loss', None)
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    val_data_loader = dataloader.get_val_inr_dataloader(dl_args)
    global_step = 0
    loss_fxn = nn.CrossEntropyLoss()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
    top3 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    if entropy_loss:
        entropy_loss = lambda logits: torch.distributions.Categorical(logits=logits).entropy().mean()
    if contrastive_loss:
        contrastive_loss = contrastive_loss_fxn


    model = get_model(args)
    # wandb.watch(model, log="all", log_freq=100)
    optimizer = util.get_optimizer(model, args)

    # ratio = 16
    # dense_sampler = model.sampler.copy()
    # dense_sampler['sample points'] *= ratio/4
    # query_layers = nn.Sequential()
    # flow_layers = nn.Sequential()
    # LearnedSampler(dense_sampler, query_layers, flow_layers, ratio=ratio)
    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=2,
    #         repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(paths["job output dir"]),
    #     profile_memory=True,
    #     with_flops=True,
    #     with_modules=True,
    # ) as profiler:
    start_time = time.time()
    for img_inr, labels in data_loader:
        model.train()
        global_step += 1
        discretizations = point_set.get_discretizations_for_args(args)
        in_disc = discretizations['input']
        ntype = args["network"]['type'].lower()
        if contrastive_loss:
            logits, features = model(img_inr, sampler=in_disc)
        elif util.is_model_nerfsampler(args) or ntype.startswith('mlp') or ntype.startswith('nuft'):
            logits = model(img_inr, sampler=in_disc)
        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)

        if entropy_loss:
            loss += entropy_loss(logits)
        elif contrastive_loss:
            loss += contrastive_loss(features)
        else:
            loss = loss_fxn(logits, labels)
        pred_cls = logits.topk(k=3).indices
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        wandb.log({'train_loss':loss.item(),
            'train_t3_acc': top3(pred_cls, labels).item(),
            'train_t1_acc': top1(pred_cls, labels).item(),
            'mins_elapsed': (time.time() - start_time)/60,
        })
        print('.', end='', flush=True)
        
        if global_step % 200 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
        if global_step % 10 == 0:
            model.eval()
            img_inr, labels = next(val_data_loader)
            with torch.no_grad():
                test_in_disc = discretizations['test_in']
                if util.is_model_nerfsampler(args) or ntype.startswith('mlp') or ntype.startswith('nuft'):
                    logits = model(img_inr, sampler=test_in_disc)
                else:
                    img = img_inr.produce_images(*dl_args['image shape'])
                    logits = model(img)
                loss = loss_fxn(logits, labels)
                pred_cls = logits.topk(k=3).indices
            
            wandb.log({'val_loss':loss.item(),
                'val_t3_acc': top3(pred_cls, labels).item(),
                'val_t1_acc': top1(pred_cls, labels).item(),
            })#, step=global_step

            # if global_step == 10: # only log once
            #     img = img_inr.produce_images(*dl_args['image shape'])
            #     wandb.log({f"img_val_{labels[0].item()}" : wandb.Image(img[0].detach().cpu().permute(1,2,0).numpy()),
            #     f"img_val_{labels[1].item()}" : wandb.Image(img[1].detach().cpu().permute(1,2,0).numpy()),
            #     f"img_val_{labels[2].item()}" : wandb.Image(img[2].detach().cpu().permute(1,2,0).numpy())})
        
        # if global_step == 1: # only log once
        #     img = img_inr.produce_images(*dl_args['image shape'])
        #     wandb.log({f"img_train_{labels[0].item()}" : wandb.Image(img[0].detach().cpu().permute(1,2,0).numpy()),
        #     f"img_train_{labels[1].item()}" : wandb.Image(img[1].detach().cpu().permute(1,2,0).numpy()),
        #     f"img_train_{labels[2].item()}" : wandb.Image(img[2].detach().cpu().permute(1,2,0).numpy())})
        
        if global_step >= args["optimizer"]["max steps"]:
            break

        # profiler.step()

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))

def test_inr_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1, N = 0,0,0
    origin = args['target_job']
    kwargs = {"data loading": args["data loading"]}
    model = load_model_from_job(origin, **kwargs).eval()
    orig_args = job_mgmt.get_job_args(origin)
    ntype = orig_args["network"]['type'].lower()
    if ntype.startswith('nuft'):
        model.sampler['type'] = 'rqmc'

    with torch.no_grad():
        model.eval()
        for img_inr, labels in data_loader:
            if util.is_model_nerfsampler(orig_args) or ntype.startswith('mlp') or ntype.startswith('nuft'):
                logits = model(img_inr)
            else:
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)

            pred_cls = logits.topk(k=3).indices
            top3 += (labels.unsqueeze(1) == pred_cls).amax(1).long().sum().item()
            top1 += (labels == pred_cls[:,0]).long().sum().item()
            N += labels.shape[0]

    open(osp.join(paths["job output dir"], "stats.txt"), 'w').write(f"{top3}, {top1}, {N}, {top3/N}")
    # torch.save((top1, top3, N), osp.join(paths["job output dir"], "stats.pt"))

def get_model(args):
    ntype = args["network"]["type"]
    depth = args['network'].get('depth', 3)
    C = args['network'].get('channels', 128)
    if ntype.lower().startswith('mlp'):
        return nerfsampler.baselines.hypernets.MLPCls(args['data loading']['classes'],
                depth=depth, C=C).cuda()
    in_ch = args['data loading']['input channels']
    n_classes = args['data loading']['classes']
    kwargs = dict(in_channels=in_ch, out_dims=n_classes)
    if args['optimizer'].get('discretization contrastive loss', None):
        kwargs['return_features'] = True
    
    if hasattr(nerfsampler.baselines.classifier, ntype):
        module = getattr(nerfsampler.baselines.classifier, ntype)
        model = module(**kwargs)
    elif hasattr(nerfsampler.inn.nets.classifier, ntype):
        kwargs = {**kwargs, **args["network"]['conv']}
        module = getattr(nerfsampler.inn.nets.classifier, ntype)
        model = module(**kwargs)
    elif ntype.lower().startswith('nuft'):
        model = nerfsampler.baselines.nuft.NUFTCls(grid_size=(32,32), C=C, depth=depth, **kwargs)
    elif ntype.startswith("Tx"):
        module = getattr(nerfsampler.baselines.classifier, ntype[2:])
        base = module(**kwargs)
        img_shape = args["data loading"]["image shape"]
        model, _ = inn.conversion.translate_discrete_model(base.layers, img_shape, sampler=sampler)
        # if args["data loading"]["discretization type"] != "grid":
        inn.nerfsampler.replace_conv_kernels(model, k_type='mlp', k_ratio=args["network"]["kernel expansion ratio"])
        # if net_args['frozen'] is True:
        #     inn.nerfsampler.freeze_layer_types(InrNet)
    else:
        raise NotImplementedError(f"Network type {ntype} not implemented")
        
    # if hasattr(model, 'layers'):
    #     wandb.watch(model.layers[0][0], log_freq=100)
    # else:
    #     wandb.watch(model[0][0], log_freq=100)

    return model.cuda()


def load_model_from_job(origin, **kwargs):
    orig_args = {**job_mgmt.get_job_args(origin), **kwargs}
    path = osp.expanduser(f"{RESULTS_DIR}/{origin}/weights/best.pth")
    model = get_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model
    