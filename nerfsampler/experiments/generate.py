import os, pdb, torch, wandb
from nerfsampler.inn import point_set

from nerfsampler.utils import jobs as job_mgmt, util
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from nerfsampler.data import dataloader
from nerfsampler.utils import losses
import nerfsampler.inn.nets.wgan
from nerfsampler import RESULTS_DIR
import nerfsampler.baselines.wgan

rescale_float = mtr.ScaleIntensity()

n_ch = 1
def load_model(args):
    net_args = args["network"]
    if net_args["type"] == "wgan":
        G,D = nerfsampler.baselines.wgan.wgan()
    elif net_args["type"] == "inr-wgan":
        G,D = nerfsampler.inn.nets.wgan.translate_wgan_model()
    elif net_args["type"] == "inr-4":
        sampler = point_set.get_sampler_from_args(args['data loading'])
        # _,D = nerfsampler.models.wgan.Gan4(reshape=args['data loading']['initial grid shape'])
        # D, _ = inn.conversion.translate_discrete_model(D, args['data loading']['image shape'])
        G,D = nerfsampler.inn.nets.wgan.Gan4(sampler=sampler,
                reshape=args['data loading']['initial grid shape'])
    elif net_args["type"] == "cnn-4":
        G,D = nerfsampler.baselines.wgan.Gan4(reshape=args['data loading']['initial grid shape'])
    else:
        raise NotImplementedError('bad net type')
    wandb.watch((G,D), log="all", log_freq=100)
    return G.cuda(), D.cuda()

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    Gpath = osp.expanduser(f"{RESULTS_DIR}/{origin}/weights/G.pth")
    Dpath = osp.expanduser(f"{RESULTS_DIR}/{origin}/weights/D.pth")
    G,D = load_model(orig_args)
    G.load_state_dict(torch.load(Gpath))
    D.load_state_dict(torch.load(Dpath))
    return G,D

def train_generator(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    G_loss_fxn, D_loss_fxn = losses.adv_loss_fxns(args["loss settings"])
    D_step = args["optimizer"]['D steps']
    G,D = load_model(args)
    G_optim = util.get_optimizer(G, args, lr=args["optimizer"]["G learning rate"])
    D_optim = util.get_optimizer(D, args, lr=args["optimizer"]["D learning rate"])
    dims = dl_args['image shape']
    grid_outputs = dl_args['discretization type'] == 'grid' or not util.is_model_nerfsampler(args)

    for true_inr in data_loader:
        global_step += 1
        noise = torch.randn(dl_args["batch size"], 64, device='cuda')
        discretization = point_set.get_discretization_for_args(args)

        with torch.set_grad_enabled(mode=global_step % D_step == 0):
            if util.is_model_nerfsampler(args):
                print(true_inr.domain, flush=True)
                gen_inr = G(noise, discretization)
                fake_logits = D(gen_inr)
            else:
                gen_img = G(noise)
                fake_logits = D(gen_img)

        if global_step % D_step == 0:
            G_loss = G_loss_fxn(fake_logits)
            G_optim.zero_grad(set_to_none=True)
            G_loss.backward()
            G_optim.step()
            G_loss = G_loss.item()
            if np.isnan(G_loss):
                print('NaN G loss')
                pdb.set_trace()

            wandb.log({'train_G_loss': G_loss}, step=global_step)

            del fake_logits
            torch.cuda.empty_cache()

        if util.is_model_nerfsampler(args):
            gen_inr.values.detach()
            fake_logits = D(gen_inr)
            true_logits = D(true_inr)
            gp = losses.gradient_penalty_inr(true_inr, gen_inr, D)
        else:
            true_img = true_inr.produce_images(*dims)
            fake_logits = D(gen_img.detach())
            true_logits = D(true_img)
            gp = losses.gradient_penalty(true_img, gen_img, D)
        
        D_loss = D_loss_fxn(fake_logits, true_logits) + gp * args["loss settings"]["GP weight"]
        D_optim.zero_grad(set_to_none=True)
        D_loss.backward()
        D_optim.step()
        if torch.isnan(D_loss):
            print('NaN D loss')
            pdb.set_trace()
        wandb.log({'train_D_loss': D_loss, 'train_gradient_penalty': gp}, step=global_step)

        if global_step % 100 == 0:
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "G.pth"))
            torch.save(D.state_dict(), osp.join(paths["weights dir"], "D.pth"))

            if grid_outputs:
                wandb.log({'train_true': wandb.Image(true_img[0].permute(1,2,0)),
                    'train_fake': wandb.Image(gen_img[0].reshape(*dims,n_ch))},
                    step=global_step)
            
        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(G.state_dict(), osp.join(paths["weights dir"], "G.pth"))
    torch.save(D.state_dict(), osp.join(paths["weights dir"], "D.pth"))


def test_inr_generator(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    dataloader.get_inr_dataloader(dl_args)
    origin = args['target_job']
    G = load_model_from_job(origin)[0].cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)
    bsz = dl_args['batch size']
    num_samples = dl_args['N']
    dims = dl_args['image shape']
    assert num_samples % bsz == 0

    for ix in range(0, num_samples, bsz):
        disc = point_set.get_discretization_for_args(orig_args)
        noise = torch.randn(bsz, 64, device='cuda')
        with torch.no_grad():
            raise NotImplementedError

            for i in range(bsz):
                if orig_args["network"]['type'].startswith('inr'):
                    gen_img = rescale_float(gen_imgs[i].reshape(*dims,n_ch))
                else:
                    gen_img = rescale_float(gen_imgs[i].permute(1,2,0))
                    
            plt.imsave(paths["job output dir"]+f"/imgs/{ix+i}.png",
                gen_img.squeeze().detach().cpu().numpy(), cmap='gray')
