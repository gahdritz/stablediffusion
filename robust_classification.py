import argparse
from contextlib import nullcontext
import einops
import hashlib
import pickle
import os

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)


SD_IM_DIM = 256


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print(f"Missing keys: \n{m}")
    if len(u) > 0 and verbose:
        print(f"Unexpected keys: \n{u}")

    model.eval()
    return model


def load_model(args, verbose=False):
    config = OmegaConf.load(f"{args.config_path}")
    
    if(args.use_pickle_cache):
        unique_model_id = ''.join([
            args.config_path, args.ckpt_path,
        ])
        model_id_hash = hashlib.sha256(unique_model_id.encode())
        pickle_filename = model_id_hash.hexdigest() + ".pickle"
        pickle_path = os.path.join(args.pickle_cache_dir, pickle_filename)

        if(os.path.isfile(pickle_path)):
            print(f"Loading model from {pickle_path}...")
            with open(pickle_path, "rb") as fp:
                model = pickle.load(fp)     
        else:
            model = load_model_from_config(config, f"{args.ckpt_path}")
            os.makedirs(args.pickle_cache_dir, exist_ok=True)
            with open(pickle_path, "wb") as fp:
                pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model = load_model_from_config(config, f"{args.ckpt_path}")

    return model, config


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the classes
    with open(args.class_file, "r") as fp:
        classes = [l.strip() for l in fp.readlines()]

    print(classes)

    # Load the image
    img = Image.open(args.img_path)

    # Square images only
    assert(img.width == img.height)

    # Resize the image to 256 x 256, the size on which SD was trained
    #img = img.resize((SD_IM_DIM, SD_IM_DIM), resample=Image.Resampling.LANCZOS)

    img_np = np.array(img.convert("RGB"), copy=True)
    img_t = torch.as_tensor(img_np)
    img_t = img_t.to(device=device)
    img_t = img_t.to(dtype=torch.float32)

    # Load the model
    model, config = load_model(args) 
    model = model.to(device=device)

    # model is of type LatentDiffusion, so we need to encode the input
    img_t_reshaped = torch.stack([img_t for _ in range(args.batch_size)])
    img_t_reshaped = einops.rearrange(img_t_reshaped, 'b h w c -> b c h w')
    img_t_reshaped = img_t_reshaped.to(memory_format=torch.contiguous_format)
    
    # This seems to be important to be able to decode the image properly
    img_t_reshaped /= 255
    img_t_reshaped = img_t_reshaped * 2 - 1

    encoder_posterior = model.encode_first_stage(img_t_reshaped)
    latent = model.get_first_stage_encoding(encoder_posterior)

    # Inspect the decoded image
    #latent = latent[:1]
    #x_sample = model.decode_first_stage(latent)
    #x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    #x_sample = x_sample[0]
    #x_sample = 255 * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    #img = Image.fromarray(x_sample.astype(np.uint8))
    #path = f"tests/img.png"
    #img.save(path)

    # The codebase is poorly documented and so the positional encodings are
    # quite mysterious.
    assert(not model.use_positional_encodings)
 
    b = args.batch_size
    no_batches = len(classes) // b
    losses = [0. for _ in range(no_batches)]
    noise = torch.randn_like(latent[0])
    noise = torch.stack([noise for _ in range(b)])
    for t in [500]:#range(0, model.num_timesteps, 10):
        print(t)
        t_tensor = torch.tensor(
            [t for _ in range(args.batch_size)],
            device=device,
            dtype=torch.int64,
        )
        for i,c in enumerate([classes[i * b:(i + 1) * b] for i in range(no_batches)]):  
            # Prepare the conditioning
            print(c)
            cond = model.get_learned_conditioning(c)

            precision_scope = (
                torch.autocast if args.precision == "autocast" else nullcontext
            )
            with torch.no_grad(), \
                precision_scope("cuda"), \
                model.ema_scope():

                # Compute reconstruction loss
                x_noisy = model.q_sample(
                    x_start=latent, 
                    t=t_tensor, 
                    noise=noise
                )

                #x_samples = model.decode_first_stage(x_noisy)
                #x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                #for c, x_sample in zip(c, x_samples):
                #    x_sample = 255 * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                #    img = Image.fromarray(x_sample.astype(np.uint8))
                #    path = f"tests/{c}_{t}.png"
                #    img.save(path)

                model_output = model.apply_model(
                    x_noisy, 
                    t_tensor, 
                    cond
                )
 
                loss_dict = {}
                prefix = 'train' if model.training else 'val'

                if model.parameterization == "x0":
                    target = x_start
                elif model.parameterization == "eps":
                    target = noise
                elif model.parameterization == "v":
                    target = model.get_v(
                        latent, 
                        noise, 
                        t_tensor
                    )
                else:
                    raise NotImplementedError()

                loss_simple = model.get_loss(model_output, target, mean=False).mean([1, 2, 3])
                loss_dict.update({f'{prefix}/loss_simple': loss_simple})

                logvar_t = model.logvar.to(model.device)[t]
                loss = loss_simple / torch.exp(logvar_t) + logvar_t
                if model.learn_logvar:
                    loss_dict.update({f'{prefix}/loss_gamma': loss})
                    loss_dict.update({'logvar': model.logvar.data})

                loss *= model.l_simple_weight

                loss_vlb = model.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
                loss_vlb = (model.lvlb_weights[t] * loss_vlb)
                loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
                loss += (model.original_elbo_weight * loss_vlb)
                loss_dict.update({f'{prefix}/loss': loss})

            losses[i] += loss

    print(torch.cat(losses))

    print("it worked!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to image to be classified")
    parser.add_argument("config_path", type=str, help="Path to model config")
    parser.add_argument("ckpt_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "class_file", type=str, 
        help="Path to file containing classes, one per line"
    )
    parser.add_argument(
        "--precision", type=str, 
        choices=["full", "autocast"], default="autocast",
        help="Evaluate at this precision",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Classification batch size",
    )
    parser.add_argument(
        "--use_pickle_cache", action="store_true", default=False, 
        help="Whether to pickle models to improve load times"
    )
    parser.add_argument(
        "--pickle_cache_dir", type=str, default=None,
        help=(
            "Directory in which to save pickle files. Must be specified if "
            "--use_pickle_cache is active"
        )
    )
   
    args = parser.parse_args()

    main(args)
