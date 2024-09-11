import os, json, requests, runpod

import random, time
import torch
import numpy as np
from PIL import Image
import nodes
from nodes import NODE_CLASS_MAPPINGS
from nodes import load_custom_node
from comfy_extras import nodes_custom_sampler
from comfy_extras import nodes_flux
from comfy import model_management

load_custom_node("/content/ComfyUI/custom_nodes/comfyui_controlnet_aux")
load_custom_node("/content/ComfyUI/custom_nodes/x-flux-comfyui")

CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
XlabsSampler = NODE_CLASS_MAPPINGS["XlabsSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoadFluxControlNet = NODE_CLASS_MAPPINGS["LoadFluxControlNet"]()
ApplyFluxControlNet = NODE_CLASS_MAPPINGS["ApplyFluxControlNet"]()
LoadImage =  NODE_CLASS_MAPPINGS["LoadImage"]()
DepthAnythingV2Preprocessor =  NODE_CLASS_MAPPINGS["DepthAnythingV2Preprocessor"]()
CLIPTextEncodeFlux = nodes_flux.NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()

with torch.inference_mode():
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("flux1-dev-fp8-all-in-one.safetensors")
    controlnet = LoadFluxControlNet.loadmodel(model_name="flux-dev", controlnet_path="flux-depth-controlnet-v3.safetensors")[0]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    controlnet_image = values['input_image_check']
    controlnet_image = download_file(url=controlnet_image, save_dir='/content/ComfyUI/input')
    controlnet_strength = values['controlnet_strength']
    final_width = values['final_width']
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    guidance = values['guidance']
    lora_strength_model = values['lora_strength_model']
    lora_strength_clip = values['lora_strength_clip']
    custom_lora_strength_model = values['custom_lora_strength_model']
    custom_lora_strength_clip = values['custom_lora_strength_clip']
    lora_file = values['lora_file']
    custom_lora_url = values['custom_lora_url']
    custom_lora_file = download_file(url=custom_lora_url, save_dir='/content/ComfyUI/models/loras')
    custom_lora_file = os.path.basename(custom_lora_file)

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    custom_lora_unet, custom_lora_clip = LoraLoader.load_lora(unet, clip, custom_lora_file, custom_lora_strength_model, custom_lora_strength_clip)
    lora_unet, lora_clip = LoraLoader.load_lora(custom_lora_unet, custom_lora_clip, lora_file, lora_strength_model, lora_strength_clip)
    conditioning = CLIPTextEncodeFlux.encode(lora_clip, positive_prompt, positive_prompt, 4.0)[0]
    neg_conditioning = CLIPTextEncodeFlux.encode(lora_clip, negative_prompt, negative_prompt, 4.0)[0]
    controlnet_image_width, controlnet_image_height = Image.open(controlnet_image).size
    controlnet_image_aspect_ratio = controlnet_image_width / controlnet_image_height
    final_height = final_width / controlnet_image_aspect_ratio
    controlnet_image = LoadImage.load_image(controlnet_image)[0]
    controlnet_depth = DepthAnythingV2Preprocessor.execute(controlnet_image, "depth_anything_v2_vitl.pth", resolution=1024)[0]
    controlnet_condition = ApplyFluxControlNet.prepare(controlnet, controlnet_depth, controlnet_strength)[0]
    latent_image = EmptyLatentImage.generate(closestNumber(final_width, 16), closestNumber(final_height, 16))[0]
    sample = XlabsSampler.sampling(model=lora_unet, conditioning=conditioning, neg_conditioning=neg_conditioning,
                            noise_seed=seed, steps=steps, timestep_to_start_cfg=1, true_gs=guidance,
                            image_to_image_strength=0, denoise_strength=1,
                            latent_image=latent_image, controlnet_condition=controlnet_condition)[0]
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/tost_flux_pose_lora.png")

    result = "/content/tost_flux_pose_lora.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})