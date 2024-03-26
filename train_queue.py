
import json
import os
import shutil
import subprocess
import requests
import re
from supabase import Client, create_client
import toml
import global_vars
import requests
import redis
from accelerate.utils import write_basic_config

url: str = os.environ.get('SUPABASE_ENDPOINT') or "https://rtfoijxfymuizzxzbnld.supabase.co"
key: str = os.environ.get('SUPABASE_KEY') or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0Zm9panhmeW11aXp6eHpibmxkIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTY1Nzc4MTQsImV4cCI6MjAxMjE1MzgxNH0.ChbqzCyTnUkrZ8VMie8y9fpu0xXB07fdSxVrNF9_psE"
supabase: Client = create_client(url, key)

def json_to_toml_file(json_data, file_path):
  # dict = json.loads(json_data)
  # convert to toml and output to file
  with open(file_path, "w") as f:
      toml.dump(json_data, f)

def download_file(url, file_path):
  response = requests.get(url)
  if response.status_code == 200:
      with open(file_path, 'wb') as file:
          file.write(response.content)

def config_overwrite(dataset_config_file, config_file, project_path):
  with open(dataset_config_file, 'r') as f:
    dataset_config_lines = f.readlines()
  
  for i, line in enumerate(dataset_config_lines):
    if "image_dir" in line.strip():
      dataset_config_lines[i] = f'image_dir = "{os.path.join(project_path, "dataset")}"\n'
    elif "metadata_file" in line.strip():
      dataset_config_lines[i] = f'metadata_file = "{os.path.join(project_path, "dataset", "meta_lat.json")}"\n'
  with open(dataset_config_file, 'w') as f:
    f.writelines(dataset_config_lines)
  

  with open(config_file, 'r') as f:
    config_lines = f.readlines()
  
  for i, line in enumerate(config_lines):
    if "output_dir" in line.strip():
      config_lines[i] = f'output_dir = "{os.path.join(os.path.dirname(config_file),"output")}"\n'
    if "logging_dir" in line.strip():
      config_lines[i] = f'logging_dir = "{os.path.join(os.path.dirname(config_file),"_logs")}"\n'
    if "output_name" in line.strip():
      config_lines[i] = f'output_name = "{os.path.basename(project_path.rstrip("/"))}"\n'
    if "log_prefix" in line.strip():
      config_lines[i] = f'log_prefix = "{os.path.basename(project_path.rstrip("/"))}"\n'
    if "pretrained_model_name_or_path" in line.strip():
      config_lines[i] = f'pretrained_model_name_or_path = "{os.path.join("/", "basemodel.safetensors")}"\n'
  with open(config_file, 'w') as f:
    f.writelines(config_lines)


def start_training(training_id, log):
  data, count = supabase.table('Trainings').update({
    "status": "training",
    "debug": log
  }).eq('id', training_id).execute()

def read_txt_files(directory):
    file_contents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                file_contents[filename] = file.read()
    return file_contents


def caption(dataset_id, method, trigger_words, project_dir_path, images):
  # This will give you the full path to the script file
  script_path = os.path.abspath(__file__)

  # This will give you the directory of the script file
  script_dir = os.path.dirname(script_path)
  os.chdir(script_dir)
  project_dir_path = os.path.join(project_dir_path, "dataset")
  try:

    print("\n⭐ Starting caption...\n")
  
    print(f"train_data_dir: {project_dir_path}")

    script_name = ""
    if method == "BLIP":
      script_name = "make_captions.py"
    else:
      script_name = "tag_images_by_wd14_tagger.py"

    return_code = subprocess.run(f"python {script_dir}/finetune/{script_name} --batch_size 8 --max_data_loader_n_workers 2  --caption_extension=.txt {project_dir_path}", shell=True) 

    file_contents = read_txt_files(project_dir_path)
    for filename, content in file_contents.items():

      content = trigger_words + ", " + content

      image_index = next((i for i, x in enumerate(images) if filename.replace(".txt", "") in x["image_url"]), None)
      # caption_url = s3Storage_base64_upload(content_bytes, key)

      images[image_index]["caption"] = content

    supabase.table("Datasets").update({
      "images": images,
      "caption_status": "captioned"
    }).eq("id", dataset_id).execute()

  except subprocess.CalledProcessError as e:
    raise ValueError("Failed to caption")


def train(training_id):
  data, count = supabase.table("Trainings").select("*, dataset: dataset_id(*)").eq("id", training_id).execute()

  data = data[1]
  # print(data[0])
  # return
  training_config = data[0]["config"]
  # dataset_config = data[0]["dataset"]["config"]
  dataset_config = data[0]["dataset_config"]

  dataset = data[0]["dataset"]["images"]

  log = {
    "training_config": training_config,
    "dataset_config": dataset_config,
    "dataset": dataset
  }
  start_training(training_id, str(log))


  project_name = re.sub(r'\s+', '_', data[0]["dataset"]["name"]) 

  checkpoint_url = data[0]["model_url"]

  preview_prompt = data[0]["preview_prompts"]
  
  print("Data Accquire Success")

  #download checkpoint
  # download_file(checkpoint_url, os.path.join(os.getcwd(), "basemodel.safetensors"))
  print("Dowwnload ckpt Success")

  #download dataset
  project_dir_path = os.path.join(os.getcwd(), project_name) 
  os.makedirs(project_dir_path, exist_ok=True)
  os.makedirs(os.path.join(project_dir_path, "dataset") , exist_ok=True)

  for item in dataset:
    image_url = item.get("image_url")

    download_file(image_url,  os.path.join(project_dir_path, "dataset", image_url.split("/")[-1] ))
  print("Download File Success")

  # if data[0]["dataset"]["caption_status"] != "captioned":
  dataset_id = data[0]["dataset"]["id"]
  method = data[0]["dataset"]["caption_method"]
  trigger_words = data[0]["dataset"]["trigger_words"]
  proj_dir_path = project_dir_path
  images = data[0]["dataset"]["images"]
  caption(dataset_id, method, trigger_words, proj_dir_path, images)
  print("Captioned")


  # dataset_config_filepath = f"{project_dir_path}/dataset_config.toml"
  dataset_config_filepath = os.path.join(project_dir_path, "dataset_config.toml")
  json_to_toml_file(dataset_config, dataset_config_filepath)

  # training_config_filepath = f"{project_dir_path}/training_config.toml"
  training_config_filepath = os.path.join(proj_dir_path, "training_config.toml")
  json_to_toml_file(training_config, training_config_filepath)

  prompt_filepath = f"{project_dir_path}/prompt.txt"
  with open(prompt_filepath, 'w') as f:
    f.writelines(preview_prompt)

  config_overwrite(dataset_config_filepath, training_config_filepath, project_dir_path)
  print("Config File Setup Success")
  print("\n⭐ Starting trainer...\n")
  # os.chdir("/kohya_ss")
  
  accelerate_config_file = os.path.join(os.getcwd(), "accelerate.yaml")
  write_basic_config(save_location=accelerate_config_file)
  # prompt_path = "/prompt.txt"
  try:
    process = subprocess.run(f"accelerate launch --config_file={accelerate_config_file} --num_cpu_threads_per_process=1 train_network.py --dataset_config={dataset_config_filepath} --config_file={training_config_filepath} --sample_prompts={prompt_filepath}", shell=True) 
    if process.returncode != 0:
      data, count = supabase.table('Trainings').update({
        "status": "failed"
      }).eq('id', training_id).execute()
  except subprocess.CalledProcessError as e:
    # raise ValueError("Failed to train")
    data, count = supabase.table('Trainings').update({
      "status": "failed"
    }).eq('id', training_id).execute()
  finally:
    shutil.rmtree(project_dir_path)
    
  
def self_destroy():
  # id = os.environ.get("CONTAINER_ID")
  # key = os.environ.get("CONTAINER_API_KEY")

  # url = f"https://console.vast.ai/api/v0/instances/{id}/"

  # headers = {
  #   'Accept': 'application/json',
  #   'Authorization': f'Bearer {key}'
  # }

  # response = requests.request("DELETE", url, headers=headers)
  subprocess.run(f"/vast destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY", shell=True) 


def queue_processing():
  redis_uri = os.environ.get("REDIS_URI") or ""
  while 1:
    train_request = None

    redis_client = redis.from_url(redis_uri, decode_responses=True)
    train_request_id = redis_client.rpop('trainQueue')
    redis_client.close()

    if not train_request_id:
      break
    else:
      global_vars.training_metadata["id"] = train_request_id
      # print(f"train_request_id: {train_request_id}, training_id: {training_id}")

      os.environ["TRAINING_ID"] = train_request_id
      train(train_request_id)

  #seft destroy
  self_destroy()

if __name__ == "__main__":
    queue_processing()