from ast import Raise
import os
import subprocess
from os import listdir
base_dir = '/root/Loras'
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
      config_lines[i] = f'pretrained_model_name_or_path = "./basemodel.safetensors"\n'
  with open(config_file, 'w') as f:
    f.writelines(config_lines)

def train(dataset_config_file, config_file, prompt_path):
  print("\n⭐ Starting trainer...\n")
  os.chdir("/kohya_ss")
  accelerate_config_file = "/accelerate.yaml"
  # prompt_path = "/prompt.txt"
  try:
    return_code = subprocess.run(f"accelerate launch --config_file={accelerate_config_file} --num_cpu_threads_per_process=1 train_network.py --dataset_config={dataset_config_file} --config_file={config_file} --sample_prompts={prompt_path}", shell=True) 
  except subprocess.CalledProcessError as e:
    raise ValueError("Failed to train")

def train_multiple_setup(project_path):

  list_config_dir = [config for config in listdir(project_path) if config.startswith("Config")]

  for config_dir in list_config_dir:
    dataset_config = os.path.join(project_path, config_dir, "dataset_config.toml")
    config = os.path.join(project_path, config_dir, "training_config.toml")
    prompt = os.path.join(project_path, "prompt.txt")

    try:
      config_overwrite(dataset_config, config, project_path)
      train(dataset_config, config, prompt )
    except subprocess.CalledProcessError as e:
      raise ValueError("Failed to train")

def train_multiple_project(project_path):
  # list_project_dir = [project for project in listdir(project_path) if "Done" not in project]

  while True:
    # is_done = False
    is_failed = False
    project_dir = [project for project in listdir(project_path) if "Done" and "Failed" not in project]

    if project_dir:
      project_dir = project_dir[0]
      print(f"Start Trainging project %s" % project_dir)
      try:
        train_multiple_setup(os.path.join(project_path,project_dir))
        # is_done = True
      except subprocess.CalledProcessError as e:
        is_failed = True

      if is_failed != False:
        os.rename(os.path.join(project_path,project_dir), os.path.join(project_path,project_dir + "_Done"))
      else:
        os.rename(os.path.join(project_path,project_dir), os.path.join(project_path,project_dir + "_Failed"))

# if __name__ == "__main__":
#   train_multiple_project(r"C:\Users\MinhNhat\Desktop\TrainData\Test")