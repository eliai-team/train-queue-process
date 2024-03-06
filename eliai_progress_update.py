import base64
import io
import global_vars
import boto3
from supabase import Client, create_client
import os
import datetime


training_id = os.environ.get("TRAINING_ID")

url: str = os.environ.get('SUPABASE_ENDPOINT') or "https://rtfoijxfymuizzxzbnld.supabase.co"
key: str = os.environ.get('SUPABASE_KEY') or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0Zm9panhmeW11aXp6eHpibmxkIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTY1Nzc4MTQsImV4cCI6MjAxMjE1MzgxNH0.ChbqzCyTnUkrZ8VMie8y9fpu0xXB07fdSxVrNF9_psE"
supabase: Client = create_client(url, key)

s3client = boto3.client('s3', endpoint_url= os.environ.get('AWS_ENDPOINT') or 'https://hn.ss.bfcplatform.vn',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or "1XHPESY0IXGEMWYKN6PL",
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or "Kl8vL45TnD9KmVBcn2siOj9tno6rOimIxZlITSr1")
bucket_name = os.environ.get('BUCKET_NAME') or "eliai-server"
server_domain = os.environ.get('STORAGE_DOMAIN') or "https://eliai-server.hn.ss.bfcplatform.vn/"


def s3Storage_base64_upload(image_bytes: bytes, train_id: str, epoch: int):
    # image_binary = base64.b64decode(base64_image)
    object_key = f"training/{train_id}/samples/{train_id}_{epoch}.jpg"

    s3client.upload_fileobj(
      Fileobj=io.BytesIO(image_bytes),
      Bucket=bucket_name,
      Key=object_key,
      ExtraArgs={'ACL': 'public-read'}  # Optional: Set ACL to make the image public
    )

    return server_domain + object_key



def progress_update(time_elaped, rate, avr_loss, epoch, n, total):
  print(f"rate: {rate}, total: {total}, n: {n}")
  remaining = (total - n) / rate if rate and total else 0

  # data, count = supabase.table('Trainings').select('*').eq('id', training_id).execute()
  # update = data[1][0]
  # update['training_process_metadata'] = {
  #   "remaining": remaining,
  #   "time_elaped": time_elaped,
  #   "avr_loss": avr_loss,
  #   "steps": n,
  #   "total": total
  # }

  supabase.table("Trainings").update({
    'training_process_metadata':{
      "remaining": remaining,
      "time_elaped": time_elaped,
      "avr_loss": avr_loss,
      "steps": n,
      "total": total
    } 
  }).eq('id', training_id).execute()

def start_training():
  data, count = supabase.table('Trainings').update({
    "status": "training"
  }).eq('id', training_id).execute()

def finish_training():
  data, count = supabase.table('Trainings').update({
    "status": "done",
    "finished_at": datetime.datetime.utcnow().isoformat()
  }).eq('id', training_id).execute()

def sample_upload(image_bytes: bytes, epoch: int):
  print(f"epoch: {epoch}")
  # image_binary = base64.b64decode(base64_image)
  object_key = f"training/{training_id}/samples/{training_id}_{epoch}.jpg"

  s3client.upload_fileobj(
    Fileobj=io.BytesIO(image_bytes),
    Bucket=bucket_name,
    Key=object_key,
    ExtraArgs={'ACL': 'public-read'}  # Optional: Set ACL to make the image public
  )

  image_url = server_domain + object_key
  print(f"image_url {image_url}")

  data, count = supabase.table('Trainings').select('*').eq('id', training_id).execute()

  trained_epochs = data[1][0]["trained_epochs"]
  update_epoch_index = next((i for i, x in enumerate(trained_epochs) if x["epoch_num"] == epoch), None)
  update_epoch = trained_epochs[update_epoch_index]
  update_epoch['preview_url'] = image_url

  trained_epochs[update_epoch_index] = update_epoch

  data, count = supabase.table('Trainings').update({
    "trained_epochs": trained_epochs
  }).eq('id', training_id).execute()




def lora_upload(lora: bytes, epoch: int, ckpt_name:str,  avr_loss:float):
  # image_binary = base64.b64decode(base64_image)
  object_key = f"training/{training_id}/loras/{ckpt_name}"

  s3client.upload_fileobj(
    Fileobj=io.BytesIO(lora),
    Bucket=bucket_name,
    Key=object_key,
    ExtraArgs={'ACL': 'public-read'}  # Optional: Set ACL to make the image public
  )

  lora_url = server_domain + object_key
  print(f"lora_url {lora_url}")

  data, count = supabase.table('Trainings').select('*').eq('id', training_id).execute()

  trained_epochs = data[1][0]["trained_epochs"]
  print(f"Epoch: {epoch}")
  print(f"trained_epochs: {trained_epochs}")
  update_epoch_index = next((i for i, x in enumerate(trained_epochs) if x["epoch_num"] == epoch), None)
  print(f"epoch_index: {update_epoch_index}")

  update_epoch = trained_epochs[update_epoch_index]
  update_epoch['lora_url'] = lora_url
  update_epoch['loss'] = avr_loss
  update_epoch['name'] = ckpt_name

  trained_epochs[update_epoch_index] = update_epoch

  data, count = supabase.table('Trainings').update({
    "trained_epochs": trained_epochs
  }).eq('id', training_id).execute()
