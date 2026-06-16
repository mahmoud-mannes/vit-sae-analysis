import torch

def predict(model, dataloader, condition, RPI= False, magnitude = 1.0):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Attach hook to apply RPI intervention

  if RPI:
    def RPI_hook(module, input, output):
      out = output.view(output.shape[0], output.shape[1], -1)

      perm = torch.randperm(out.shape[-1])
      return out[:,:,perm]

    if condition == "transformers":
      model.vit.embeddings.patch_embeddings.projection.register_forward_hook(RPI_hook)
    elif condition == "timm":
      model.patch_embed.proj.register_forward_hook(RPI_hook)

  # Scale positional encodings (for the PE magnitude scaling experiment)
  if condition == "transformers":
    try:
      model._modules['vit'].embeddings.position_embeddings = torch.nn.Parameter(model._modules['vit'].embeddings.position_embeddings * magnitude)
    except:
      pass
  elif condition == "timm":
    pass
  acc_list = [] # List of accuracies

  model.eval()
  model = model.half()
  device = model.device
  with torch.inference_mode():
    for images, labels in dataloader:
      images = images.to(device)
      outputs = model(**images)
      logits = outputs.logits
      predicted_class_idx = logits.argmax(-1)[0].to(device)
      accuracy = (predicted_class_idx == torch.tensor(labels).to(device)).sum()
      acc_list.append(accuracy)
      print(accuracy)
  
  return sum(acc_list) / len(acc_list)