import torch

def predict(model, dataloader, RPI= False, magnitude = 1.0):
  # Attach hook to apply RPI intervention

  if RPI:
    def RPI_hook(module, input, output):
      out = output.view(output.shape[0], output.shape[1], -1)

      perm = torch.randperm(out.shape[-1])
      return out[:,:,perm]

    model.vit.embeddings.patch_embeddings.projection.register_forward_hook(RPI_hook)

  # Scale positional encodings (for the PE magnitude scaling experiment)
  
  model._modules['vit'].embeddings.position_embeddings = torch.nn.Parameter(model._modules['vit'].embeddings.position_embeddings * magnitude)
  
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