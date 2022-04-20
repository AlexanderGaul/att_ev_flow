import torch

from model import EventTransformer


model = EventTransformer()

torch.save(model, "/storage/user/gaul/gaul/thesis/output/test_model_checkpoint")


model.output_scale = 0.5

model = torch.load("/storage/user/gaul/gaul/thesis/output/test_model_checkpoint")

print(model.output_scale)
