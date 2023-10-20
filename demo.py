from mvavatar import MVAvatar

model = MVAvatar.from_pretrained('models', device='cuda')

# image = model.inference(
#     '__base__, __color__, __special_costume__, __style__, __other__', seed=123
# )
image = model.inference(
    '__base__, __color__, __special_costume__, __style__, __other__'
)
# image = model.inference(
#     'a girl'
# )
image.save('data/test.jpg')
