from mvavatar import MVAvatar

model = MVAvatar.from_pretrained('models', device='cuda')

# image = model.inference(
#     '__base__, __color__, __special_costume__, __style__, __other__', seed=123
# )
# image = model.inference(
#     '__base__, __color__, __special_costume__, __style__, __other__'
# )
image = model.inference(
    'absurdres, highres, ultra detailed, 1 young beautiful girl, '
    'streetwear, print shirt , cargo pants, bomber jacket, high-top sneakers, snapback hats, bold patterns'
)
# image = model.inference('The Spider-Man, best quality, highres, marvel concept')

image.save('data/test.jpg')
