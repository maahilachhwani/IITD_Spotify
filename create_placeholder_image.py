from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple placeholder image
img = Image.new('RGB', (400, 200), color='#1DB954')  # Spotify green
draw = ImageDraw.Draw(img)

# Add text
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

draw.text((100, 80), "Spotify", fill='white', font=font)

# Save the image
img.save('Spotify.jpg')
print("Created Spotify.jpg placeholder image") 