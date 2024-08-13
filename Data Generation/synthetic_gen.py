from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import pandas as pd
import string

# Adding Gaussian noise
def add_gaussian_noise(image):
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    row, col = image.size
    gauss = np.random.normal(mean, sigma, (col, row, 3))  # Ensure dimensions match
    noisy = np.array(image) + gauss * 255
    noisy = np.clip(noisy, 0, 255)
    noisy_image = Image.fromarray(np.uint8(noisy))
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.01):
    img_array = np.array(image)
    row, col, _ = img_array.shape
    num_salt = np.ceil(amount * img_array.size * 0.5)
    num_pepper = np.ceil(amount * img_array.size * 0.5)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 1

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0

    noisy_image = Image.fromarray(np.uint8(img_array))
    return noisy_image


image_count = 50000  # Number of images to generate
# Define a list of font attributes to choose from
font_attributes = [
    "arial.ttf",  # Font file, bold, italic 3/5,4/5
    "arialbd.ttf",  # Bold 
    "ariali.ttf",  # Italic
    "times.ttf",  # Different font
    "calibri.ttf",  # Another different font
    "verdana.ttf"   # Yet another different font 1/3,2/3,4/5,3/5
]
OpenSans_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("OpenSans")]
font_attributes = font_attributes + OpenSans_font_files

Roboto_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("Roboto")]
font_attributes = font_attributes + Roboto_font_files

Raleway_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("Raleway")]
font_attributes = font_attributes + Raleway_font_files

Oswald_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("Oswald")]
font_attributes = font_attributes + Oswald_font_files

Lato_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("Lato")]
font_attributes = font_attributes + Lato_font_files

Pacifico_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("Pacifico")]
font_attributes = font_attributes + Pacifico_font_files

SourceSansPro_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("SourceSansPro")]
font_attributes = font_attributes + SourceSansPro_font_files

PlayfairDisplay_font_files = [("fonts/"+file) for file in os.listdir("fonts") if file.startswith("PlayfairDisplay")]
font_attributes = font_attributes + PlayfairDisplay_font_files

print(len(font_attributes))

# Create a directory to save the generated images
output_directory = "fraction_images_val"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

data = []

for iterator in range(image_count):
    # Randomly select font attributes
    font_file = random.choice(font_attributes)
    font_size = random.randint(10, 20)
    font = ImageFont.truetype(font_file, font_size)

    # Randomly select a fraction
    special_char = [",",".",">","<","?","[","]","{","}","=","+","!","@","#","$","%","^","&","*","(",")","`","~",";","'","|",".",":"," ","—"]
    if font_file == "arial.ttf" or font_file == "arialbd.ttf" or font_file == "ariali.ttf":
         components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"", "⅛\"", "⅜\"","-","—"]
    elif font_file == "verdana.ttf":
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "¼\"", "⅛\"", "⅜\"","-","—"]
    elif font_file =="times.ttf" or font_file=="calibri.ttf":
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼", "⅗", "⅘", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"", "⅗\"", "⅘\"", "⅛\"", "⅜\"","-","—"]
    elif font_file.split("/")[1].startswith("OpenSans"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "¼\"", "⅛\"", "⅜\"","-","—"]
    elif font_file.split("/")[1].startswith("Roboto"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "¼\"", "⅛\"", "⅜\"","-","—"]
    elif font_file.split("/")[1].startswith("Raleway"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"", "⅛\"", "⅜\"","-","—"]
        
        
    elif font_file.split("/")[1].startswith("Oswald"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "¼", "/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"","¼\"","-"]

    elif font_file.split("/")[1].startswith("Lato"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼", "⅗", "⅘", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"", "⅗\"", "⅘\"", "⅛\"", "⅜\"","-"]

    elif font_file.split("/")[1].startswith("Pacifico"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "¼","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "¼\"","-"]
        # components = [ "½", "¾", "⅔", "⅓", "¼", "⅗", "⅘", "⅛", "⅜"]

    elif font_file.split("/")[1].startswith("SourceSansPro"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼", "⅛", "⅜","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"", "⅛\"", "⅜\"","-"]

    elif font_file.split("/")[1].startswith("PlayfairDisplay"):
        components = ["-","1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "½", "¾", "⅔", "⅓", "¼","/","X","x", "1\"", "2\"", "3\"", "4\"", "5\"", "6\"", "7\"", "8\"", "9\"", "0\"", "½\"", "¾\"", "⅔\"", "⅓\"", "¼\"","-"]       
        
        
    # Set the number of components to combine to form the numerical text
    num_components = random.randint(1, 3)  # You can adjust this as needed
    numerical_text = "".join(random.choice(components) for _ in range(num_components))
    numerical_text = numerical_text + random.choice(["","\"","mm","MM","\'"])

    # # Define components for each part of the text
    prefix = [''.join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 2)))]
    number1 = [str(i) for i in range(10000)]
    number2 = [str(i) for i in range(100)]
    hp = [''.join(random.choice(string.ascii_letters) for _ in range(2))]
    suffix = [str(i) for i in range(100)]

    # Randomly select from each component
    selected_prefix = random.choice(prefix)
    selected_number1 = random.choice(number1)
    selected_number2 = random.choice(number2)
    selected_hp = random.choice(hp)
    selected_suffix = random.choice(suffix)
    selected_suffix_2 = random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])

    # Create the random text
    random_text = [
        f"{selected_prefix} - {random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1} - {selected_number2}\" - {selected_hp} - {selected_suffix}{selected_suffix_2}",
        f"{selected_prefix} - {random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1}",
        f"{selected_prefix} -{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1} - {selected_number2}\" - {selected_hp}",
        f"{selected_prefix} -{selected_number1}",
        f"{selected_prefix} -{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}",
        f"{selected_prefix}",
        f"{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}",
        f"{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1}",
        f"{numerical_text}",
        f"{selected_prefix} - {random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{numerical_text} - {selected_number2}\" - {selected_hp} - {selected_suffix}{selected_suffix_2}",
        f"{selected_prefix} - {random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1} - {numerical_text}\" - {selected_hp} - {selected_suffix}{selected_suffix_2}"

        #specia char cases
        f"{selected_prefix}{random.choice(special_char)}{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1}{random.choice(special_char)}{selected_number2}\" {random.choice(special_char)} {selected_hp} {random.choice(special_char)} {selected_suffix}{selected_suffix_2}", 
        f"{selected_prefix} {random.choice(special_char)} {random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1}",
        f"{selected_prefix} {random.choice(special_char)}{random.choice([''.join(random.choice(string.ascii_uppercase) for _ in range(random.randint(0, 2)))])}{selected_number1} {random.choice(special_char)} {selected_number2}\" {random.choice(special_char)} {selected_hp}" 
    ]
    # text = random.choice()
    selected_prefix = random.choice(random_text)
    random_text = [numerical_text,selected_prefix]
    text = random.choice(random_text)
    # print(text)
    bbox = font.getbbox(text)
    if bbox[2]!=0 and bbox[3]!=0:
        # Create a blank image
        image = Image.new("RGB", (bbox[2],bbox[3]), (255,255,255))
        draw = ImageDraw.Draw(image)
        # Draw the fraction text on the image
        draw.text((0,0), text, fill="black", font=font)
        
        if random.random()>0.6:
            # Rotate the image by 90 degrees
            image = image.transpose(Image.ROTATE_90)
            label = 90
        else:
            label = 0
        # Save the image with a unique filename
        num_iterator = iterator
        image_filename = os.path.join(output_directory, f"fraction_{num_iterator}.png")
        if random.random()<0.15:
            image = add_gaussian_noise(image)
        elif 0.15<=random.random()<0.3:
            image = add_salt_and_pepper_noise(image)
        else:
            pass
        image.save(image_filename)
        # Replace the fractions
        text = text.replace("½", " 1/2 ")
        text = text.replace("¾", " 3/4 ")
        text = text.replace("⅔", " 2/3 ")
        text = text.replace("⅓", " 1/3 ")
        text = text.replace("¼", " 1/4 ")
        text = text.replace("⅗", " 3/5 ")
        text = text.replace("⅘", " 4/5 ")
        text = text.replace("⅛", " 1/8 ")
        text = text.replace("⅜", " 3/8 ")
        # label = text.replace("\"", "")
       
        data.append((f"fraction_{num_iterator}.png", text))
        # data.append((f"fraction_{num_iterator}.png", label))

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=['image_filename', 'fraction_label'])

# Shuffle the DataFrame if needed
df = df.sample(frac=1).reset_index(drop=True)

# Optionally, save the DataFrame to a CSV file
df.to_csv('fraction_labels_val.csv', index=False)

# Display the first few rows of the DataFrame
print(df.head())
print(f"Generated {image_count} images in the '{output_directory}' directory.")

import csv

csv_file = 'fraction_labels_val.csv'
txt_file = 'val_list.txt'

with open(csv_file, 'r') as csv_file, open(txt_file, 'w') as txt_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        image_filename = row['image_filename']
        fraction_label = row["fraction_label"]
        txt_file.write(image_filename+"\t"+fraction_label+"\n")