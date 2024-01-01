from PIL import Image
import os

def rotate_and_save_images(input_folder, rotation_angle):
    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust file extensions as needed
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Rotate the image
            rotated_image = image.rotate(rotation_angle)

            # Save the rotated image back to the same folder
            rotated_image.save(image_path)

if __name__ == "__main__":
    # Set your input folders
    deer_folder = "tracks_data/deer"
    dog_folder = "tracks_data/dog"

    # Set the rotation angle (you can experiment with different angles)
    rotation_angle = 90

    # Rotate and save images in the deer folder
    rotate_and_save_images(deer_folder, rotation_angle)

    # Rotate and save images in the dog folder
    rotate_and_save_images(dog_folder, rotation_angle)
