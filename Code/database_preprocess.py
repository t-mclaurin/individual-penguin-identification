import os
import cv2

def apply_clahe_color(img):
    """Apply CLAHE to the L-channel of a color image in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def process_images(source_dir, destination_dir, valid_extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    for root, _, files in os.walk(source_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            source_path = os.path.join(root, filename)
            relative_path = os.path.relpath(source_path, source_dir)
            dest_path = os.path.join(destination_dir, relative_path)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            img = cv2.imread(source_path)
            if img is None:
                print(f"Warning: Failed to load {source_path}")
                continue

            clahe_img = apply_clahe_color(img)
            cv2.imwrite(dest_path, clahe_img)
            print(f"Processed and saved: {dest_path}")

if __name__ == "__main__":
    source_directory = "Penguin_Photos/indiviuals"
    output_directory = "Penguin_Photos/Histogram_normalised_individuals"

    process_images(source_directory, output_directory)
