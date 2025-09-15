import os
import sys
import csv
import shutil
from tkinter import Tk, Canvas, Button, NW, mainloop, PhotoImage, messagebox
from PIL import Image, ImageTk

# ---- Configuration ----
SOURCE_DIR = '/Users/theomclaurin/Desktop/Penguin_Photos/fix/tofix'
DEST_DIR = '/Users/theomclaurin/Desktop/Penguin_Photos/fix/fixed'
PROGRESS_FILE = '/Users/theomclaurin/Desktop/Penguin_Photos/fix/progress.csv'
OCCLUDER_SIZE = 60  # default px size, change as needed
LARGE_OCCLUDER_SIZE = 90 # for large occluders
SUPER_OCCLUDER_SIZE = 120 # for super large occluders
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ---- Progress Management ----
def load_progress(progress_path):
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row['image_path']] = row
    return done

def append_progress(progress_path, row):
    exists = os.path.exists(progress_path)
    with open(progress_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'occluder_x', 'occluder_y', 'status', 'occluder_size'])
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---- File Utilities ----
def list_images(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                yield os.path.join(root, file)

def make_dest_path(src_path):
    rel = os.path.relpath(src_path, SOURCE_DIR)
    dest_path = os.path.join(DEST_DIR, rel)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    return dest_path

def copy_image(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)

def save_occluded_image(src, dest, occ_x, occ_y, size):
    image = Image.open(src).convert("RGB")
    draw = Image.new("RGB", image.size)
    draw.paste(image)
    # Ensure the occluder fits in the image
    left = max(0, occ_x - size // 2)
    top = max(0, occ_y - size // 2)
    right = min(image.width, left + size)
    bottom = min(image.height, top + size)
    # Draw black rectangle
    for y in range(top, bottom):
        for x in range(left, right):
            draw.putpixel((x, y), (0, 0, 0))
    draw.save(dest)

# ---- Main Annotation GUI ----
class ImageAnnotator:
    def __init__(self, image_paths, progress, occluder_size):
        self.image_paths = image_paths
        self.progress = progress
        self.occluder_size = OCCLUDER_SIZE  # store size with each placement
        self.idx = 0
        self.root = Tk()
        self.root.title("Occluder Tool")
        self.canvas = Canvas(self.root)
        self.canvas.pack()
        self.undo_button = Button(self.root, text="Undo", command=self.undo_occluder)
        self.undo_button.pack(side="left")
        self.skip_button = Button(self.root, text="Skip (no tag)", command=self.skip_image)
        self.skip_button.pack(side="left")
        self.save_exit_button = Button(self.root, text="Save & Exit", command=self.save_and_exit)
        self.save_exit_button.pack(side="right")
        self.confirm_button = Button(self.root, text="Confirm (Enter)", command=self.confirm_occlusion)
        self.confirm_button.pack(side="right")
        self.root.bind('<Button-1>', self.on_click)
        self.root.bind('<Return>', lambda event: self.confirm_occlusion())
        self.occluder = None  # (x, y) or None
        self.current_image = None
        self.tk_img = None
        self.load_image()
        self.quit_now = False

    def load_image(self):
        if self.idx >= len(self.image_paths):
            messagebox.showinfo("Done", "All images processed!")
            self.root.destroy()
            sys.exit(0)
        path = self.image_paths[self.idx]
        self.current_path = path
        pil_img = Image.open(path)
        # Resize to fit screen if huge
        screen_w = self.root.winfo_screenwidth() - 200
        screen_h = self.root.winfo_screenheight() - 200
        scale = min(screen_w/pil_img.width, screen_h/pil_img.height, 1.0)
        self.display_w = int(pil_img.width * scale)
        self.display_h = int(pil_img.height * scale)
        self.pil_img = pil_img.resize((self.display_w, self.display_h), Image.LANCZOS)
        self.canvas.config(width=self.display_w, height=self.display_h)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)
        self.occluder = None
        self.redraw()

    def on_click(self, event):
        # Detect Shift key for large, Shift+Control for super large occluder
        is_shift = (event.state & 0x0001) != 0  # Shift mask
        is_control = (event.state & 0x0004) != 0  # Control mask

        if is_shift and is_control:
            self.occluder_size = SUPER_OCCLUDER_SIZE
        elif is_shift:
            self.occluder_size = LARGE_OCCLUDER_SIZE
        else:
            self.occluder_size = OCCLUDER_SIZE

        if 0 <= event.x < self.display_w and 0 <= event.y < self.display_h:
            self.occluder = (event.x, event.y)
            self.redraw()


    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)
        if self.occluder:
            x, y = self.occluder
            half = self.occluder_size // 2
            left = max(0, x - half)
            top = max(0, y - half)
            right = min(self.display_w, x + half)
            bottom = min(self.display_h, y + half)
            self.canvas.create_rectangle(left, top, right, bottom, fill="black", outline="red", width=2)

    def undo_occluder(self):
        self.occluder = None
        self.redraw()

    def skip_image(self):
        # Copy image, log skip, go to next
        dest = make_dest_path(self.current_path)
        copy_image(self.current_path, dest)
        append_progress(PROGRESS_FILE, {
            'image_path': self.current_path,
            'occluder_x': '',
            'occluder_y': '',
            'status': 'skipped'
        })
        self.idx += 1
        self.load_image()

    def confirm_occlusion(self):
        if self.occluder:
            x, y = self.occluder
            orig = Image.open(self.current_path)
            scale_x = orig.width / self.display_w
            scale_y = orig.height / self.display_h
            occ_x = int(x * scale_x)
            occ_y = int(y * scale_y)
            occ_size = int(self.occluder_size * scale_x)  # scale size accordingly
            dest = make_dest_path(self.current_path)
            save_occluded_image(self.current_path, dest, occ_x, occ_y, occ_size)
            append_progress(PROGRESS_FILE, {
                'image_path': self.current_path,
                'occluder_x': occ_x,
                'occluder_y': occ_y,
                'occluder_size': occ_size,  # log which size was used
                'status': 'occluded'
            })
        else:
            self.skip_image()
            return
        self.idx += 1
        self.load_image()


    def save_and_exit(self):
        # Just close the window, progress file is updated per-image
        self.root.destroy()
        print("Progress saved. Exiting...")
        sys.exit(0)

def main():
    progress = load_progress(PROGRESS_FILE)
    images = [p for p in list_images(SOURCE_DIR) if p not in progress]
    if not images:
        print("No unprocessed images found.")
        return
    app = ImageAnnotator(images, progress, OCCLUDER_SIZE)
    mainloop()

if __name__ == "__main__":
    main()
