from PIL import Image
import os

def make_combined_gifs(phase, input_dir="photo_test/", output_dir="animations/", total_epochs=224, ms_per_frame=60):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def build_gif(direction, out_name):

        print(f"Animation: combined_{direction}")

        frames = []
        for i in range(total_epochs):

            x_path = os.path.join(f"phase{phase}/{input_dir}", f"x_series_{direction}_{i}.png")
            y_path = os.path.join(f"phase{phase}/{input_dir}", f"y_series_{direction}_{i}.png")

            if (not os.path.isfile(x_path)) or (not os.path.isfile(y_path)):
                continue

            X = Image.open(x_path).convert("RGB")
            Y = Image.open(y_path).convert("RGB")

            # Make a single frame: [ X | Y ]
            W = X.width + Y.width
            H = max(X.height, Y.height)

            combined = Image.new("RGB", (W, H), (255, 255, 255))
            combined.paste(X, (0, 0))
            combined.paste(Y, (X.width, 0))

            frames.append(combined)

        if len(frames) > 0:
            frames[0].save(
                os.path.join(f"phase{phase}/{output_dir}", out_name),
                save_all=True,
                append_images=frames[1:],
                duration=ms_per_frame,
                loop=0
            )

    # Build combined GIFs 
    build_gif(direction="10", out_name="combined_10.gif")
    build_gif(direction="40", out_name="combined_40.gif")
