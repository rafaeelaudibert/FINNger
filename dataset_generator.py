from threading import Thread
import uuid

import cv2
import PySimpleGUI as sg
import click


key_pressed: bool = False
pictures_taken = 0


def detect_key_press():
    global key_pressed
    didnt_press_before = False

    while True:
        input("Press anything to take a screenshot")
        key_pressed = True
        if didnt_press_before:
            print("Thank you!")
            didnt_press_before = True


def save_image(image, identifier: str, path: str, size: int) -> None:
    global pictures_taken

    generated_uuid = uuid.uuid4()
    full_name = f"{generated_uuid}_{identifier}.png"

    resized_array = cv2.resize(image, (size, size))
    black_and_white = cv2.cvtColor(resized_array, cv2.COLOR_RGB2GRAY)

    full_path = path + full_name
    cv2.imwrite(full_path, black_and_white)

    pictures_taken += 1
    print(f"Saved image {pictures_taken} to {full_path}")

    print("Shape read back is", cv2.imread(full_path).shape)


@click.command()
@click.option("--identifier", required=True, help="The identifier appended to the end of the image")
@click.option("--path", required=True, help="The path where the photos will be saved on")
@click.option("--size", default=128, help="Size to resize the image to", show_default=True)
@click.option("--n_images", default=float("inf"), help="How many images we should generate", show_default=True)
def main(identifier: str, path: str, size: int, n_images: int):
    global key_pressed

    # Thread used to detect the key pressing
    thread = Thread(target=detect_key_press)
    thread.start()

    window = sg.Window(
        'Dataset Generator',
        [[sg.Image(filename='', key='image')], ],
        location=(800, 400),
    )

    cap = cv2.VideoCapture(0)  # Setup the camera as a capture device
    while True:
        # get events for the window with 20ms max wait
        event, _values = window.Read(timeout=20, timeout_key='timeout')
        if event is None:  # if user closed window, quit
            break

        _ret, image = cap.read()

        # Update image in window
        window_image = window.FindElement('image')
        encoded_image = cv2.imencode('.png', image)[1].tobytes()
        window_image.Update(data=encoded_image)

        # This is handled in a different thread, responsible for detecting the key press
        if key_pressed:
            save_image(image, identifier, path, size)
            key_pressed = False

            if pictures_taken >= n_images:
                print(
                    f"Finished generating {pictures_taken} images. Quitting application...")
                break


if __name__ == "__main__":
    main()
