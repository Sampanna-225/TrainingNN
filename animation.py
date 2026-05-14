import sys
import time
import threading

def loading_screen(duration:int=3) -> None:
    print("Initializing NeuralBrain...")
    chars = ["/", "-", "\\", "|"] # The rotating animation
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for char in chars:
            # \r moves the cursor to the start of the line
            sys.stdout.write(f"\rLoading {char} ")
            sys.stdout.flush() # to load the line immediately 
            time.sleep(0.1)

def parallel_screen(stop_command: threading.Event)->None:
    chars = ["/", "-", "\\", "|"] # The rotating animation
    while not stop_command.is_set():
        for c in chars:
            sys.stdout.write(f"\rTraining NN {c}")
            sys.stdout.flush()
            time.sleep(0.1)#delay
    sys.stdout.write("Training Complete")

def dot_animation(duration:int=1)->None:
    end = time.time() + duration
    chars = ['-','*-','**-','***-','****-','*****-',"******"]
    while time.time() < end:
        for c in chars:
            sys.stdout.write(f"\rThank You have a great day: {c}")
            sys.stdout.flush()
            time.sleep(0.2)