import argparse
import sys
import os

# Add workspace directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera.camera import run_camera_test
from src.inference.realtime_inference import run_realtime_application
from src.inference.live_test import run_letter_test
from src.dataset.dataset_validator import inspect_dataset_directory
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("MainAppLauncher")

def display_interactive_menu():
    print("\n=======================================================")
    print("   REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION SYSTEM   ")
    print("=======================================================")
    print("Select Operational Mode:")
    print("  1. Test Laptop Camera Hardware (OpenCV Test)")
    print("  2. Run Static Sign Recognition (Live OpenCV Webcam)")
    print("  3. Run Dynamic Sign Recognition (Live OpenCV Webcam)")
    print("  4. Inspect Static Image Dataset (Dataset/ Directory)")
    print("  5. Guided Live Letter Test A-Z (saves results CSV)")
    print("  6. Guided Live Number Test 0-5 (saves results CSV)")
    print("  Q. Quit Application")
    print("=======================================================")
    
    choice = input("Enter option [1-6 or Q]: ").strip().upper()
    return choice

def main():
    parser = argparse.ArgumentParser(description="Real-Time ISL Recognition Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["camera_test", "static", "dynamic", "inspect", "letter_test", "number_test"],
        help="Operating mode: camera_test | static | dynamic | inspect | letter_test | number_test"
    )
    args = parser.parse_args()

    mode = args.mode

    if not mode:
        choice = display_interactive_menu()
        if choice == "1":
            mode = "camera_test"
        elif choice == "2":
            mode = "static"
        elif choice == "3":
            mode = "dynamic"
        elif choice == "4":
            mode = "inspect"
        elif choice == "5":
            mode = "letter_test"
        elif choice == "6":
            mode = "number_test"
        elif choice == "Q":
            print("Exiting ISL Recognition Application. Goodbye!")
            return
        else:
            print("Invalid option selected. Exiting.")
            return

    logger.info(f"Launching ISL Recognition System in mode: {mode.upper()}")

    if mode == "camera_test":
        run_camera_test()
    elif mode == "static":
        run_realtime_application(mode="STATIC")
    elif mode == "dynamic":
        run_realtime_application(mode="DYNAMIC")
    elif mode == "letter_test":
        run_letter_test()
    elif mode == "number_test":
        run_letter_test(numbers=True)
    elif mode == "inspect":
        training_dir = resolve_path("Dataset/Training")
        target_dir = training_dir if training_dir.exists() else resolve_path("Dataset")
        if target_dir.exists():
            inspect_dataset_directory(target_dir)
        else:
            print(f"[ERROR] Dataset directory '{target_dir}' not found.")

if __name__ == "__main__":
    main()
