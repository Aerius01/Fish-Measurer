"""Main entry point for the Fish Measurer application."""

from gui import FishMeasurerApplication


def main():
    """Main application entry point."""
    # Create and run the application
    app = FishMeasurerApplication()
    app.run()


if __name__ == "__main__":
    main()


# PyInstaller
# Open Anaconda prompt and activate the cam-measurer environment: activate cam-measurer
# Navigate to directory: cd C:\Users\james\Desktop\04_Cam-Measurer\01_Code\02_Spec-Files
# Run it: pyinstaller main.spec

