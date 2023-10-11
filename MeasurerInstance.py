from Cameras import Cameras
import cv2
import numpy as np
import statistics
import os
from datetime import datetime
import pandas as pd
import concurrent.futures as cf

from ProcessingInstance import ProcessingInstance

class MeasurerInstance():
    def __init__(self):
        # Instance variables
        # Set by app_constructor.py, once the start button is pressed
        self.outputFolder = None
        self.format = None
        
        # Folders where the each instance will save its respective images
        self.target_folder = None
        self.raw_folder = None
        self.skeleton_LP_folder = None
        self.watermarked_folder = None
        
        # Error handling
        self.background_is_trained = False
        self.block_tkinter_start_button = False
        
        # Class variables
        # Controlled by app_constructor.py
        MeasurerInstance.threshold = None 
        MeasurerInstance.fishID = None
        MeasurerInstance.addText = None
        
        # Error handling and aborting the run
        MeasurerInstance.errors = {key: [] for key in ["interrupt"]}
        MeasurerInstance.stop = False
        
        # Iterative Instancing
        MeasurerInstance.processingFrame = None
        MeasurerInstance.trial_count = 0
        MeasurerInstance.completed_threads = 0
        
        Cameras.ConnectMeasurer(self)

    def SubtractBackground(self, frame):
        # Subtract the background and binarize via thresholding (according to shadows setting)
        fgmask = self.fgbg.apply(frame, learningRate=0)
        fully_binarized = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Fill all blob contours and only show the contour with largest area
        try:
            contour, hier = cv2.findContours(fully_binarized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            biggestContour = max(contour, key = cv2.contourArea)
            
            self.im_bw = np.zeros(np.shape(fully_binarized)).astype('uint8')
            cv2.drawContours(self.im_bw, [biggestContour],-1,255,thickness=cv2.FILLED)
            self.block_tkinter_start_button = False
                
        except ValueError as e:
            self.block_tkinter_start_button = True
            print(str(type(e).__name__) + ": The biggestContour list is empty, no blobs are being picked up -->", e)
            
            return None
            
        return self.im_bw
        
    def TrainBackground(self):
        print("Pulling background frames")
        self.pulling_background = True
        (background_images, empty) = Cameras.GetFixedNumFrames(Cameras.framerate * 1)
        self.pulling_background = False
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        for image in background_images:
            self.fgbg.apply(image)
            
        self.background_is_trained = True
    
    def Analyze(self, frames):
        self.measurements = []
        (raw, binarized) = frames
        
        # Make the target location for image writing on the system
        self.CreateFolderStructure()
        
        # Run through each frame and process it
        instances = []
        for i in range(len(raw)):
            if MeasurerInstance.stop:
                break
            
            MeasurerInstance.processingFrame = i
            current_measurement = ProcessingInstance(i, raw[i], binarized[i], self.outputFolder)
            if current_measurement.successful_init:
                instances.append(current_measurement)
                
        # create a thread pool with the default number of worker threads
        MeasurerInstance.completed_threads = 0
        executor = cf.ThreadPoolExecutor()
        
        # https://stackoverflow.com/questions/67219755/python-threadpoolexecutor-map-with-instance-methods
        if not MeasurerInstance.stop:              
            # Construct the list of futures (async runners), each future corresponding to the processing of a single frame
            futures = [executor.submit(instance.ConstructLongestPath) for instance in instances]
            
            # Process results as they come in
            for future in cf.as_completed(futures):
                MeasurerInstance.completed_threads += 1
                
                instance = future.result() # blocks
                if isinstance(instance, ProcessingInstance):
                    if instance.successful_pathing:
                        self.measurements.append(instance)
                        print("\n".join(instance.output_log))
                        print("Frame: {0}; length (pix): {1:.2f}; length (cm): {2:.2f}".format(instance.process_id, instance.fil_length_pixels, Cameras.ConvertPixelsToLength(instance.fil_length_pixels)))

                        # Save the images
                        cv2.imwrite(os.path.join(self.raw_folder, "raw-{0}{1}".format(instance.process_id, self.format)), instance.raw_frame)
                        cv2.imwrite(os.path.join(self.skeleton_LP_folder, "skeleton_LP-{0}{1}".format(instance.process_id, self.format)), instance.skeleton_contour)
        
            executor.shutdown()
       
        # For Tkinter ReinstateSettings()                    
        MeasurerInstance.processingFrame = None
        
        # Remove outliers & perform group-wide statistics
        if not MeasurerInstance.stop:
            if self.measurements:
                refined_list = MeasurerInstance.RunStatistics(self.measurements)
                MeasurerInstance.trial_count = len(refined_list)
                
                if len(refined_list) < 2:
                    # Effectively interrupts the flow, the method ends after this block
                    print("All lengths filtered out!")
                    error_message = "The lengths obtained are too variant to be consolidated (the variance is too high). The data is unreliable and will not be saved, please re-measure"
                    
                    if error_message not in MeasurerInstance.errors["interrupt"]:
                        MeasurerInstance.errors["interrupt"].append(error_message)
                else:
                    MeasurerInstance.ExportData(self.watermarked_folder, self.target_folder, self.format, self.measurements)
            else:
                # Effectively interrupts the flow, the method ends after this block
                error_message = "No length values could be obtained from the collected images. Either the blob was too small and filtered out, or the skeletonization process was too complex and failed. Please try again"
                    
                if error_message not in MeasurerInstance.errors["interrupt"]:
                    MeasurerInstance.errors["interrupt"].append(error_message)
            
        MeasurerInstance.completed_threads = 0
                
    def WatermarkImage(instance, length_stats=None):
        # Watermark the closest result
        chosen_image = instance.processed_frame
        
        length_string = ""
        if length_stats is not None:
            # Mark number of trials
            trial_count = "{0} images".format(MeasurerInstance.trial_count)
            chosen_image = cv2.putText(chosen_image, trial_count, (15, chosen_image.shape[0]-120), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
            # Get the length mark
            length_string = "Avg Length: {0:.2f}cm +/- {1:.2f}cm (This: {2:.2f}cm)".format(length_stats[0], length_stats[1], Cameras.ConvertPixelsToLength(instance.fil_length_pixels))
        else:
            length_string = "Length: {0:.2f} cm".format(Cameras.ConvertPixelsToLength(instance.fil_length_pixels))
        
        # Apply length and date marks
        chosen_image = cv2.putText(chosen_image, length_string, (15, chosen_image.shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        chosen_image = cv2.putText(chosen_image, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Apply optional marks
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            chosen_image = cv2.putText(chosen_image, "Fish ID: " + MeasurerInstance.fishID, (15, 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if MeasurerInstance.addText != None and MeasurerInstance.addText != '':
            text = MeasurerInstance.addText
            y0, dy = 250, 75
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                chosen_image = cv2.putText(chosen_image, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        return chosen_image
    
    def RunStatistics(measurements):
        print("\ny = {0:.2f}x + {1:.2f}".format(Cameras.GetSlope(), Cameras.GetIntercept()))
                
        # Get the average length of all successful instances
        initial_list = [instance.fil_length_pixels for instance in measurements]
        avg_length = statistics.mean(initial_list)
        print("Avg length: {0:.2f}".format(avg_length))
        
        # Only retain any given measurement if its length is within 10% error of the average
        refined_list = [instance for instance in measurements if abs((instance.fil_length_pixels - avg_length) / avg_length) <= 0.1]
        trial_count = len(refined_list)
        print("Refined list entries: " + str(trial_count))
        
        return refined_list
    
    def ExportData(frames_path, target_folder_name, format, measurements):
        length_stats = (statistics.mean([Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in measurements]), statistics.stdev([Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in measurements]))
        
        # Export the data to .csv
        df = pd.DataFrame(data={"frame_number": [instance.process_id for instance in measurements], "length_in_pixels": [instance.fil_length_pixels for instance in measurements], "filfinder_length": [Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in measurements], "pixel_count_length": [Cameras.ConvertPixelsToLength(len(instance.long_path_pixel_coords)) for instance in measurements], "pixel_distance_length": [Cameras.ConvertPixelsToLength(instance.manual_length) for instance in measurements]})
        df.to_csv(os.path.join(target_folder_name, "data-output.csv"), sep=';',index=False) 
        
        # Find the instance with the closest length value
        local_index = [Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in measurements].index(min([Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in measurements], key=lambda x:abs(x-length_stats[0])))
        closest_instance = [instance for instance in measurements][local_index]
        closest_index = closest_instance.process_id
        
        print("\nFINAL\nAvg pix: {0:.2f}; Avg cm: {2:.2f}; Closest ID: {1}".format(Cameras.ConvertLengthToPixels(length_stats[0]),closest_index, length_stats[0]))
        
        # Save principal image
        chosen_image = MeasurerInstance.WatermarkImage(closest_instance, length_stats=length_stats)
        cv2.imwrite(os.path.join(target_folder_name, "closest-image" + str(format)), chosen_image)
        
        # # Watermark and save all subsequent images
        for instance in measurements:  
            if instance.process_id == closest_index:
                cv2.imwrite(os.path.join(frames_path, "watermarked-{0}{1}".format(instance.process_id, format)), chosen_image)
            else:
                watermarked_image = MeasurerInstance.WatermarkImage(instance)
                cv2.imwrite(os.path.join(frames_path, "watermarked-{0}{1}".format(instance.process_id, format)), watermarked_image)

    def CreateFolderStructure(self):
        # Create the destination folder and folder within it for the individual frames
        # Start with the instance folder at the top of the hierarchy
        self.target_folder = None
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            self.target_folder = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + \
                "_ID-" + str(MeasurerInstance.fishID))
        else:
            self.target_folder = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
        
        if not os.path.isdir(self.target_folder):
            os.mkdir(self.target_folder)

        # Create the frames directory, and then the subsequent three sub-directories
        frames_path = os.path.join(self.target_folder, "frames")
        if not os.path.isdir(frames_path):
            os.mkdir(frames_path)
            
        self.watermarked_folder = os.path.join(frames_path, "watermarked")
        if not os.path.isdir(self.watermarked_folder):
            os.mkdir(self.watermarked_folder)
            
        self.raw_folder = os.path.join(frames_path, "raw")
        if not os.path.isdir(self.raw_folder):
            os.mkdir(self.raw_folder)
        
        self.skeleton_LP_folder = os.path.join(frames_path, "skeleton_and_longpath")
        if not os.path.isdir(self.skeleton_LP_folder):
            os.mkdir(self.skeleton_LP_folder)