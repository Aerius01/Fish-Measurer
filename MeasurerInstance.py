from Cameras import Cameras
import cv2
import numpy as np
import statistics
import os
from datetime import datetime
import pandas as pd

from Measurement import Measurement

class MeasurerInstance():
    def __init__(self, outputFolder, format):
        ## 26.35
        self.outputFolder = outputFolder
        self.format = format
        self.background = None
        MeasurerInstance.threshold = None 
        MeasurerInstance.fishID = None
        MeasurerInstance.addText = None
        MeasurerInstance.error = (False, None)
        MeasurerInstance.processingFrame = None

        Cameras.ConnectMeasurer(self)

    def SubtractBackground(self, frame):
        # Subtract the background and binarize via thresholding (according to shadows setting)
        fgmask = self.fgbg.apply(frame, learningRate=0)
        fully_binarized = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Fill all blob contours and only show the contour with largest area
        try:
            contour, hier = cv2.findContours(fully_binarized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            biggestContour = max(contour, key = cv2.contourArea)
            
            self.im_bw = np.zeros(np.shape(fully_binarized)).astype('uint8')
            cv2.drawContours(self.im_bw, [biggestContour],-1,255,thickness=cv2.FILLED)
            
            if (MeasurerInstance.error[1] == "The threshold value is set too high and all blobs are being filtered out. Please lower it a bit"):
                MeasurerInstance.error = (False, None)
                
        except ValueError as e:
            print(str(type(e).__name__) + ": The biggestContour list is empty, no blobs are being picked up -->", e)
            self.im_bw = fully_binarized
            MeasurerInstance.error = (True, "The threshold value is set too high and all blobs are being filtered out. Please lower it a bit")
            
        return self.im_bw
        
    def TrainBackground(self):
        (background_images, empty) = Cameras.GetFixedNumFrames(Cameras.framerate * 1)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        for image in background_images:
            fgmask = self.fgbg.apply(image)
            
        self.background = fgmask
    
    def Analyze(self, frames):
        self.measurements = []
        (raw, binarized) = frames
        
        # Create the destination folder
        target_folder_name = None
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            target_folder_name = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + \
                "_ID-" + str(MeasurerInstance.fishID))
        else:
            target_folder_name = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
        
        if not os.path.isdir(target_folder_name):
            os.mkdir(target_folder_name)

        # Create the folder for the individual frames
        frames_path = os.path.join(target_folder_name, "frames")
        if not os.path.isdir(frames_path):
            os.mkdir(frames_path)
        
        # Run through each frame and process it
        for i in range(len(raw)):
            MeasurerInstance.processingFrame = i
            current_measurement = Measurement(i, raw[i], binarized[i])
            
            if current_measurement.successful_init:
                extended_frames_path = os.path.join(frames_path, str(i) + str(self.format))
                cv2.imwrite(extended_frames_path, current_measurement.skeleton)
                
                # Add it to the pool if successful
                if current_measurement.ConstructLongestPath():
                    self.measurements.append(current_measurement)
                    print("Frame: {0}; length (pix): {1}; length (cm): {2}".format(i, current_measurement.fil_length_pixels, Cameras.ConvertPixelsToLength(current_measurement.fil_length_pixels)))

                    skeleton_frames_path = os.path.join(frames_path, "full_skeleton_{0}{1}".format(i, self.format))
                    cv2.imwrite(skeleton_frames_path, current_measurement.skeleton_contour)
                    
        # For Tkinter ReinstateSettings()                    
        MeasurerInstance.processingFrame = None
        
        # Remove outliers & perform group-wide statistics
        if self.measurements:
            print("\ny = {0:.2f}x + {1:.2f}".format(Cameras.GetSlope(), Cameras.GetIntercept()))
            
            # Get the average length of all successful instances
            initial_list = [instance.fil_length_pixels for instance in self.measurements]
            avg_length = statistics.mean(initial_list)
            print("Avg length: {0:.2f}".format(avg_length))
            
            # Only retain any given measurement if its length is within 10% error of the average
            refined_list = [instance for instance in self.measurements if abs((instance.fil_length_pixels - avg_length) / avg_length) <= 0.1]
            MeasurerInstance.trial_count = len(refined_list)
            print("Refined list entries: " + str(MeasurerInstance.trial_count))
            
            if not refined_list:
                print("All lengths filtered out!")
                MeasurerInstance.error = (True, "The lengths obtained are too variant to be consolidated. The data will not be saved, please re-measure")
            else:
                self.length_stats = (statistics.mean([instance.fil_length_pixels for instance in self.measurements]), statistics.stdev([instance.fil_length_pixels for instance in self.measurements]))
                
                # Export the data to .csv
                df = pd.DataFrame(data={"frame_number": [instance.process_id for instance in self.measurements], "length_pix": [instance.fil_length_pixels for instance in self.measurements], "length_cm": [Cameras.ConvertPixelsToLength(instance.fil_length_pixels) for instance in self.measurements]})
                df.to_csv(os.path.join(target_folder_name, "data-output.csv"), sep=';',index=False) 
                
                # Find the instance with the closest length value
                try:
                    local_index = [instance.fil_length_pixels for instance in self.measurements].index(min([instance.fil_length_pixels for instance in self.measurements], key=lambda x:abs(x-self.length_stats[0])))
                    
                    closest_instance = [instance for instance in self.measurements][local_index]
                    closest_index = closest_instance.process_id
                except (KeyError):
                    MeasurerInstance.error = (True, "There was an error processing the obtained data. The data wsa not saved, please try again")
                
                print("\nFINAL\nAvg pix: {0:.2f}; Avg cm: {2:.2f}; Closest ID: {1}".format(self.length_stats[0],closest_index, Cameras.ConvertPixelsToLength(self.length_stats[0])))
                
                # Save principal image
                chosen_image = MeasurerInstance.WatermarkImage(closest_instance, self.length_stats)
                cv2.imwrite(os.path.join(target_folder_name, "closest-image" + str(self.format)), chosen_image)
                
                # Watermark and save all subsequent images
                for instance in self.measurements:
                    watermarked_image = MeasurerInstance.GenericWatermark(instance)
                    cv2.imwrite(os.path.join(target_folder_name, "watermarked-{0}{1}".format(instance.process_id, self.format)), watermarked_image)
                    
        else:
            MeasurerInstance.error = (True, "The length values could not be obtained from the image. Either the blob was too small and filtered out, or the skeletonization process was too complex. Please try again")
                
    def WatermarkImage(closest_instance, length_stats):
        # Watermark the results
        chosen_image = cv2.putText(closest_instance.processed_frame, 
                                    "Avg Length: {0:.2f}cm +/- {1:.2f}cm (This: {2:.2f}cm)".format(Cameras.ConvertPixelsToLength(length_stats[0]),
                                                                                                   Cameras.ConvertPixelsToLength(length_stats[1]), 
                                                                                                   Cameras.ConvertPixelsToLength(closest_instance.fil_length_pixels)), 
                                    (15, closest_instance.processed_frame.shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        chosen_image = cv2.putText(chosen_image, "{0} images".format(MeasurerInstance.trial_count),
                                    (15, chosen_image.shape[0]-120), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Add metadata
        chosen_image = cv2.putText(chosen_image, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            chosen_image = cv2.putText(chosen_image, "Fish ID: " + MeasurerInstance.fishID, (15, 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if MeasurerInstance.addText != None and MeasurerInstance.addText != '':
            text = MeasurerInstance.addText
            y0, dy = 250, 75
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                chosen_image = cv2.putText(chosen_image, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        return chosen_image
    
    def GenericWatermark(instance):
        chosen_image = cv2.putText(instance.processed_frame, 
                                    "Length: {0:.2f}cm)".format(Cameras.ConvertPixelsToLength(instance.fil_length_pixels)), 
                                    (15, instance.processed_frame.shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Add metadata
        chosen_image = cv2.putText(chosen_image, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            chosen_image = cv2.putText(chosen_image, "Fish ID: " + MeasurerInstance.fishID, (15, 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if MeasurerInstance.addText != None and MeasurerInstance.addText != '':
            text = MeasurerInstance.addText
            y0, dy = 250, 75
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                chosen_image = cv2.putText(chosen_image, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        return chosen_image

    def ShowImage(image1, image2, resize=0.5, name='Image', pausekey=False, show=False):
        image = cv2.addWeighted(image1,0.65,image2,0.35,0)
        if show:
            temp = cv2.resize(image, None, fy=resize, fx=resize)
            cv2.imshow(name, temp) 
            if pausekey:
                cv2.waitKey(0)
            
        return image
        
        
        