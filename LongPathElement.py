import numpy as np
import cv2
import PointUtils

class LongPathElement:
    def __init__(self, branch_index, intersection_pts, intersection_pt_indices, end_pts=None, end_pt_indices=None):
        self.branch_index = branch_index
        self.intersection_pts = intersection_pts
        self.intersection_pt_indices = intersection_pt_indices
        self.end_pts = end_pts
        self.end_pt_indices = end_pt_indices
        
        # Set externally by the Measurement class, intersection points
        self.head_point = None
        self.tail_point = None
        
        # All set in the ProcessElement() method
        self.ordered_branch_points_full = None
        self.ordered_branch_points_adjusted = None
        self.total_pixel_length = 0
        self.total_adjusted_length = 0
        self.unit_length = 0
    
    def ProcessElement(self, base_branch_points, base_branch_length):
        """Orders and trims the constitutive branch based on the head and tail points (set externally) which serve as connectivity points with the adjacent branches

        Args:
            base_branch_points (np.ndarray): the (n,2) array of (y,x) pixel coordinate tuples that make up the branch
            base_branch_length (float): The branch length as calculated by FilFinder

        Returns:
            bool: Whether the operation completed successfully
        """
        # Perform point ordering to 100% certify resulting order
        base_branch_points = PointUtils.OptimizePath(base_branch_points)
        
        # Base attributions
        self.ordered_branch_points_full = base_branch_points
        self.ordered_branch_points_adjusted = base_branch_points
        self.total_pixel_length = base_branch_length
        self.total_adjusted_length = base_branch_length
        self.unit_length = self.total_pixel_length / len(self.ordered_branch_points_full)
        print("Branch contains {0} points and is {1:.2f} pixels long".format(np.shape(self.ordered_branch_points_full)[0], self.total_pixel_length))
        
        # Ensure the branch is ordered from head point to tail point
        head_kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + self.head_point - (2,2))
        head_bool_array = PointUtils.ContainsMutualPoints(head_kernel, base_branch_points, return_array=True)
        if not any(head_bool_array):
            return False
        head_index = np.argwhere(head_bool_array)[0][0]
        
        tail_kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + self.tail_point - (2,2))
        tail_bool_array = PointUtils.ContainsMutualPoints(tail_kernel, base_branch_points, return_array=True)
        if not any(tail_bool_array):
            return False
        tail_index = np.argwhere(tail_bool_array)[0][-1]
        
        # TODO: If the branch is very small, say 3 pixels, the tail point will incorrectly be considered "at the end" of the branch despite being at the front due to the (5,5) kernel size
        print("Head index: {0}; tail index: {1}".format(head_index, tail_index))
        if head_index > tail_index:
            print("Branch orientation is from tail to head, reversing")
            self.ordered_branch_points_full = np.flip(self.ordered_branch_points_full, axis=0)
        else:
            print("Branch is correctly oriented")
        
        # Determine if the head and tail points are at the extremes of the branch or not and adjust accoringly
        # It's important to start with the tail first, otherwise the indices found above will be invalidated
        if not PointUtils.PointInNeighborhood(self.tail_point, self.ordered_branch_points_adjusted[-1]):
            # The tail point occurs somewhere mid-branch, we need to trim from that point on
            trimmed_length = (len(self.ordered_branch_points_adjusted) - tail_index) * self.unit_length
            self.total_adjusted_length -= trimmed_length
            self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[:tail_index+1, :]
            print("Tail index is mid-branch, trimming {0:.2f} pixels".format(trimmed_length))
        else:
            print("Tail point is at the end of the list")
            
        if not PointUtils.PointInNeighborhood(self.head_point, self.ordered_branch_points_adjusted[0]):
            # The head point occurs somewhere mid-branch, we need to trim up until that point
            trimmed_length = head_index * self.unit_length
            self.total_adjusted_length -= trimmed_length
            self.ordered_branch_points_adjusted = self.ordered_branch_points_adjusted[head_index:, :]
            print("Head index is mid-branch, trimming {0:.2f} pixels".format(trimmed_length))
        else:
            print("Head point is at the beginning of the list")
        
        return True