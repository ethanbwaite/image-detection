from util import add_history, generate_saturated_color
import random, string
import numpy as np

class KnownObject:
    def __init__(self,
                  label,
                  bbox,
                  age=0, 
                  time_since_last_detection=0,
                  history=None,
                  color=(255, 255, 255),
                  max_history=100):
        # The objects unique ID
        self.id = self.generate_random_id()

        # The objects text descriptor label
        self.label = label

        # The objects bounding box
        self.bbox = bbox

        # The objects age in frames or ticks
        self.age = age

        # The objects time since last detection, in the same unit of age
        self.time_since_last_detection = time_since_last_detection

        # The objects bounding box history
        if history is None:
            history = []
        self.history = history

        # The objects color (b, g, r)
        self.color = color

        # The maximum length of the objects history
        self.max_history = max_history

    
    def add_box(self, new_bbox):
        """
        Update the object's box and add it to the history list
        """
        self.bbox = new_bbox
        self.history.append(new_bbox)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        print(len(self.history))

    def generate_random_id(self, length=6):
        # Generate a random 4-digit ID
        return ''.join(random.choices(string.digits, k=length))


    def __repr__(self):
        return f"KnownObject(id={self.id}, label='{self.label}', bbox={self.bbox}, age={self.age}, time_since_last_detection={self.time_since_last_detection}, history={self.history}, color={self.color}, max_history={self.max_history})"