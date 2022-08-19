#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import json
from pathlib import Path

__all__ = ["MaskJsonUtils"]


class MaskJsonUtils:
    """Creates a JSON definition file for image masks."""

    def __init__(self, output_dir):
        """Initializes the class.
        Args:
        output_dir: the directory where the definition file will be saved"""
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """Adds a new category to the set of the corresponding super_category
        Args:
        category: e.g. 'eagle'
        super_category: e.g. 'bird'
        Returns:
        True if successful, False if the category was already in the dictionary"""
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True  # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """Takes an image path, its corresponding mask path, and its color categories,
        and adds it to the appropriate dictionaries
        Args:
        image_path: the relative path to the image, e.g. './images/00000001.png'
        mask_path: the relative path to the mask image, e.g. './masks/00000001.png'
        color_categories: the legend of color categories, for this particular mask,
            represented as a rgb-color keyed dictionary of category names and their super categories.
            (the color category associations are not assumed to be consistent across images)
        Returns:
        True if successful, False if the image was already in the dictionary"""
        if self.masks.get(image_path):
            return False  # image/mask is already in the dictionary

        # Create the mask definition
        mask = {"mask": mask_path, "color_categories": color_categories}

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item["category"], item["super_category"])

        return True  # Addition was successful

    def get_masks(self):
        """Gets all masks that have been added"""
        return self.masks

    def get_super_categories(self):
        """Gets the dictionary of super categories for each category in a JSON
        serializable format
        Returns:
        A dictionary of lists of categories keyed on super_category"""
        serializable_super_cats = dict()
        for super_cat, categories_set in self.super_categories.items():
            # Sets are not json serializable, so convert to list
            serializable_super_cats[super_cat] = list(categories_set)
        return serializable_super_cats

    def write_masks_to_json(self):
        """Writes all masks and color categories to the output file path as JSON"""
        # Serialize the masks and super categories dictionaries
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories()
        masks_obj = {
            "masks": serializable_masks,
            "super_categories": serializable_super_cats,
        }

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / "mask_definitions.json"
        with open(output_file_path, "w+") as json_file:
            json_file.write(json.dumps(masks_obj))
