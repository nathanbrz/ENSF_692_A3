"""
School Enrollment Statistics

This code provides classes and functions to handle and analyze school enrollment data.
It includes functionalities for setting up school data, looking up schools by name or code,
and performing statistical analysis on the enrollment data.

Author: Nathan De Oliveira (30113724)
Version: 1.0
Since: 2024-05-30
"""

import numpy as np
import pandas as pd
from given_data import year_2013, year_2014, year_2015, year_2016, year_2017, year_2018, year_2019, year_2020, year_2021, year_2022

class SchoolData:
    """
    Handles the school data including setting school names and IDs, and retrieving school data.

    @classvar data ((numpy.ndarray)): The reshaped school data (years, schools, grades).
    @classvar school_names_ids (dict): A dictionary mapping school names to their IDs or so called codes.
    @classvar school_ids_names (dict): A dictionary mapping school codes to their names.
    """
    def __init__(self, data, school_df):
        # Stack year arrays to make 2D array
        data = np.array(data)
        # Reshape array to separate grades (make it 3D array)
        self.data = data.reshape((10, 20, 3))
        self.school_names_ids, self.school_ids_names = self.set_schools_and_ids(school_df)

    def set_schools_and_ids(self, dataframe):
        """
        Creates a dictionaries mapping school names to their IDs/codes and school ids/codes to their names.

        @param dataframe (pandas.DataFrame): DataFrame containing school data with columns 'School Name' and 'School Code'.
        @return dict: Dictionary with school names as keys and school codes as values.
        @return dict: Dictionary with school codes as keys and school names as values.
        """
        school_names_ids = {}
        school_ids_names = {}
        school_data = dataframe[["School Name", "School Code"]].drop_duplicates().reset_index(drop=True)
        for _, row in school_data.iterrows():
            school_names_ids[row["School Name"]] = row["School Code"]

        for _, row in school_data.iterrows():
            school_ids_names[row["School Code"]] = row["School Name"]
    
        return school_names_ids, school_ids_names

    def data_shape_dim(self):
        """
        Returns the shape and number of dimensions of the data array.

        @return tuple: Shape of the data array.
        @return int: Number of dimensions of the data array.
        """
        return self.data.shape, self.data.ndim

    def look_up_school(self, id_or_name):
        """
        Looks up a school by its name or code.

        @param id_or_name (str or int): School name or school code.
        @return tuple: School name and school code.
        @raises ValueError: If the input is not a valid school name or code.
        """
        if id_or_name in self.school_names_ids.keys():
            school_name = id_or_name
            school_code = self.school_names_ids[id_or_name]
            return school_name, school_code
        elif int(id_or_name) in self.school_ids_names.keys():
            school_code = int(id_or_name)
            school_name = self.school_ids_names[int(id_or_name)]
            return school_name, school_code
        else:
            raise ValueError("ValueError: You MUST enter a VALID school name or code.\n")


    def get_school_data(self, school_code):
        """
        Retrieves data for a specific school by its code.

        @param school_code (int): The school code.
        @return numpy.ndarray: The data array for the specified school (years, one school by index, grades).
        """
        school_array_index = list(self.school_names_ids.values()).index(school_code)
        return self.data[:, school_array_index, :]

class Statistics:
    """
    Provides statistical analysis on school data.
    """
    @staticmethod
    def school_enrol_grade_means(school_data):
        """
        Calculates the mean enrollment for each grade (grade 10, grade 11, and grade 12) in a school.

        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return tuple: A tuple containing the mean enrollment for grade 10, grade 11, and grade 12 respectively.
        """
        mean_grade_10 = np.mean(school_data[:, 0])
        mean_grade_11 = np.mean(school_data[:, 1])
        mean_grade_12 = np.mean(school_data[:, 2])
        return mean_grade_10, mean_grade_11, mean_grade_12

    @staticmethod
    def school_enrol_max_min(school_data):
        """
        Finds the maximum and minimum enrollments in a school.
        
        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return tuple: Maximum and minimum enrollments.
        """
        highest_enrollment = np.max(school_data)
        lowest_enrollment = np.min(school_data)
        return highest_enrollment, lowest_enrollment

    @staticmethod
    def school_enrol_tot_per_year(school_data):
        """
        Calculates the total enrollment for each year in a school.

        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return scalar: the total enrollment for each year.
        """
        return np.sum(school_data, axis=1)

    @staticmethod
    def school_tot_ten_year(school_data):
        """
        Calculates the total enrollment over a ten-year period for a specific school.
        
        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return scalar: total enrollment over 10 years.
        """
        return np.sum(Statistics.school_enrol_tot_per_year(school_data))

    @staticmethod
    def mean_tot_enrol(school_data):
        """
        Calculates the mean total enrollment for a specific school over a ten-year period.

        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return scalar: The mean total enrollment over 10 years.
        """
        return np.mean(Statistics.school_enrol_tot_per_year(school_data))

    @staticmethod
    def median_over_500(school_data):
        """
        Calculates the median of enrollments that are greater than 500.

        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @return scalar: The median of enrollments that are greater than 500.
        """
        return np.median(school_data[school_data > 500])

    @staticmethod
    def print_school_stats(school_data):
        """
        Prints statistical information about a school's enrollment data.

        @param school_data (numpy.ndarray): The 2D data array for a specific school over the years wwith each grade. Shape (years, one school, grades).
        @print Mean enrollment for grade 10, grade 11, and grade 12, Highest and lowest enrollment for a single grade, 
        Total enrollment for each year, Total ten year enrollment, Mean total enrollment, Median enrollment over 500.
        """
        mean_grades = Statistics.school_enrol_grade_means(school_data)
        print(f"Mean enrollment for grade 10: {mean_grades[0]:.0f}\nMean enrollment for grade 11: {mean_grades[1]:.0f}\nMean enrollment for grade 12: {mean_grades[2]:.0f}")

        max_min_enroll = Statistics.school_enrol_max_min(school_data)
        print(f"Highest enrollment for a single grade: {max_min_enroll[0]:.0f}\nLowest enrollment for a single grade: {max_min_enroll[1]:.0f}")

        total_enrollment_per_year = Statistics.school_enrol_tot_per_year(school_data)
        for year, total in zip(range(2013, 2023), total_enrollment_per_year):
            print(f"Total enrollment in {year}: {total:.0f}")

        print(f"Total ten year enrollment: {Statistics.school_tot_ten_year(school_data):.0f}")
        print(f"Mean total enrollment over 10 years: {Statistics.mean_tot_enrol(school_data):.0f}")
        print(f"For all enrollments over 500, the median value was: {Statistics.median_over_500(school_data):.0f}")

    @staticmethod
    def general_stats(data_obj):
        """
        Calculates and prints general statistics for all schools over a ten-year period.

        @param data_obj (SchoolData): An instance of SchoolData class containing the school data.
        @print Mean enrollment in 2013. Mean enrollment in 2022. Total graduating class of 2022.
        Highest enrollment for a single grade. Lowest enrollment for a single grade.
        """
        mean_2013 = np.nanmean(data_obj.data[0, :, :])
        mean_2022 = np.nanmean(data_obj.data[9, :, :])
        total_grad_2022 = np.nansum(data_obj.data[9, :, 2])
        max_enrol_grade = np.nanmax(data_obj.data[:, :, :])
        min_enrol_grade = np.nanmin(data_obj.data[:, :, :])

        print(f"Mean enrollment in 2013: {mean_2013:.0f}")
        print(f"Mean enrollment in 2022: {mean_2022:.0f}")
        print(f"Total graduating class of 2022: {total_grad_2022:.0f}")
        print(f"Highest enrollment for a single grade: {max_enrol_grade:.0f}")
        print(f"Lowest enrollment for a single grade: {min_enrol_grade:.0f}")

def input_handler_for_part2(data_obj, user_input):
    """
    Handles user input for Part 2 of the program. It looks up the school by the user's input,
    prints the school's name and code, retrieves the school's data, and prints the school's statistics.

    @param data_obj (SchoolData): An instance of SchoolData class containing the school data.
    @param user_input (str or int): The user's input, which can be a school name or code.
    @print School Name: The name of the school. School Code: The code of the school.
    School Statistics.
    """
    name, code = data_obj.look_up_school(user_input)
    print(f"School Name: {name}, School Code: {code}")
    school_data = data_obj.get_school_data(code)
    Statistics.print_school_stats(school_data)

def input_handler_for_part3(data_obj):
    """
    Handles user input for Part 3 of the program. It calculates and prints general statistics for all schools over a ten-year period.

    @param data_obj (SchoolData): An instance of SchoolData class containing the school data.
    @print General Statistics.
    """
    Statistics.general_stats(data_obj)

def main():
    # Make while loop so that program runs despite the errors or end of code, until user says so.
    while (True):
        try:
            print("ENSF 692 School Enrollment Statistics")

            # Print Stage 1 requirements here

            # List of arrays to feed SchoolData class to stack them and reshape into 3D array (years, schools, grades).
            data = [year_2013, year_2014, year_2015, year_2016, year_2017,
                        year_2018, year_2019, year_2020, year_2021, year_2022]
            
            # Read in the school names and codes from the CSV file
            school_df = pd.read_csv("Assignment3Data.csv")

            # Use SchoolData Class who's constructor prepares the school data and the name/code dict.
            data_obj = SchoolData(data, school_df)

            # Get object's data array shape and dim. Print them.
            shape, dim = data_obj.data_shape_dim()
            print(f"Shape of the array: {shape}")
            print(f"Dimensions of the array: {dim}")

            # Prompt for user input 
            user_input = input("Enter a school name or school code (enter q to leave): ")

            # Allow user to select "q" to leave loop or continue prompt.
            if (user_input == "q"):
                break
            else:
                # Print Stage 2 requirements here
                print("\n***Requested School Statistics***\n")
                input_handler_for_part2(data_obj, user_input)

                # Print Stage 3 requirements here
                print("\n***General Statistics for All Schools***\n")
                input_handler_for_part3(data_obj)
                continue
        # The input handdler function for part 2 raises a ValueError if the school name or code is not found. 
        # This excepts that and prints the error and continues the loop for the user until the input q to leave.
        except ValueError as e:
            print(e)
            continue


if __name__ == '__main__':
    main()