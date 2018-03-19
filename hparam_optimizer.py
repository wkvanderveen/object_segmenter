"""Helper file for automatic hyperparameter grid search.
   This file should not be modified -- for changing variables, go to
   parameters.py.
   Copyright 2018 Werner van der Veen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied. See the License for the specific language governing
   permissions and limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import object_segmenter
import parameters as par
import utils
import pandas as pd

# Set filename of csv file containing the search results
FILENAME = 'results.csv'

VAR1 = par.hyperparameter1_search
VAR2 = par.hyperparameter2_search

# Overwrite the results file if it already exists
if os.path.isfile(f"./{FILENAME}"):
    os.remove(f"./{FILENAME}")

# Empty the folder where the prediction plots will be stored
utils.prepare_dir(par.pred_dir, empty=True)


def write_to_csv(row):
    """Write a text row to a csv file."""
    with open(FILENAME, 'a', newline='') as file:
        writer = csv.writer(file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


# Write the column titles to the results file.
write_to_csv([VAR1['Name'], VAR2['Name'], 'Accuracy'])

# Set a counter for terminal progress printing
RUN_COUNT = 0

VAR1_RANGE = range(VAR1["min_val"], VAR1["max_val"]+VAR1["step"], VAR1["step"])
VAR2_RANGE = range(VAR2["min_val"], VAR2["max_val"]+VAR2["step"], VAR2["step"])

for val1 in VAR1_RANGE:
    for val2 in VAR2_RANGE:
        RUN_COUNT += 1

        # Print a box containing progress information
        utils.print_big_title(["Hyperoptimization:",
                               f"Progress: {RUN_COUNT}/" +
                               str(len(VAR1_RANGE)*len(VAR2_RANGE)),
                               f"Now testing with:",
                               f"  {VAR1['Name']} = {val1},",
                               f"  {VAR2['Name']} = {val2}"])

        # Empty the model directory before training the network
        utils.prepare_dir(par.model_dir, empty=True)
        config = ['object_segmenter.py',
                  [VAR1["Name"], val1],
                  [VAR2["Name"], val2]]

        # Run the network and write the results to the results file.
        accuracy = object_segmenter.main(config)
        write_to_csv([str(config[1][1]),
                      str(config[2][1]),
                      accuracy])
        print(f"The accuracy of this configuration is {accuracy:.4f}",
              f"and has been appended to the {FILENAME} file.")

utils.print_big_title(["Results of hyperparameter optimization"])

# Construct a pretty-print data frame and sort by 'Accuracy'.
DATAFRAME = pd.DataFrame(pd.read_csv(FILENAME))
print(DATAFRAME.sort_values(by=['Accuracy'], ascending=False))
