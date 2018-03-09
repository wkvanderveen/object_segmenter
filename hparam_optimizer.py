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

import tumor_detector
import parameters as par
import utils
import pandas as pd
import csv
import os

# Set filename of csv file containing the search results
filename = 'results.csv'

var1 = par.hyperparameter1_search
var2 = par.hyperparameter2_search

# Overwrite the results file if it already exists
if os.path.isfile(f"./{filename}"):
    os.remove(f"./{filename}")

# Empty the folder where the prediction plots will be stored
utils.prepare_dir(par.pred_dir, empty=True)


def write_to_csv(row):
    """Write a text row to a csv file."""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


# Write the column titles to the results file.
write_to_csv([var1['Name'], var2['Name'], 'Accuracy'])

# Set a counter for terminal progress printing
run_count = 0

var1_range = range(var1["min_val"], var1["max_val"]+var1["step"], var1["step"])
var2_range = range(var2["min_val"], var2["max_val"]+var2["step"], var2["step"])

for val1 in var1_range:
    for val2 in var2_range:
        run_count += 1

        # Print a box containing progress information
        utils.print_big_title(["Hyperoptimization:",
                               f"Progress: {run_count}/" +
                               str(len(var1_range)*len(var2_range)),
                               f"Now testing with:",
                               f"  {var1['Name']} = {val1},",
                               f"  {var2['Name']} = {val2}"])

        # Empty the model directory before training the network
        utils.prepare_dir(par.model_dir, empty=True)
        config = ['tumor_detector.py',
                  [var1["Name"], val1],
                  [var2["Name"], val2]]

        # Run the network and write the results to the results file.
        accuracy = tumor_detector.main(config)
        write_to_csv([str(config[1][1]),
                      str(config[2][1]),
                      accuracy])
        print(f"The accuracy of this configuration is {accuracy:.4f}",
              f"and has been appended to the {filename} file.")

utils.print_big_title(["Results of hyperparameter optimization"])

# Construct a pretty-print data frame and sort by 'Accuracy'.
df = pd.DataFrame(pd.read_csv(filename))
print(df.sort_values(by=['Accuracy'], ascending=False))
