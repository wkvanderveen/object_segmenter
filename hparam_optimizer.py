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
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tumor_detector
import csv
import pandas as pd
import os
import parameters as par

filename = 'results.csv'

var1 = par.hyperparameter1_search
var2 = par.hyperparameter2_search

if os.path.isfile(f"./{filename}"):
    os.remove(f"./{filename}")

par.prepare_dir(par.pred_dir, empty=True)


def write_to_csv(row):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


write_to_csv([var1['Name'], var2['Name'], 'Accuracy'])
run_count = 0
var1_range = range(var1["min_val"], var1["max_val"], var1["step"])
var2_range = range(var2["min_val"], var2["max_val"], var2["step"])

for val1 in var1_range:
    for val2 in var2_range:
        run_count += 1
        par.print_big_title(["Hyperoptimization:",
                             f"Progress: {run_count}/" +
                             str(len(var1_range)*len(var2_range)),
                             f"Now testing with:",
                             f"  {var1['Name']} = {val1},",
                             f"  {var2['Name']} = {val2}"])

        par.prepare_dir(par.model_dir, empty=True)
        config = ['tumor_detector.py',
                  [var1["Name"], val1],
                  [var2["Name"], val2]]

        write_to_csv([str(config[1][1]),
                      str(config[2][1]),
                      tumor_detector.main(config)])

par.print_big_title(["Results of hyperparameter optimization"])

df = pd.DataFrame(pd.read_csv("results.csv"))

print(df.sort_values(by=['Accuracy'], ascending=False))
