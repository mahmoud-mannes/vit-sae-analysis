This is essentially just a dump of the top positional feature candidates based on token selectivity (comparing activations of every feature at each position from 0 to 196 to activations at other positions), then sorting the top ones from those based on row/column selectivity. Note that while the word "selectivity" is used twice here, the metrics are fundamentally different. Token selectivity compares the mean activations of the features at one specific location to their mean activations at other locations, meanwhile row/column selectivity metrics extract the heatmap of when the features activate most, and calculate whether the activations are more concentrated in a specific row or column than others. 
It's important to note that values greater than 3 suggest decent row/column selectivity, and values greater than 5 indicate great row/column selectivity. You can also visualize these features using modules from the SAE_feature_analysis directory of the repo.
Another important note is that the current row/column selectivity metric will be replaced by an entropy-based one, the current way the metric is calculated (which can be found in the repo) is simple, but temporary and imperfect.


----- ROW 0 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 5 -----
Feature 1533, Column selectivity 1.5916825532913208
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 6 -----
Feature 1533, Column selectivity 1.5916825532913208
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 7 -----
Feature 1352, Column selectivity 1.3423168659210205
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 9 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 10 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 11 -----
Feature 1223, Column selectivity 2.1999118328094482
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 3140, Row selectivity 6.149264335632324
----- ROW 0 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 2 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 4 -----
Feature 1800, Column selectivity 2.204594373703003
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 5 -----
Feature 1533, Column selectivity 1.5916825532913208
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 6 -----
Feature 1533, Column selectivity 1.5916825532913208
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 7 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 9 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 11 -----
Feature 2516, Column selectivity 2.2405333518981934
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 12 -----
Feature 2442, Column selectivity 2.966299057006836
Feature 3140, Row selectivity 6.149264335632324
----- ROW 1 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 2 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 3 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 4 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 5 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 6 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 7 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 9 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 3140, Row selectivity 6.149264335632324
----- ROW 2 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 11 -----
Feature 2516, Column selectivity 2.2405333518981934
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 12 -----
Feature 2442, Column selectivity 2.966299057006836
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 2 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 3 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 2 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 3 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 4 -----
Feature 1800, Column selectivity 2.204594373703003
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 5 -----
Feature 1800, Column selectivity 2.204594373703003
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 6 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 7 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 9 -----
Feature 1223, Column selectivity 2.1999118328094482
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 11 -----
Feature 2516, Column selectivity 2.2405333518981934
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 3 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 4 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 4 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 4 COLUMN 2 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 4 -----
Feature 1800, Column selectivity 2.204594373703003
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 5 -----
Feature 3688, Column selectivity 2.049085855484009
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 6 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 7 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 9 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 4 COLUMN 11 -----
Feature 1223, Column selectivity 2.1999118328094482
Feature 1533, Row selectivity 2.8932318687438965
----- ROW 4 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 516, Row selectivity 1.345036506652832
----- ROW 4 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 5 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 2 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 5 -----
Feature 3688, Column selectivity 2.049085855484009
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 6 -----
Feature 3688, Column selectivity 2.049085855484009
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 7 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 8 -----
Feature 3591, Column selectivity 1.760025978088379
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 9 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 11 -----
Feature 1223, Column selectivity 2.1999118328094482
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 705, Row selectivity 3.3901526927948
----- ROW 5 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 1 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 2 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 6 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 7 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 8 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 9 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 11 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 705, Row selectivity 3.3901526927948
----- ROW 6 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 302, Row selectivity 1.8945960998535156
----- ROW 7 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 1389, Row selectivity 1.5874621868133545
----- ROW 7 COLUMN 2 -----
Feature 3715, Column selectivity 2.268659830093384
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 6 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 7 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 8 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 9 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 705, Row selectivity 3.3901526927948
----- ROW 7 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 3435, Row selectivity 2.385411500930786
----- ROW 7 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 8 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1029, Row selectivity 1.9501012563705444
----- ROW 8 COLUMN 6 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1029, Row selectivity 1.9501012563705444
----- ROW 8 COLUMN 7 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 8 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 9 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 3435, Row selectivity 2.385411500930786
----- ROW 8 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 9 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1389, Row selectivity 1.5874621868133545
----- ROW 9 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 1389, Row selectivity 1.5874621868133545
----- ROW 9 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 9 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 9 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 3435, Row selectivity 2.385411500930786
----- ROW 9 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1029, Row selectivity 1.9501012563705444
----- ROW 9 COLUMN 6 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1029, Row selectivity 1.9501012563705444
----- ROW 9 COLUMN 7 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 1029, Row selectivity 1.9501012563705444
----- ROW 9 COLUMN 8 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1197, Row selectivity 2.807244062423706
----- ROW 9 COLUMN 9 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 9 COLUMN 10 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 9 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 516, Row selectivity 1.345036506652832
----- ROW 9 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 3435, Row selectivity 2.385411500930786
----- ROW 9 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 10 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 1389, Row selectivity 1.5874621868133545
----- ROW 10 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 589, Row selectivity 1.7049126625061035
----- ROW 10 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 6 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 7 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 8 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 9 -----
Feature 670, Column selectivity 1.7149373292922974
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 10 -----
Feature 2409, Column selectivity 2.1292147636413574
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 10 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 516, Row selectivity 1.345036506652832
----- ROW 11 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 3388, Row selectivity 1.832126498222351
----- ROW 11 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 5 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 6 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 7 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 8 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 9 -----
Feature 670, Column selectivity 1.7149373292922974
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 10 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 11 COLUMN 13 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 1197, Row selectivity 2.807244062423706
----- ROW 12 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 2 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 5 -----
Feature 1197, Column selectivity 1.694156527519226
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 6 -----
Feature 3324, Column selectivity 1.6924829483032227
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 7 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 8 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 9 -----
Feature 370, Column selectivity 1.7182999849319458
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 10 -----
Feature 670, Column selectivity 1.7149373292922974
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 2974, Row selectivity 8.869734764099121
----- ROW 12 COLUMN 13 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 0 -----
Feature 2429, Column selectivity 6.257824897766113
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 1 -----
Feature 1389, Column selectivity 3.0918514728546143
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 2 -----
Feature 2208, Column selectivity 2.76371169090271
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 3 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 4 -----
Feature 589, Column selectivity 2.8215482234954834
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 5 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 6 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 7 -----
Feature 1029, Column selectivity 1.8203742504119873
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 8 -----
Feature 3059, Column selectivity 2.124983549118042
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 9 -----
Feature 670, Column selectivity 1.7149373292922974
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 10 -----
Feature 670, Column selectivity 1.7149373292922974
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 11 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 12 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 2974, Row selectivity 8.869734764099121
----- ROW 13 COLUMN 13 -----
Feature 516, Column selectivity 3.5803792476654053
Feature 2974, Row selectivity 8.869734764099121

{(0, 0): (3140, 2429),
 (0, 1): (3140, 2208),
 (0, 2): (3140, 589),
 (0, 3): (3140, 589),
 (0, 4): (3140, 589),
 (0, 5): (3140, 1533),
 (0, 6): (3140, 1533),
 (0, 7): (3140, 1352),
 (0, 8): (3140, 3591),
 (0, 9): (3140, 370),
 (0, 10): (3140, 370),
 (0, 11): (3140, 1223),
 (0, 12): (3140, 516),
 (0, 13): (3140, 2429),
 (1, 0): (3140, 2429),
 (1, 1): (3140, 2208),
 (1, 2): (3140, 3715),
 (1, 3): (3140, 589),
 (1, 4): (3140, 1800),
 (1, 5): (3140, 1533),
 (1, 6): (3140, 1533),
 (1, 7): (3140, 3591),
 (1, 8): (3140, 3591),
 (1, 9): (3140, 370),
 (1, 10): (3140, 2409),
 (1, 11): (3140, 2516),
 (1, 12): (3140, 2442),
 (1, 13): (3140, 2429),
 (2, 0): (1533, 2429),
 (2, 1): (1533, 2208),
 (2, 2): (1533, 3715),
 (2, 3): (1533, 3715),
 (2, 4): (3140, 3591),
 (2, 5): (3140, 3591),
 (2, 6): (3140, 3591),
 (2, 7): (3140, 3591),
 (2, 8): (3140, 3591),
 (2, 9): (3140, 3591),
 (2, 10): (1533, 2409),
 (2, 11): (1533, 2516),
 (2, 12): (1533, 2442),
 (2, 13): (516, 2429),
 (3, 0): (1533, 2429),
 (3, 1): (1533, 2208),
 (3, 2): (1533, 3715),
 (3, 3): (1533, 3715),
 (3, 4): (1533, 1800),
 (3, 5): (1533, 1800),
 (3, 6): (1533, 3591),
 (3, 7): (1533, 3591),
 (3, 8): (1533, 3591),
 (3, 9): (1533, 1223),
 (3, 10): (1533, 2409),
 (3, 11): (1533, 2516),
 (3, 12): (1533, 516),
 (3, 13): (516, 2429),
 (4, 0): (1533, 2429),
 (4, 1): (1533, 2208),
 (4, 2): (705, 2208),
 (4, 3): (705, 589),
 (4, 4): (705, 1800),
 (4, 5): (705, 3688),
 (4, 6): (705, 3591),
 (4, 7): (705, 3591),
 (4, 8): (705, 3591),
 (4, 9): (705, 3591),
 (4, 10): (705, 2409),
 (4, 11): (1533, 1223),
 (4, 12): (516, 516),
 (4, 13): (516, 2429),
 (5, 0): (705, 2429),
 (5, 1): (705, 2208),
 (5, 2): (705, 2208),
 (5, 3): (705, 589),
 (5, 4): (705, 589),
 (5, 5): (705, 3688),
 (5, 6): (705, 3688),
 (5, 7): (705, 3591),
 (5, 8): (705, 3591),
 (5, 9): (705, 370),
 (5, 10): (705, 2409),
 (5, 11): (705, 1223),
 (5, 12): (705, 516),
 (5, 13): (705, 2429),
 (6, 0): (705, 2429),
 (6, 1): (705, 2208),
 (6, 2): (705, 2208),
 (6, 3): (705, 589),
 (6, 4): (705, 589),
 (6, 5): (705, 1029),
 (6, 6): (705, 1029),
 (6, 7): (705, 3059),
 (6, 8): (705, 370),
 (6, 9): (705, 2409),
 (6, 10): (705, 2409),
 (6, 11): (705, 2409),
 (6, 12): (705, 516),
 (6, 13): (705, 2429),
 (7, 0): (302, 2429),
 (7, 1): (1389, 1389),
 (7, 2): (705, 3715),
 (7, 3): (705, 589),
 (7, 4): (705, 589),
 (7, 5): (705, 1029),
 (7, 6): (705, 1029),
 (7, 7): (705, 1029),
 (7, 8): (705, 1029),
 (7, 9): (705, 370),
 (7, 10): (705, 2409),
 (7, 11): (705, 516),
 (7, 12): (3435, 516),
 (7, 13): (516, 2429),
 (8, 0): (3435, 2429),
 (8, 1): (3435, 1389),
 (8, 2): (3435, 589),
 (8, 3): (3435, 589),
 (8, 4): (3435, 589),
 (8, 5): (1029, 1029),
 (8, 6): (1029, 1029),
 (8, 7): (3435, 1029),
 (8, 8): (3435, 1029),
 (8, 9): (3435, 1029),
 (8, 10): (3435, 2409),
 (8, 11): (3435, 516),
 (8, 12): (3435, 516),
 (8, 13): (516, 2429),
 (9, 0): (1389, 2429),
 (9, 1): (1389, 1389),
 (9, 2): (3435, 589),
 (9, 3): (3435, 589),
 (9, 4): (3435, 589),
 (9, 5): (1029, 1029),
 (9, 6): (1029, 1029),
 (9, 7): (1029, 3059),
 (9, 8): (1197, 1029),
 (9, 9): (1197, 1197),
 (9, 10): (1197, 1197),
 (9, 11): (516, 516),
 (9, 12): (3435, 516),
 (9, 13): (516, 2429),
 (10, 0): (1197, 2429),
 (10, 1): (1389, 1389),
 (10, 2): (589, 589),
 (10, 3): (1197, 589),
 (10, 4): (1197, 589),
 (10, 5): (1197, 1029),
 (10, 6): (1197, 1197),
 (10, 7): (1197, 3059),
 (10, 8): (1197, 1197),
 (10, 9): (1197, 670),
 (10, 10): (1197, 2409),
 (10, 11): (1197, 516),
 (10, 12): (1197, 516),
 (10, 13): (516, 2429),
 (11, 0): (1197, 2429),
 (11, 1): (3388, 1389),
 (11, 2): (1197, 589),
 (11, 3): (1197, 589),
 (11, 4): (1197, 589),
 (11, 5): (1197, 1197),
 (11, 6): (1197, 1197),
 (11, 7): (1197, 3059),
 (11, 8): (1197, 3059),
 (11, 9): (1197, 670),
 (11, 10): (1197, 516),
 (11, 11): (1197, 516),
 (11, 12): (1197, 516),
 (11, 13): (1197, 516),
 (12, 0): (2974, 2429),
 (12, 1): (2974, 1389),
 (12, 2): (2974, 589),
 (12, 3): (2974, 589),
 (12, 4): (2974, 589),
 (12, 5): (2974, 1197),
 (12, 6): (2974, 3324),
 (12, 7): (2974, 3059),
 (12, 8): (2974, 3059),
 (12, 9): (2974, 370),
 (12, 10): (2974, 670),
 (12, 11): (2974, 516),
 (12, 12): (2974, 516),
 (12, 13): (2974, 2429),
 (13, 0): (2974, 2429),
 (13, 1): (2974, 1389),
 (13, 2): (2974, 2208),
 (13, 3): (2974, 589),
 (13, 4): (2974, 589),
 (13, 5): (2974, 1029),
 (13, 6): (2974, 1029),
 (13, 7): (2974, 1029),
 (13, 8): (2974, 3059),
 (13, 9): (2974, 670),
 (13, 10): (2974, 670),
 (13, 11): (2974, 516),
 (13, 12): (2974, 516),
 (13, 13): (2974, 516)}