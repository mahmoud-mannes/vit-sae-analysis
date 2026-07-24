
This is essentially just a dump of the top positional feature candidates based on token selectivity 
(comparing activations of every feature at each position from 0 to 196 to activations at other positions), then sorting the top ones from those based on row/column selectivity. Note that while the word "selectivity" is used twice here, the metrics are fundamentally different. Token selectivity compares the mean activations of the features at one specific location to their mean activations at other locations, meanwhile row/column selectivity metrics extract the heatmap of when the features activate most, and calculate whether the activations are more concentrated in a specific row or column than others. 
It's important to note that values greater than 3 suggest decent row/column selectivity, and values greater than 5 indicate great row/column selectivity. You can also visualize these features using modules from the SAE_feature_analysis directory of the repo.
Another important note is that the current row/column selectivity metric will be replaced by an entropy-based one, the current way the metric is calculated (which can be found in the repo) is simple, but temporary and imperfect.

----- ROW 0 COLUMN 0 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 1 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 2 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 3 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 4 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 5 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 6 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 7 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 8 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 9 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 10 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 11 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 12 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 0 COLUMN 13 -----
Feature 306, Column selectivity 1.464734673500061
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 0 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 1 COLUMN 1 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 1 COLUMN 2 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 3 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 4 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 5 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 6 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 7 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 8 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 9 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 10 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 11 -----
Feature 306, Column selectivity 1.464734673500061
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 12 -----
Feature 1214, Column selectivity 1.2731194496154785
Feature 277, Row selectivity 2.7244558334350586
----- ROW 1 COLUMN 13 -----
Feature 306, Column selectivity 1.464734673500061
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 2 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 2 COLUMN 1 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 2 COLUMN 2 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 3 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 2 COLUMN 4 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 5 -----
Feature 1241, Column selectivity 1.2009814977645874
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 6 -----
Feature 1241, Column selectivity 1.2009814977645874
Feature 1241, Row selectivity 1.3568304777145386
----- ROW 2 COLUMN 7 -----
Feature 1241, Column selectivity 1.2009814977645874
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 8 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 9 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 10 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 11 -----
Feature 2948, Column selectivity 1.120833158493042
Feature 277, Row selectivity 2.7244558334350586
----- ROW 2 COLUMN 12 -----
Feature 2662, Column selectivity 1.4854766130447388
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 2 COLUMN 13 -----
Feature 306, Column selectivity 1.464734673500061
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 3 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 3 COLUMN 1 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 3 COLUMN 2 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 3 COLUMN 3 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 3 COLUMN 4 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 3 COLUMN 5 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 3 COLUMN 6 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 376, Row selectivity 1.2052890062332153
----- ROW 3 COLUMN 7 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 376, Row selectivity 1.2052890062332153
----- ROW 3 COLUMN 8 -----
Feature 1241, Column selectivity 1.2009814977645874
Feature 1241, Row selectivity 1.3568304777145386
----- ROW 3 COLUMN 9 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 3 COLUMN 10 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 277, Row selectivity 2.7244558334350586
----- ROW 3 COLUMN 11 -----
Feature 2920, Column selectivity 1.4357810020446777
Feature 277, Row selectivity 2.7244558334350586
----- ROW 3 COLUMN 12 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 306, Row selectivity 1.2910348176956177
----- ROW 3 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 4 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 4 COLUMN 1 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 4 COLUMN 2 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 4 COLUMN 3 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 376, Row selectivity 1.2052890062332153
----- ROW 4 COLUMN 4 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 4 COLUMN 5 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 1496, Row selectivity 1.2880144119262695
----- ROW 4 COLUMN 6 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 376, Row selectivity 1.2052890062332153
----- ROW 4 COLUMN 7 -----
Feature 2009, Column selectivity 1.094326138496399
Feature 1551, Row selectivity 1.1705424785614014
----- ROW 4 COLUMN 8 -----
Feature 2082, Column selectivity 1.2246257066726685
Feature 2948, Row selectivity 1.1064494848251343
----- ROW 4 COLUMN 9 -----
Feature 1947, Column selectivity 1.2895078659057617
Feature 1496, Row selectivity 1.2880144119262695
----- ROW 4 COLUMN 10 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 376, Row selectivity 1.2052890062332153
----- ROW 4 COLUMN 11 -----
Feature 2948, Column selectivity 1.120833158493042
Feature 3549, Row selectivity 1.17597234249115
----- ROW 4 COLUMN 12 -----
Feature 1059, Column selectivity 1.401252269744873
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 4 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 1214, Row selectivity 1.6990314722061157
----- ROW 5 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 5 COLUMN 1 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 5 COLUMN 2 -----
Feature 1843, Column selectivity 1.6374558210372925
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 5 COLUMN 3 -----
Feature 1225, Column selectivity 1.1785231828689575
Feature 56, Row selectivity 1.1867930889129639
----- ROW 5 COLUMN 4 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 5 COLUMN 5 -----
Feature 1897, Column selectivity 1.108432650566101
Feature 1551, Row selectivity 1.1705424785614014
----- ROW 5 COLUMN 6 -----
Feature 1273, Column selectivity 1.1547995805740356
Feature 56, Row selectivity 1.1867930889129639
----- ROW 5 COLUMN 7 -----
Feature 2948, Column selectivity 1.120833158493042
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 5 COLUMN 8 -----
Feature 56, Column selectivity 1.110121726989746
Feature 56, Row selectivity 1.1867930889129639
----- ROW 5 COLUMN 9 -----
Feature 2479, Column selectivity 1.1285823583602905
Feature 3602, Row selectivity 1.3528716564178467
----- ROW 5 COLUMN 10 -----
Feature 2009, Column selectivity 1.094326138496399
Feature 2891, Row selectivity 1.3421576023101807
----- ROW 5 COLUMN 11 -----
Feature 3650, Column selectivity 1.1419947147369385
Feature 56, Row selectivity 1.1867930889129639
----- ROW 5 COLUMN 12 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 5 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 306, Row selectivity 1.2910348176956177
----- ROW 6 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 6 COLUMN 1 -----
Feature 3499, Column selectivity 1.35798180103302
Feature 56, Row selectivity 1.1867930889129639
----- ROW 6 COLUMN 2 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 6 COLUMN 3 -----
Feature 1225, Column selectivity 1.1785231828689575
Feature 56, Row selectivity 1.1867930889129639
----- ROW 6 COLUMN 4 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 6 COLUMN 5 -----
Feature 1760, Column selectivity 1.106044888496399
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 6 COLUMN 6 -----
Feature 1897, Column selectivity 1.108432650566101
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 6 COLUMN 7 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 2341, Row selectivity 1.1293658018112183
----- ROW 6 COLUMN 8 -----
Feature 2479, Column selectivity 1.1285823583602905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 6 COLUMN 9 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 376, Row selectivity 1.2052890062332153
----- ROW 6 COLUMN 10 -----
Feature 2948, Column selectivity 1.120833158493042
Feature 148, Row selectivity 1.2229259014129639
----- ROW 6 COLUMN 11 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 6 COLUMN 12 -----
Feature 2662, Column selectivity 1.4854766130447388
Feature 306, Row selectivity 1.2910348176956177
----- ROW 6 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 7 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 7 COLUMN 1 -----
Feature 474, Column selectivity 1.5390554666519165
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 7 COLUMN 2 -----
Feature 2640, Column selectivity 1.3982568979263306
Feature 2640, Row selectivity 1.1874217987060547
----- ROW 7 COLUMN 3 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 2547, Row selectivity 1.2768301963806152
----- ROW 7 COLUMN 4 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 7 COLUMN 5 -----
Feature 1897, Column selectivity 1.108432650566101
Feature 3244, Row selectivity 1.1132168769836426
----- ROW 7 COLUMN 6 -----
Feature 1273, Column selectivity 1.1547995805740356
Feature 56, Row selectivity 1.1867930889129639
----- ROW 7 COLUMN 7 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 7 COLUMN 8 -----
Feature 56, Column selectivity 1.110121726989746
Feature 56, Row selectivity 1.1867930889129639
----- ROW 7 COLUMN 9 -----
Feature 2479, Column selectivity 1.1285823583602905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 7 COLUMN 10 -----
Feature 376, Column selectivity 1.2165753841400146
Feature 376, Row selectivity 1.2052890062332153
----- ROW 7 COLUMN 11 -----
Feature 2662, Column selectivity 1.4854766130447388
Feature 56, Row selectivity 1.1867930889129639
----- ROW 7 COLUMN 12 -----
Feature 2920, Column selectivity 1.4357810020446777
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 7 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 8 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 8 COLUMN 1 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 474, Row selectivity 1.470920443534851
----- ROW 8 COLUMN 2 -----
Feature 474, Column selectivity 1.5390554666519165
Feature 474, Row selectivity 1.470920443534851
----- ROW 8 COLUMN 3 -----
Feature 3499, Column selectivity 1.35798180103302
Feature 1897, Row selectivity 1.0901869535446167
----- ROW 8 COLUMN 4 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 8 COLUMN 5 -----
Feature 1273, Column selectivity 1.1547995805740356
Feature 2479, Row selectivity 1.1461671590805054
----- ROW 8 COLUMN 6 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 2547, Row selectivity 1.2768301963806152
----- ROW 8 COLUMN 7 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 8 COLUMN 8 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 8 COLUMN 9 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 2341, Row selectivity 1.1293658018112183
----- ROW 8 COLUMN 10 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 148, Row selectivity 1.2229259014129639
----- ROW 8 COLUMN 11 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 8 COLUMN 12 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 8 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 9 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1843, Row selectivity 1.4120453596115112
----- ROW 9 COLUMN 1 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 9 COLUMN 2 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 9 COLUMN 3 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 9 COLUMN 4 -----
Feature 2479, Column selectivity 1.1285823583602905
Feature 2479, Row selectivity 1.1461671590805054
----- ROW 9 COLUMN 5 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 9 COLUMN 6 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 9 COLUMN 7 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 56, Row selectivity 1.1867930889129639
----- ROW 9 COLUMN 8 -----
Feature 2695, Column selectivity 1.1200724840164185
Feature 56, Row selectivity 1.1867930889129639
----- ROW 9 COLUMN 9 -----
Feature 1897, Column selectivity 1.108432650566101
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 9 COLUMN 10 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 9 COLUMN 11 -----
Feature 2662, Column selectivity 1.4854766130447388
Feature 1713, Row selectivity 1.2717781066894531
----- ROW 9 COLUMN 12 -----
Feature 3494, Column selectivity 1.641731858253479
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 9 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 10 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 96, Row selectivity 1.4848167896270752
----- ROW 10 COLUMN 1 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 10 COLUMN 2 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 3667, Row selectivity 1.3945913314819336
----- ROW 10 COLUMN 3 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 10 COLUMN 4 -----
Feature 2479, Column selectivity 1.1285823583602905
Feature 1451, Row selectivity 1.2054556608200073
----- ROW 10 COLUMN 5 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 2547, Row selectivity 1.2768301963806152
----- ROW 10 COLUMN 6 -----
Feature 794, Column selectivity 1.0674564838409424
Feature 2138, Row selectivity 1.187922477722168
----- ROW 10 COLUMN 7 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 10 COLUMN 8 -----
Feature 1897, Column selectivity 1.108432650566101
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 10 COLUMN 9 -----
Feature 1351, Column selectivity 1.1006604433059692
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 10 COLUMN 10 -----
Feature 2009, Column selectivity 1.094326138496399
Feature 2955, Row selectivity 1.1789666414260864
----- ROW 10 COLUMN 11 -----
Feature 56, Column selectivity 1.110121726989746
Feature 56, Row selectivity 1.1867930889129639
----- ROW 10 COLUMN 12 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 2547, Row selectivity 1.2768301963806152
----- ROW 10 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 11 COLUMN 0 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 96, Row selectivity 1.4848167896270752
----- ROW 11 COLUMN 1 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 96, Row selectivity 1.4848167896270752
----- ROW 11 COLUMN 2 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 11 COLUMN 3 -----
Feature 602, Column selectivity 1.8736940622329712
Feature 3667, Row selectivity 1.3945913314819336
----- ROW 11 COLUMN 4 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 11 COLUMN 5 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 11 COLUMN 6 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 11 COLUMN 7 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 2547, Row selectivity 1.2768301963806152
----- ROW 11 COLUMN 8 -----
Feature 2009, Column selectivity 1.094326138496399
Feature 494, Row selectivity 2.151524782180786
----- ROW 11 COLUMN 9 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 11 COLUMN 10 -----
Feature 2782, Column selectivity 1.1193050146102905
Feature 2009, Row selectivity 1.1384477615356445
----- ROW 11 COLUMN 11 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 494, Row selectivity 2.151524782180786
----- ROW 11 COLUMN 12 -----
Feature 2664, Column selectivity 1.7588090896606445
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 11 COLUMN 13 -----
Feature 2394, Column selectivity 2.10300350189209
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 12 COLUMN 0 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 12 COLUMN 1 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 12 COLUMN 2 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 3 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 4 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 5 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 6 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 7 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 8 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 9 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 10 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 11 -----
Feature 2547, Column selectivity 1.1522115468978882
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 12 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 494, Row selectivity 2.151524782180786
----- ROW 12 COLUMN 13 -----
Feature 3494, Column selectivity 1.641731858253479
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 13 COLUMN 0 -----
Feature 3667, Column selectivity 1.7317243814468384
Feature 1899, Row selectivity 1.5529612302780151
----- ROW 13 COLUMN 1 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 2 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 3 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 4 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 5 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 6 -----
Feature 1899, Column selectivity 1.2283687591552734
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 7 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 8 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 9 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 10 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 11 -----
Feature 1713, Column selectivity 1.3463491201400757
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 12 -----
Feature 96, Column selectivity 1.3775701522827148
Feature 494, Row selectivity 2.151524782180786
----- ROW 13 COLUMN 13 -----
Feature 3494, Column selectivity 1.641731858253479
Feature 3494, Row selectivity 1.5060908794403076
----- ROW 14 COLUMN 0 -----
Feature 2, Column selectivity 3.4629268646240234
Feature 2, Row selectivity 4.576295852661133


DICT OF TOP SELECTIVITY: 

{(0, 0): (277, 1843),
 (0, 1): (277, 1214),
 (0, 2): (277, 1214),
 (0, 3): (277, 1214),
 (0, 4): (277, 1214),
 (0, 5): (277, 1214),
 (0, 6): (277, 1214),
 (0, 7): (277, 1214),
 (0, 8): (277, 1214),
 (0, 9): (277, 1214),
 (0, 10): (277, 1214),
 (0, 11): (277, 1214),
 (0, 12): (277, 1214),
 (0, 13): (277, 306),
 (1, 0): (1214, 3667),
 (1, 1): (1214, 1843),
 (1, 2): (277, 1843),
 (1, 3): (277, 2640),
 (1, 4): (277, 1843),
 (1, 5): (277, 1843),
 (1, 6): (277, 1843),
 (1, 7): (277, 1843),
 (1, 8): (277, 1843),
 (1, 9): (277, 1214),
 (1, 10): (277, 1214),
 (1, 11): (277, 306),
 (1, 12): (277, 1214),
 (1, 13): (1214, 306),
 (2, 0): (1843, 602),
 (2, 1): (1843, 1843),
 (2, 2): (277, 2640),
 (2, 3): (1843, 1843),
 (2, 4): (277, 2640),
 (2, 5): (277, 1241),
 (2, 6): (1241, 1241),
 (2, 7): (277, 1241),
 (2, 8): (277, 376),
 (2, 9): (277, 1843),
 (2, 10): (277, 376),
 (2, 11): (277, 2948),
 (2, 12): (1214, 2662),
 (2, 13): (1214, 306),
 (3, 0): (1843, 602),
 (3, 1): (1843, 1843),
 (3, 2): (2640, 2640),
 (3, 3): (2640, 2640),
 (3, 4): (1843, 1843),
 (3, 5): (56, 2782),
 (3, 6): (376, 2640),
 (3, 7): (376, 2640),
 (3, 8): (1241, 1241),
 (3, 9): (56, 2782),
 (3, 10): (277, 376),
 (3, 11): (277, 2920),
 (3, 12): (306, 2394),
 (3, 13): (1214, 2394),
 (4, 0): (1843, 602),
 (4, 1): (1843, 1843),
 (4, 2): (1843, 602),
 (4, 3): (376, 2640),
 (4, 4): (2640, 2640),
 (4, 5): (1496, 2640),
 (4, 6): (376, 376),
 (4, 7): (1551, 2009),
 (4, 8): (2948, 2082),
 (4, 9): (1496, 1947),
 (4, 10): (376, 376),
 (4, 11): (3549, 2948),
 (4, 12): (1214, 1059),
 (4, 13): (1214, 2394),
 (5, 0): (1843, 602),
 (5, 1): (1843, 3667),
 (5, 2): (1843, 1843),
 (5, 3): (56, 1225),
 (5, 4): (56, 2782),
 (5, 5): (1551, 1897),
 (5, 6): (56, 1273),
 (5, 7): (2009, 2948),
 (5, 8): (56, 56),
 (5, 9): (3602, 2479),
 (5, 10): (2891, 2009),
 (5, 11): (56, 3650),
 (5, 12): (1713, 2394),
 (5, 13): (306, 2394),
 (6, 0): (1843, 602),
 (6, 1): (56, 3499),
 (6, 2): (2640, 2640),
 (6, 3): (56, 1225),
 (6, 4): (2640, 2640),
 (6, 5): (2009, 1760),
 (6, 6): (2009, 1897),
 (6, 7): (2341, 2782),
 (6, 8): (56, 2479),
 (6, 9): (376, 376),
 (6, 10): (148, 2948),
 (6, 11): (1713, 1713),
 (6, 12): (306, 2662),
 (6, 13): (3494, 2394),
 (7, 0): (1843, 602),
 (7, 1): (1899, 474),
 (7, 2): (2640, 2640),
 (7, 3): (2547, 2547),
 (7, 4): (56, 2782),
 (7, 5): (3244, 1897),
 (7, 6): (56, 1273),
 (7, 7): (2009, 2782),
 (7, 8): (56, 56),
 (7, 9): (56, 2479),
 (7, 10): (376, 376),
 (7, 11): (56, 2662),
 (7, 12): (1713, 2920),
 (7, 13): (3494, 2394),
 (8, 0): (1843, 602),
 (8, 1): (474, 3667),
 (8, 2): (474, 474),
 (8, 3): (1897, 3499),
 (8, 4): (56, 2782),
 (8, 5): (2479, 1273),
 (8, 6): (2547, 2547),
 (8, 7): (56, 2782),
 (8, 8): (56, 2782),
 (8, 9): (2341, 2782),
 (8, 10): (148, 2782),
 (8, 11): (1713, 1713),
 (8, 12): (1713, 1713),
 (8, 13): (3494, 2394),
 (9, 0): (1843, 602),
 (9, 1): (1899, 602),
 (9, 2): (1899, 602),
 (9, 3): (1899, 1899),
 (9, 4): (2479, 2479),
 (9, 5): (56, 2782),
 (9, 6): (2009, 2782),
 (9, 7): (56, 2782),
 (9, 8): (56, 2695),
 (9, 9): (2009, 1897),
 (9, 10): (1713, 1713),
 (9, 11): (1713, 2662),
 (9, 12): (3494, 3494),
 (9, 13): (3494, 2394),
 (10, 0): (96, 602),
 (10, 1): (1899, 3667),
 (10, 2): (3667, 3667),
 (10, 3): (1899, 602),
 (10, 4): (1451, 2479),
 (10, 5): (2547, 2547),
 (10, 6): (2138, 794),
 (10, 7): (1899, 1899),
 (10, 8): (2009, 1897),
 (10, 9): (2009, 1351),
 (10, 10): (2955, 2009),
 (10, 11): (56, 56),
 (10, 12): (2547, 1713),
 (10, 13): (3494, 2394),
 (11, 0): (96, 602),
 (11, 1): (96, 3667),
 (11, 2): (1899, 3667),
 (11, 3): (3667, 602),
 (11, 4): (1899, 1899),
 (11, 5): (1899, 1899),
 (11, 6): (494, 1899),
 (11, 7): (2547, 2547),
 (11, 8): (494, 2009),
 (11, 9): (1899, 1899),
 (11, 10): (2009, 2782),
 (11, 11): (494, 1713),
 (11, 12): (3494, 2664),
 (11, 13): (3494, 2394),
 (12, 0): (1899, 3667),
 (12, 1): (1899, 3667),
 (12, 2): (494, 96),
 (12, 3): (494, 3667),
 (12, 4): (494, 1899),
 (12, 5): (494, 3667),
 (12, 6): (494, 2547),
 (12, 7): (494, 1899),
 (12, 8): (494, 1899),
 (12, 9): (494, 1899),
 (12, 10): (494, 1899),
 (12, 11): (494, 2547),
 (12, 12): (494, 1713),
 (12, 13): (3494, 3494),
 (13, 0): (1899, 3667),
 (13, 1): (494, 96),
 (13, 2): (494, 96),
 (13, 3): (494, 96),
 (13, 4): (494, 96),
 (13, 5): (494, 96),
 (13, 6): (494, 1899),
 (13, 7): (494, 96),
 (13, 8): (494, 96),
 (13, 9): (494, 96),
 (13, 10): (494, 96),
 (13, 11): (494, 1713),
 (13, 12): (494, 96),
 (13, 13): (3494, 3494),
 (14, 0): (2, 2)}