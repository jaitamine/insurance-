# insurance-
This repository contains a simple algorithm to predict the cost of health insurance.It was implemented using Scikit-learn.

Weka was used to discriminate what algorithm and model is most efficient. It turned out that multiple linear regression is relatively the simplest and fastest to implement, although the multilayer perceptron yielded the lowest mean absolute error.Also the variable "region" was disgarded as it didn't change the mean absolute error, and the difference in cost per region is negligible.

In this repository you'll find the python file format and the jupyter notebook's.

In this branch I added some polynomial features to the 2nd degree, and also scaled them.
The absolte standard error came down to ≈2900$
