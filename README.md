# Predicting Superconductivity Critical Temperature
The program predicts the critical temperatures of a superconductor, given its certain features (atomic mass, atomic radius, thermal conductivity, etc.)
<br/>
It cleans the original dataset stored in "org_data.csv" and uses Polynomial Regression with Ridge to train the model.
Resultant is a training accuracy of 84% and an RMSE of 13.77K.

# Running the Program
This program uses python3 and is made to be run on Linux systems. 
The following libraries need to be installed before execution:
- numpy
- pandas
- matplotlib
- sklearn
- seaborn

# Commented Code in the Program
A lot of the code has been commented out due to requirements from the coursework supervisor.
They involve figures, different subsets of data and different training algorithms with respect to these subsets.