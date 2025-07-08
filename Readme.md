


estimate price program

estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)


1st program: training program.
The training program read the data set, train the model on it, perform a linear regression with the datas,
Once completed, store the variables theta0 and theta1


step 1:
Parse the .csv

step 2:
Normalize the datas


We can simply use open or the library csv, but the library panda does a lot for us since it converts automatically into an array the values



installation:
source venv/bin/activate
pip install -r requirements.txt
rm theta/theta.csv

faire makefile pour rm le precedent modele
