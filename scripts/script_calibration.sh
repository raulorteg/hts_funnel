echo "#### SCRIPT for CALIBRATION ####"
echo "-	Used to run the half-pipeline on conditions between HOMO-LUMO [1eV-5.5eV] in steps of 0.1 eV"
echo "-	For every HOMO-LUMO condition the funnel produces 10 molecules, 10 times, resulting in 100 molecules per 0.1 eV step"
echo "-	The output of this script is dumped in <calibration_script_output.txt>"
echo "----------------------------------------------------------------------------------"
for x in {10..55}; do
	y=`bc <<< "scale=1; $x/10"`
	echo $y
	echo "----------------------------------------------------------------------------------"
	python calibration.py $y

done
