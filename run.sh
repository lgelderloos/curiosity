for s in 123 234 345 456 567 678 789 890 901 12 23 34 45 56 67 78 89 90 1 100
do
    python3 -u unsupervised.py 40 .001 curious $s > outfiles/unsupervised/lr.001_curious_$s.txt
    python3 -u unsupervised.py 40 .001 random $s > outfiles/unsupervised/lr.001_random_$s.txt
    python3 -u unsupervised.py 40 .001 sn $s > outfiles/unsupervised/lr.001_sn_$s.txt
    python3 -u unsupervised.py 40 .001 plasticity $s > outfiles/unsupervised/lr.001_plasticity_$s.txt
done
