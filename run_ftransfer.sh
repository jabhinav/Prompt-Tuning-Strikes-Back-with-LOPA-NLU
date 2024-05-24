scp -r -i ~/.ssh/id_rsa ./*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/
scp -r -i ~/.ssh/id_rsa ./*.sh abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/
scp -r -i ~/.ssh/id_rsa ./utils/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/utils/
scp -r -i ~/.ssh/id_rsa ./trainers/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/trainers/

# Uncomment to update the custom peft scripts
scp -r -i ~/.ssh/id_rsa ./custom_peft/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/custom_peft/
scp -r -i ~/.ssh/id_rsa ./custom_peft/utils/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/NLU/custom_peft/utils/
